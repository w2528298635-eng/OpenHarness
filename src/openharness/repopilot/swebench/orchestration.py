from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .adapters import EvaluationArm
from .inference import verify_artifact_seal
from .models import SampleManifest


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_FAILED = "agent_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


class RunKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    arm: EvaluationArm
    repetition: int = Field(ge=1)

    @property
    def value(self) -> str:
        return f"{self.instance_id}::{self.arm.value}::{self.repetition}"


class RunCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_path: str
    artifact_sha256: str
    resolved: bool


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: RunKey
    status: RunStatus = RunStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    infrastructure_retries: int = Field(default=0, ge=0)
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    resolved: bool | None = None
    error: str | None = None
    transitions: tuple[RunStatus, ...] = (RunStatus.PENDING,)


class ExperimentCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    manifest_sha256: str
    records: dict[str, RunRecord] = {}

    def record_for(self, key: RunKey) -> RunRecord:
        return self.records.get(key.value, RunRecord(key=key))

    def with_record(self, record: RunRecord) -> ExperimentCheckpoint:
        records = dict(self.records)
        records[record.key.value] = record
        return self.model_copy(update={"records": records})

    def with_completion(
        self,
        key: RunKey,
        completion: RunCompletion,
    ) -> ExperimentCheckpoint:
        previous = self.record_for(key)
        record = previous.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "artifact_path": completion.artifact_path,
                "artifact_sha256": completion.artifact_sha256,
                "resolved": completion.resolved,
                "error": None,
                "transitions": (*previous.transitions, RunStatus.COMPLETED),
            }
        )
        return self.with_record(record)


class CheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def create(self, manifest: SampleManifest) -> ExperimentCheckpoint:
        checkpoint = ExperimentCheckpoint(manifest_sha256=manifest.sha256)
        self.save(checkpoint)
        return checkpoint

    def load(self) -> ExperimentCheckpoint:
        return ExperimentCheckpoint.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )

    def save(self, checkpoint: ExperimentCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(checkpoint.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


class InfrastructureRunError(RuntimeError):
    pass


class AgentRunError(RuntimeError):
    pass


RunWorker = Callable[[RunKey], Awaitable[RunCompletion]]


def build_run_keys(
    manifest: SampleManifest,
    arms: Sequence[EvaluationArm],
    *,
    repetitions: int,
) -> list[RunKey]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return [
        RunKey(
            instance_id=instance.instance_id,
            arm=arm,
            repetition=repetition,
        )
        for instance in manifest.instances
        for arm in arms
        for repetition in range(1, repetitions + 1)
    ]


def _valid_completion(record: RunRecord) -> bool:
    if record.status is not RunStatus.COMPLETED or record.artifact_path is None:
        return False
    path = Path(record.artifact_path)
    if not verify_artifact_seal(path):
        return False
    if record.artifact_sha256 is None:
        return False
    actual = path.with_suffix(".json.sha256").read_text(encoding="ascii").strip()
    return actual == record.artifact_sha256


class ExperimentOrchestrator:
    def __init__(
        self,
        store: CheckpointStore,
        *,
        max_infrastructure_retries: int,
    ):
        if max_infrastructure_retries < 0:
            raise ValueError("max infrastructure retries must not be negative")
        self.store = store
        self.max_infrastructure_retries = max_infrastructure_retries

    async def run(
        self,
        keys: Sequence[RunKey],
        worker: RunWorker,
    ) -> ExperimentCheckpoint:
        checkpoint = self.store.load()
        for key in keys:
            current = checkpoint.record_for(key)
            if _valid_completion(current):
                continue
            retries = current.infrastructure_retries
            while True:
                running = current.model_copy(
                    update={
                        "status": RunStatus.RUNNING,
                        "attempts": current.attempts + 1,
                        "error": None,
                        "transitions": (*current.transitions, RunStatus.RUNNING),
                    }
                )
                checkpoint = checkpoint.with_record(running)
                self.store.save(checkpoint)
                try:
                    completion = await worker(key)
                    if not verify_artifact_seal(Path(completion.artifact_path)):
                        raise InfrastructureRunError(
                            "worker returned an unsealed inference artifact"
                        )
                except InfrastructureRunError as exc:
                    retries += 1
                    if retries <= self.max_infrastructure_retries:
                        current = running.model_copy(
                            update={
                                "infrastructure_retries": retries,
                                "error": str(exc),
                            }
                        )
                        checkpoint = checkpoint.with_record(current)
                        self.store.save(checkpoint)
                        continue
                    failed = running.model_copy(
                        update={
                            "status": RunStatus.INFRASTRUCTURE_FAILED,
                            "infrastructure_retries": retries - 1,
                            "error": str(exc),
                            "transitions": (
                                *running.transitions,
                                RunStatus.INFRASTRUCTURE_FAILED,
                            ),
                        }
                    )
                    checkpoint = checkpoint.with_record(failed)
                except AgentRunError as exc:
                    failed = running.model_copy(
                        update={
                            "status": RunStatus.AGENT_FAILED,
                            "error": str(exc),
                            "transitions": (
                                *running.transitions,
                                RunStatus.AGENT_FAILED,
                            ),
                        }
                    )
                    checkpoint = checkpoint.with_record(failed)
                else:
                    checkpoint = checkpoint.with_completion(key, completion)
                self.store.save(checkpoint)
                break
        return checkpoint

