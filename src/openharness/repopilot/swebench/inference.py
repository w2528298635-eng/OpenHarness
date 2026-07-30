from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adapters import AgentAdapter, EvaluationArm, InferenceBudget
from .localization import RetrievedLocation
from .models import PublicInstance


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance: PublicInstance
    arm: EvaluationArm
    repetition: int = Field(ge=1)
    budget: InferenceBudget


class InferenceArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    instance_id: str
    arm: EvaluationArm
    repetition: int
    run_id: str
    status: str
    provider: str
    model: str
    retrieval_enabled: bool
    model_patch: str
    model_patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int
    model_calls: int
    retrieval: tuple[RetrievedLocation, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def check_patch_digest(self) -> InferenceArtifact:
        actual = hashlib.sha256(self.model_patch.encode("utf-8")).hexdigest()
        if actual != self.model_patch_sha256:
            raise ValueError("model patch SHA-256 does not match artifact content")
        return self


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_run_name(request: InferenceRequest) -> str:
    safe_instance = request.instance.instance_id.replace("/", "__").replace("\\", "__")
    return f"{safe_instance}__{request.arm.value}__r{request.repetition}"


class InferenceRunner:
    async def run(
        self,
        *,
        request: InferenceRequest,
        adapter: AgentAdapter,
        repository_path: Path,
        output_directory: Path,
    ) -> Path:
        if adapter.config.arm is not request.arm:
            raise ValueError(
                f"request arm {request.arm.value} does not match "
                f"adapter arm {adapter.config.arm.value}"
            )
        outcome = await adapter.run(
            instance=request.instance,
            repository_path=repository_path,
            budget=request.budget,
        )
        patch_digest = hashlib.sha256(
            outcome.model_patch.encode("utf-8")
        ).hexdigest()
        artifact = InferenceArtifact(
            instance_id=request.instance.instance_id,
            arm=request.arm,
            repetition=request.repetition,
            run_id=outcome.run_id,
            status=outcome.status,
            provider=adapter.config.provider,
            model=adapter.config.model,
            retrieval_enabled=adapter.config.retrieval_enabled,
            model_patch=outcome.model_patch,
            model_patch_sha256=patch_digest,
            duration_seconds=outcome.duration_seconds,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cache_hit_tokens=outcome.cache_hit_tokens,
            model_calls=outcome.model_calls,
            retrieval=outcome.retrieval,
            error=outcome.error,
        )
        artifact_path = output_directory / f"{_safe_run_name(request)}.json"
        payload = (artifact.model_dump_json(indent=2) + "\n").encode("utf-8")
        _atomic_write(artifact_path, payload)
        seal = hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n"
        _atomic_write(artifact_path.with_suffix(".json.sha256"), seal)
        return artifact_path


def verify_artifact_seal(artifact_path: Path) -> bool:
    seal_path = artifact_path.with_suffix(".json.sha256")
    if not artifact_path.is_file() or not seal_path.is_file():
        return False
    expected = seal_path.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return expected == actual

