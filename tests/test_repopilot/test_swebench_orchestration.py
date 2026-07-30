from __future__ import annotations

from pathlib import Path

import pytest

from openharness.repopilot.swebench.adapters import EvaluationArm
from openharness.repopilot.swebench.inference import InferenceArtifact
from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
    SampleManifest,
    SamplingConfig,
)
from openharness.repopilot.swebench.orchestration import (
    AgentRunError,
    CheckpointStore,
    ExperimentOrchestrator,
    InfrastructureRunError,
    RunCompletion,
    RunKey,
    RunStatus,
    build_run_keys,
)


def _manifest() -> SampleManifest:
    config = SamplingConfig(easy=1, medium=0, hard=0, seed=1)
    instance = PublicInstance(
        instance_id="django__django-1",
        repo="django/django",
        base_commit="abc",
        problem_statement="Fix issue.",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )
    return SampleManifest(
        dataset_name=config.dataset_name,
        dataset_revision="revision",
        sampling=config,
        instances=(instance,),
        sha256="a" * 64,
    )


def _write_sealed_artifact(directory: Path, name: str = "artifact") -> Path:
    import hashlib

    path = directory / f"{name}.json"
    model_patch = "diff --git a/a.py b/a.py\n"
    artifact = InferenceArtifact(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
        run_id="run-1",
        status="completed",
        provider="deepseek",
        model="deepseek-v4-flash",
        retrieval_enabled=False,
        model_patch=model_patch,
        model_patch_sha256=hashlib.sha256(model_patch.encode()).hexdigest(),
        duration_seconds=1,
        input_tokens=1,
        output_tokens=1,
        cache_hit_tokens=0,
        model_calls=1,
    )
    payload = (artifact.model_dump_json(indent=2) + "\n").encode()
    path.write_bytes(payload)
    path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    return path


def test_build_run_keys_is_stable_and_covers_every_repetition() -> None:
    keys = build_run_keys(
        _manifest(),
        (EvaluationArm.NATIVE, EvaluationArm.LEGACY),
        repetitions=3,
    )

    assert len(keys) == 6
    assert keys[0].value == "django__django-1::native_openharness::1"
    assert keys[-1].value == "django__django-1::legacy_repopilot::3"


def test_checkpoint_store_writes_atomically(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(_manifest())

    loaded = store.load()

    assert loaded == checkpoint
    assert not list(tmp_path.glob("*.tmp"))


def test_inference_completion_remains_pending_until_official_evaluation(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(_manifest())
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    artifact_path = _write_sealed_artifact(tmp_path)

    checkpoint = checkpoint.with_completion(
        key,
        RunCompletion(
            artifact_path=str(artifact_path),
            artifact_sha256=artifact_path.with_suffix(
                ".json.sha256"
            ).read_text().strip(),
            resolved=None,
        ),
    )

    record = checkpoint.records[key.value]
    assert record.status is RunStatus.EVALUATION_PENDING
    assert record.resolved is None

    evaluated = checkpoint.with_evaluation(key, resolved=True)

    assert evaluated.records[key.value].status is RunStatus.COMPLETED
    assert evaluated.records[key.value].resolved is True


@pytest.mark.asyncio
async def test_resume_does_not_repeat_sealed_inference_awaiting_evaluation(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(_manifest())
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    artifact_path = _write_sealed_artifact(tmp_path)
    checkpoint = checkpoint.with_completion(
        key,
        RunCompletion(
            artifact_path=str(artifact_path),
            artifact_sha256=artifact_path.with_suffix(
                ".json.sha256"
            ).read_text().strip(),
            resolved=None,
        ),
    )
    store.save(checkpoint)
    calls = 0

    async def worker(run_key: RunKey) -> RunCompletion:
        nonlocal calls
        calls += 1
        raise AssertionError("sealed inference must not be repeated")

    result = await ExperimentOrchestrator(
        store,
        max_infrastructure_retries=1,
    ).run([key], worker)

    assert calls == 0
    assert result.records[key.value].status is RunStatus.EVALUATION_PENDING


@pytest.mark.asyncio
async def test_orchestrator_skips_only_completed_runs_with_valid_seals(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(_manifest())
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    artifact_path = _write_sealed_artifact(tmp_path)
    checkpoint = checkpoint.with_completion(
        key,
        RunCompletion(
            artifact_path=str(artifact_path),
            artifact_sha256=artifact_path.with_suffix(".json.sha256").read_text().strip(),
            resolved=True,
        ),
    )
    store.save(checkpoint)
    calls = 0

    async def worker(run_key: RunKey) -> RunCompletion:
        nonlocal calls
        calls += 1
        return RunCompletion(
            artifact_path=str(_write_sealed_artifact(tmp_path, "replacement")),
            artifact_sha256="b" * 64,
            resolved=True,
        )

    await ExperimentOrchestrator(store, max_infrastructure_retries=1).run(
        [key],
        worker,
    )

    assert calls == 0


@pytest.mark.asyncio
async def test_orchestrator_retries_infrastructure_failure_but_not_agent_failure(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.create(_manifest())
    infra_key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    agent_key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.LEGACY,
        repetition=1,
    )
    attempts: dict[str, int] = {}

    async def worker(key: RunKey) -> RunCompletion:
        attempts[key.value] = attempts.get(key.value, 0) + 1
        if key == infra_key and attempts[key.value] == 1:
            raise InfrastructureRunError("Docker network interrupted")
        if key == agent_key:
            raise AgentRunError("Agent produced no patch")
        artifact = _write_sealed_artifact(tmp_path, key.arm.value)
        return RunCompletion(
            artifact_path=str(artifact),
            artifact_sha256=artifact.with_suffix(".json.sha256").read_text().strip(),
            resolved=True,
        )

    result = await ExperimentOrchestrator(
        store,
        max_infrastructure_retries=1,
    ).run([infra_key, agent_key], worker)

    assert attempts[infra_key.value] == 2
    assert attempts[agent_key.value] == 1
    assert result.records[infra_key.value].status is RunStatus.COMPLETED
    assert result.records[infra_key.value].infrastructure_retries == 1
    assert result.records[agent_key.value].status is RunStatus.AGENT_FAILED
