from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openharness.repopilot.swebench.adapters import EvaluationArm
from openharness.repopilot.swebench.docker_runner import HarnessResult
from openharness.repopilot.swebench.execution import (
    BatchHarnessEvaluator,
    evaluate_pending_matrix,
)
from openharness.repopilot.swebench.inference import InferenceArtifact
from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
    SampleManifest,
    SamplingConfig,
)
from openharness.repopilot.swebench.orchestration import (
    CheckpointStore,
    InfrastructureRunError,
    RunCompletion,
    RunKey,
    RunRecord,
    RunStatus,
)


def _manifest() -> SampleManifest:
    config = SamplingConfig(easy=1, medium=0, hard=0)
    return SampleManifest(
        dataset_name=config.dataset_name,
        dataset_revision="dataset-sha",
        sampling=config,
        instances=(
            PublicInstance(
                instance_id="django__django-1",
                repo="django/django",
                base_commit="a" * 40,
                problem_statement="Fix public issue.",
                source_difficulty="<15 min fix",
                difficulty=DifficultyStratum.EASY,
            ),
        ),
        sha256="a" * 64,
    )


def _sealed_artifact(path: Path) -> Path:
    model_patch = "diff --git a/a.py b/a.py\n"
    artifact = InferenceArtifact(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
        run_id="agent-run",
        status="completed",
        provider="deepseek",
        model="deepseek-v4-flash",
        retrieval_enabled=False,
        model_patch=model_patch,
        model_patch_sha256=hashlib.sha256(model_patch.encode()).hexdigest(),
        duration_seconds=1,
        input_tokens=10,
        output_tokens=2,
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


class RecordingHarness:
    def __init__(self):
        self.calls: list[dict] = []

    def evaluate(self, **kwargs) -> HarnessResult:
        self.calls.append(kwargs)
        return HarnessResult(
            status="completed",
            total=1,
            submitted=1,
            completed=1,
            resolved=1,
            resolved_instance_ids=("django__django-1",),
        )


def test_batch_evaluator_writes_unique_predictions_and_completes_checkpoint(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    artifact_path = _sealed_artifact(tmp_path / "artifact.json")
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(manifest).with_completion(
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
    harness = RecordingHarness()

    result = BatchHarnessEvaluator(
        checkpoint_store=store,
        harness=harness,
        dataset_name=manifest.dataset_name,
        output_directory=tmp_path / "evaluation",
    ).evaluate([key])

    prediction_path = Path(harness.calls[0]["predictions_path"])
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["instance_id"] == "django__django-1"
    assert prediction["model_patch"].startswith("diff --git")
    assert harness.calls[0]["instance_ids"] == ("django__django-1",)
    assert harness.calls[0]["clean"] is True
    assert result.resolved == 1
    record = store.load().records[key.value]
    assert record.status is RunStatus.COMPLETED
    assert record.resolved is True


def test_evaluate_pending_matrix_groups_and_completes_pending_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    artifact_path = _sealed_artifact(tmp_path / "artifact.json")
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(manifest).with_completion(
        key,
        RunCompletion(
            artifact_path=str(artifact_path),
            artifact_sha256=artifact_path.with_suffix(
                ".json.sha256"
            ).read_text().strip(),
        ),
    )
    store.save(checkpoint)
    harness = RecordingHarness()

    result = evaluate_pending_matrix(
        checkpoint_store=store,
        harness=harness,
        dataset_name=manifest.dataset_name,
        output_directory=tmp_path / "evaluation",
    )

    assert result.records[key.value].status is RunStatus.COMPLETED
    assert len(harness.calls) == 1


def test_batch_evaluator_reports_run_key_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint = store.create(manifest).with_record(
        RunRecord(key=key, status=RunStatus.EVALUATION_PENDING)
    )
    store.save(checkpoint)

    with pytest.raises(InfrastructureRunError, match=key.value):
        BatchHarnessEvaluator(
            checkpoint_store=store,
            harness=RecordingHarness(),
            dataset_name=manifest.dataset_name,
            output_directory=tmp_path / "evaluation",
        ).evaluate([key])
