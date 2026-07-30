from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.repopilot.swebench.adapters import (
    AgentAdapter,
    EvaluationArm,
    InferenceBudget,
    RunnerOutcome,
    build_arm_configs,
)
from openharness.repopilot.swebench.inference import (
    InferenceRequest,
    InferenceRunner,
    verify_artifact_seal,
)
from openharness.repopilot.swebench.models import DifficultyStratum, PublicInstance


class FixedRunner:
    async def run(
        self,
        *,
        instance: PublicInstance,
        repository_path: Path,
        config: object,
        budget: InferenceBudget,
    ) -> RunnerOutcome:
        return RunnerOutcome(
            status="completed",
            run_id="provider-run-1",
            model_patch="diff --git a/src/a.py b/src/a.py\n",
            duration_seconds=2,
            input_tokens=200,
            output_tokens=30,
        )


def _request() -> InferenceRequest:
    return InferenceRequest(
        instance=PublicInstance(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc",
            problem_statement="Fix public issue.",
            source_difficulty="15 min - 1 hour",
            difficulty=DifficultyStratum.MEDIUM,
        ),
        arm=EvaluationArm.UPGRADED_WITH_RETRIEVAL,
        repetition=1,
        budget=InferenceBudget(
            max_model_calls=8,
            max_total_tokens=100_000,
            max_wall_seconds=1800,
        ),
    )


def test_inference_request_serialization_has_no_gold_vocabulary() -> None:
    serialized = _request().model_dump_json()

    for forbidden in (
        '"patch"',
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "gold_files",
        "gold_symbols",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_inference_runner_writes_atomic_artifact_and_sha256_seal(
    tmp_path: Path,
) -> None:
    request = _request()
    adapter = AgentAdapter(
        config=build_arm_configs(model="deepseek-v4-flash")[request.arm],
        runner=FixedRunner(),
    )

    artifact_path = await InferenceRunner().run(
        request=request,
        adapter=adapter,
        repository_path=tmp_path / "repo",
        output_directory=tmp_path / "artifacts",
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["instance_id"] == "django__django-1"
    assert payload["model_patch"].startswith("diff --git")
    assert len(payload["model_patch_sha256"]) == 64
    assert verify_artifact_seal(artifact_path) is True
    assert not list((tmp_path / "artifacts").glob("*.tmp"))


@pytest.mark.asyncio
async def test_artifact_seal_detects_post_inference_tampering(tmp_path: Path) -> None:
    request = _request()
    adapter = AgentAdapter(
        config=build_arm_configs(model="deepseek-v4-flash")[request.arm],
        runner=FixedRunner(),
    )
    artifact_path = await InferenceRunner().run(
        request=request,
        adapter=adapter,
        repository_path=tmp_path / "repo",
        output_directory=tmp_path / "artifacts",
    )
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    assert verify_artifact_seal(artifact_path) is False

