from __future__ import annotations

from pathlib import Path

import pytest

from openharness.repopilot.swebench.adapters import (
    AgentAdapter,
    EvaluationArm,
    InferenceBudget,
    RunnerOutcome,
    build_arm_configs,
)
from openharness.repopilot.swebench.experiment import (
    InferenceExperimentExecutor,
    build_experiment_adapters,
)
from openharness.repopilot.swebench.inference import InferenceRunner
from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
    SampleManifest,
    SamplingConfig,
)
from openharness.repopilot.swebench.orchestration import (
    AgentRunError,
    RunKey,
)
from openharness.repopilot.swebench.runners import (
    CurrentRepoPilotRunner,
    LegacyRepoPilotRunner,
    NativeOpenHarnessRunner,
)


def _instance() -> PublicInstance:
    return PublicInstance(
        instance_id="django__django-1",
        repo="django/django",
        base_commit="a" * 40,
        problem_statement="Fix the public issue.",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )


def _manifest() -> SampleManifest:
    config = SamplingConfig(easy=1, medium=0, hard=0)
    return SampleManifest(
        dataset_name=config.dataset_name,
        dataset_revision="dataset-sha",
        sampling=config,
        instances=(_instance(),),
        sha256="b" * 64,
    )


class DisposableCache:
    def __init__(self, root: Path):
        self.root = root
        self.prepared: list[str] = []
        self.released: list[str] = []

    def prepare(self, instance: PublicInstance, *, workspace_id: str) -> Path:
        self.prepared.append(workspace_id)
        path = self.root / workspace_id
        path.mkdir()
        return path

    def release(self, instance: PublicInstance, *, workspace_id: str) -> None:
        self.released.append(workspace_id)


class StaticArmRunner:
    def __init__(self, status: str = "completed"):
        self.status = status

    async def run(self, **kwargs) -> RunnerOutcome:
        return RunnerOutcome(
            status=self.status,
            run_id="agent-run",
            model_patch=(
                "diff --git a/a.py b/a.py\n"
                if self.status == "completed"
                else ""
            ),
            duration_seconds=1,
            error=None if self.status == "completed" else "agent stopped",
        )


def _executor(tmp_path: Path, *, status: str = "completed"):
    config = build_arm_configs(model="deepseek-v4-flash")[EvaluationArm.NATIVE]
    cache = DisposableCache(tmp_path / "worktrees")
    (tmp_path / "worktrees").mkdir()
    executor = InferenceExperimentExecutor(
        manifest=_manifest(),
        adapters={
            EvaluationArm.NATIVE: AgentAdapter(
                config=config,
                runner=StaticArmRunner(status),
            )
        },
        repository_cache=cache,
        inference_runner=InferenceRunner(),
        artifact_directory=tmp_path / "artifacts",
        budget=InferenceBudget(
            max_model_calls=2,
            max_total_tokens=1000,
            max_wall_seconds=30,
        ),
    )
    return executor, cache


@pytest.mark.asyncio
async def test_executor_seals_patch_and_releases_disposable_worktree(
    tmp_path: Path,
) -> None:
    executor, cache = _executor(tmp_path)
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )

    completion = await executor.worker(key)

    assert Path(completion.artifact_path).is_file()
    assert Path(completion.artifact_path).with_suffix(".json.sha256").is_file()
    assert completion.resolved is None
    assert cache.prepared == cache.released


@pytest.mark.asyncio
async def test_executor_records_agent_failure_but_still_releases_worktree(
    tmp_path: Path,
) -> None:
    executor, cache = _executor(tmp_path, status="failed")
    key = RunKey(
        instance_id="django__django-1",
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )

    with pytest.raises(AgentRunError, match="agent stopped"):
        await executor.worker(key)

    assert cache.prepared == cache.released


def test_default_adapters_bind_each_arm_to_the_intended_runner(
    tmp_path: Path,
) -> None:
    adapters = build_experiment_adapters(
        model="deepseek-v4-flash",
        legacy_source=tmp_path / "legacy",
        artifact_root=tmp_path / "artifacts",
    )

    assert isinstance(adapters[EvaluationArm.NATIVE].runner, NativeOpenHarnessRunner)
    assert isinstance(adapters[EvaluationArm.LEGACY].runner, LegacyRepoPilotRunner)
    assert isinstance(
        adapters[EvaluationArm.UPGRADED_NO_RETRIEVAL].runner,
        CurrentRepoPilotRunner,
    )
    assert isinstance(
        adapters[EvaluationArm.UPGRADED_WITH_RETRIEVAL].runner,
        CurrentRepoPilotRunner,
    )
    assert adapters[EvaluationArm.UPGRADED_NO_RETRIEVAL].config.retrieval_enabled is False
    assert adapters[EvaluationArm.UPGRADED_WITH_RETRIEVAL].config.retrieval_enabled is True
