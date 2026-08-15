from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from openharness.repopilot.swebench.adapters import (
    LEGACY_REPOPILOT_COMMIT,
    AgentAdapter,
    EvaluationArm,
    InferenceBudget,
    RunnerOutcome,
    build_arm_configs,
)
from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
)


class FakeRunner:
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
            run_id=f"run-{instance.instance_id}",
            model_patch=f"diff --git a/{repository_path.name}.py b/{repository_path.name}.py\n",
            duration_seconds=1.5,
            input_tokens=100,
            output_tokens=20,
        )


def _instance() -> PublicInstance:
    return PublicInstance(
        instance_id="django__django-1",
        repo="django/django",
        base_commit="abc",
        problem_statement="Fix public issue.",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )


def test_current_ablation_configs_differ_only_in_retrieval_flag() -> None:
    configs = build_arm_configs(model="deepseek-v4-flash")
    without = configs[EvaluationArm.UPGRADED_NO_RETRIEVAL]
    with_retrieval = configs[EvaluationArm.UPGRADED_WITH_RETRIEVAL]

    assert without.retrieval_enabled is False
    assert with_retrieval.retrieval_enabled is True
    assert without.model_copy(
        update={"arm": with_retrieval.arm, "retrieval_enabled": True}
    ) == with_retrieval


def test_legacy_arm_is_pinned_to_the_preupgrade_commit() -> None:
    config = build_arm_configs(model="deepseek-v4-flash")[EvaluationArm.LEGACY]

    assert config.legacy_commit == LEGACY_REPOPILOT_COMMIT
    assert config.legacy_commit == "15fb5947bff15fccb2faea186240fcd76ec0e2ab"


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_formal_arm_configs_reject_deprecated_model_aliases(model: str) -> None:
    with pytest.raises(ValidationError, match="deprecated"):
        build_arm_configs(model=model)


@pytest.mark.asyncio
async def test_agent_adapter_returns_the_common_runner_outcome(tmp_path: Path) -> None:
    adapter = AgentAdapter(
        config=build_arm_configs(model="deepseek-v4-flash")[EvaluationArm.NATIVE],
        runner=FakeRunner(),
    )

    outcome = await adapter.run(
        instance=_instance(),
        repository_path=tmp_path,
        budget=InferenceBudget(
            max_model_calls=8,
            max_total_tokens=100_000,
            max_wall_seconds=1800,
        ),
    )

    assert outcome.status == "completed"
    assert outcome.input_tokens == 100
    assert outcome.output_tokens == 20

