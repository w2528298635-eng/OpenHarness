from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.stream_events import AssistantTurnComplete
from openharness.repopilot.models import (
    BudgetUsage,
    Phase,
    RepoRunState,
)
from openharness.repopilot.swebench.adapters import (
    EvaluationArm,
    InferenceBudget,
    build_arm_configs,
)
from openharness.repopilot.swebench.models import DifficultyStratum, PublicInstance
from openharness.repopilot.swebench.runners import (
    CurrentRepoPilotRunner,
    NativeOpenHarnessRunner,
    patch_presence_command,
)
from openharness.tools.base import ToolRegistry


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "baseline")
    return repo


def _instance() -> PublicInstance:
    return PublicInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="abc",
        problem_statement="Change VALUE to the correct behavior.",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )


def test_patch_presence_command_fails_before_a_diff_and_passes_after_one(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    command = patch_presence_command()

    before = subprocess.run(command, cwd=repo, check=False)
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = subprocess.run(command, cwd=repo, check=False)

    assert before.returncode == 1
    assert after.returncode == 0


@pytest.mark.asyncio
async def test_current_runner_maps_public_task_budget_retrieval_and_diff(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    captured = {}

    class FakeScheduler:
        async def start(self, spec):
            captured["spec"] = spec
            (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
            return RepoRunState(
                run_id="repopilot-run-1",
                task=spec,
                phase=Phase.COMPLETE,
                original_repo=repo,
                worktree_path=repo,
                changed_files=["app.py"],
                budgets=BudgetUsage(
                    phase_calls=4,
                    total_tokens=120,
                    input_tokens=100,
                    output_tokens=20,
                ),
            )

    def scheduler_factory(repository_path, config, budget):
        captured["repository_path"] = repository_path
        captured["config"] = config
        captured["budget"] = budget
        return FakeScheduler()

    config = build_arm_configs(model="deepseek-v4-flash")[
        EvaluationArm.UPGRADED_WITH_RETRIEVAL
    ]
    budget = InferenceBudget(
        max_model_calls=9,
        max_total_tokens=80_000,
        max_wall_seconds=1200,
    )

    outcome = await CurrentRepoPilotRunner(
        scheduler_factory=scheduler_factory
    ).run(
        instance=_instance(),
        repository_path=repo,
        config=config,
        budget=budget,
    )

    spec = captured["spec"]
    assert spec.issue == _instance().problem_statement
    assert spec.retrieval.enabled is True
    assert spec.budgets.max_phase_calls == 9
    assert spec.budgets.max_total_tokens == 80_000
    assert outcome.status == "completed"
    assert outcome.model_patch.startswith("diff --git")
    assert outcome.input_tokens == 100
    assert outcome.output_tokens == 20


@pytest.mark.asyncio
async def test_native_runner_uses_public_prompt_budget_and_returns_git_patch(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    captured = {}

    class EditingEngine:
        async def submit_message(self, prompt):
            captured["submitted_prompt"] = prompt
            (repo / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
            yield AssistantTurnComplete(
                ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="Implemented the public issue.")],
                ),
                UsageSnapshot(input_tokens=50, output_tokens=10),
            )

    async def runtime_factory(**kwargs):
        captured["runtime"] = kwargs
        return SimpleNamespace(
            engine=EditingEngine(),
            tool_registry=ToolRegistry(),
        )

    budget = InferenceBudget(
        max_model_calls=7,
        max_total_tokens=50_000,
        max_wall_seconds=900,
    )
    config = build_arm_configs(model="deepseek-v4-flash")[EvaluationArm.NATIVE]

    outcome = await NativeOpenHarnessRunner(
        runtime_factory=runtime_factory
    ).run(
        instance=_instance(),
        repository_path=repo,
        config=config,
        budget=budget,
    )

    assert captured["runtime"]["model"] == "deepseek-v4-flash"
    assert captured["runtime"]["max_turns"] == 7
    assert _instance().problem_statement in captured["submitted_prompt"]
    assert "gold patch" not in captured["submitted_prompt"].casefold()
    assert outcome.status == "completed"
    assert outcome.model_patch.startswith("diff --git")
    assert outcome.input_tokens == 50
    assert outcome.output_tokens == 10
