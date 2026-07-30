from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openharness.repopilot.models import (
    BudgetConfig,
    Phase,
    RepoTaskSpec,
    RetrievalConfig,
)
from openharness.repopilot.phase_runner import OpenHarnessPhaseRunner
from openharness.repopilot.scheduler import RepoPilotScheduler
from openharness.repopilot.store import RunStore
from openharness.repopilot.verifier import PythonPytestVerifier
from openharness.repopilot.workspace import WorkspaceManager

from .adapters import ArmConfig, InferenceBudget, RunnerOutcome
from .localization import RetrievedLocation
from .models import PublicInstance

SchedulerFactory = Callable[
    [Path, ArmConfig, InferenceBudget],
    Any,
]


def patch_presence_command() -> list[str]:
    """Return success only after the agent has produced a Git working-tree change."""
    script = (
        "import subprocess,sys;"
        "result=subprocess.run("
        "['git','status','--porcelain=v1','--untracked-files=all'],"
        "capture_output=True,text=True,check=False);"
        "sys.exit(0 if result.returncode==0 and result.stdout.strip() else 1)"
    )
    return [sys.executable, "-c", script]


def _default_scheduler(
    repository_path: Path,
    config: ArmConfig,
    budget: InferenceBudget,
) -> RepoPilotScheduler:
    api_key = os.environ.get("OPENHARNESS_OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    phase_runner = OpenHarnessPhaseRunner(
        model=config.model,
        base_url=os.environ.get("OPENHARNESS_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key,
        api_format=os.environ.get("OPENHARNESS_API_FORMAT") or "openai",
    )
    return RepoPilotScheduler(
        store=RunStore(repository_path),
        workspace=WorkspaceManager(),
        verifier=PythonPytestVerifier(
            timeout_seconds=min(300, budget.max_wall_seconds)
        ),
        phase_runner=phase_runner,
    )


def _retrieval_locations(run_dir: Path) -> tuple[RetrievedLocation, ...]:
    locations: list[RetrievedLocation] = []
    rank = 1
    for path in sorted(run_dir.glob("context-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for selected in payload.get("selected_chunks", []):
            chunk = selected.get("chunk", {})
            file = chunk.get("path")
            if not isinstance(file, str) or not file:
                continue
            text = chunk.get("text", "")
            locations.append(
                RetrievedLocation(
                    file=file,
                    symbol=chunk.get("symbol") or None,
                    rank=rank,
                    characters=len(text) if isinstance(text, str) else 0,
                )
            )
            rank += 1
    return tuple(locations)


class CurrentRepoPilotRunner:
    def __init__(
        self,
        *,
        scheduler_factory: SchedulerFactory = _default_scheduler,
    ):
        self.scheduler_factory = scheduler_factory

    async def run(
        self,
        *,
        instance: PublicInstance,
        repository_path: Path,
        config: ArmConfig,
        budget: InferenceBudget,
    ) -> RunnerOutcome:
        task = RepoTaskSpec(
            repo_path=repository_path,
            issue=instance.problem_statement,
            verify_command=patch_presence_command(),
            budgets=BudgetConfig(
                max_phase_calls=budget.max_model_calls,
                max_wall_seconds=budget.max_wall_seconds,
                max_total_tokens=budget.max_total_tokens,
            ),
            retrieval=RetrievalConfig(enabled=config.retrieval_enabled),
        )
        scheduler = self.scheduler_factory(repository_path, config, budget)
        state = await scheduler.start(task)
        if state.worktree_path is None:
            return RunnerOutcome(
                status="failed",
                run_id=state.run_id,
                duration_seconds=0,
                error="RepoPilot did not create a worktree",
            )
        model_patch = await WorkspaceManager().diff(state.worktree_path)
        run_dir = RunStore(repository_path).run_dir(state.run_id)
        duration = (
            (state.completed_at - state.started_at).total_seconds()
            if state.completed_at is not None
            else max(0.0, (state.updated_at - state.started_at).total_seconds())
        )
        status = (
            "completed"
            if state.phase is Phase.COMPLETE and bool(model_patch.strip())
            else "failed"
        )
        return RunnerOutcome(
            status=status,
            run_id=state.run_id,
            model_patch=model_patch,
            duration_seconds=duration,
            input_tokens=state.budgets.input_tokens,
            output_tokens=state.budgets.output_tokens,
            cache_hit_tokens=state.budgets.cache_hit_tokens,
            model_calls=state.budgets.phase_calls,
            retrieval=_retrieval_locations(run_dir),
            error=None if status == "completed" else state.terminal_reason,
        )


def git_diff(repository_path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=repository_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout

