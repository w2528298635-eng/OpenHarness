from __future__ import annotations

import asyncio
import os
import shutil
import stat
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Any

from pydantic import BaseModel, Field

from .benchmark import BenchmarkManifest, load_benchmark
from .models import Phase
from .task_loader import load_task


class EvaluationStrategy(str, Enum):
    SCRIPTED = "scripted"
    MODEL_NO_RETRIEVAL = "model_no_retrieval"
    MODEL_WITH_RETRIEVAL = "model_with_retrieval"


class EvaluationCaseResult(BaseModel):
    case_id: str
    strategy: EvaluationStrategy
    repetition: int = Field(ge=1)
    run_id: str
    verified: bool
    duration_seconds: float = Field(ge=0)
    tokens: int = Field(default=0, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    replan_attempts: int = Field(default=0, ge=0)
    changed_file_compliant: bool
    failure: str | None = None
    workspace: Path | None = None


class StrategyAggregate(BaseModel):
    strategy: EvaluationStrategy
    cases: int
    verified: int
    success_rate: float
    median_duration_seconds: float
    total_tokens: int
    estimated_cost: float | None
    repair_attempts: int
    replan_attempts: int
    scope_compliance_rate: float
    failure_distribution: dict[str, int]


class EvaluationReport(BaseModel):
    schema_version: int = 1
    name: str
    generated_at: datetime
    results: list[EvaluationCaseResult]
    aggregates: dict[EvaluationStrategy, StrategyAggregate]


def aggregate_evaluation(
    results: list[EvaluationCaseResult],
) -> dict[EvaluationStrategy, StrategyAggregate]:
    grouped: dict[EvaluationStrategy, list[EvaluationCaseResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)
    aggregates: dict[EvaluationStrategy, StrategyAggregate] = {}
    for strategy, items in grouped.items():
        verified = sum(item.verified for item in items)
        costs = [item.estimated_cost for item in items if item.estimated_cost is not None]
        failures = Counter(item.failure for item in items if item.failure)
        aggregates[strategy] = StrategyAggregate(
            strategy=strategy,
            cases=len(items),
            verified=verified,
            success_rate=verified / len(items),
            median_duration_seconds=statistics.median(item.duration_seconds for item in items),
            total_tokens=sum(item.tokens for item in items),
            estimated_cost=sum(costs) if costs else None,
            repair_attempts=sum(item.repair_attempts for item in items),
            replan_attempts=sum(item.replan_attempts for item in items),
            scope_compliance_rate=(sum(item.changed_file_compliant for item in items) / len(items)),
            failure_distribution=dict(sorted(failures.items())),
        )
    return aggregates


def write_evaluation_report(
    report: EvaluationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"evaluation-{stamp}.json"
    markdown_path = output_dir / f"evaluation-{stamp}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    rows = [
        "# RepoPilot evaluation",
        "",
        f"Suite: `{report.name}`",
        "",
        "| strategy | runs | verified | success | median seconds | tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, aggregate in report.aggregates.items():
        rows.append(
            f"| {strategy.value} | {aggregate.cases} | {aggregate.verified} | "
            f"{aggregate.success_rate:.1%} | "
            f"{aggregate.median_duration_seconds:.3f} | {aggregate.total_tokens} |"
        )
    rows.extend(
        [
            "",
            (
                "Scripted results validate deterministic orchestration and fixtures; "
                "they are not model-quality scores."
            ),
            "",
            "## Runs",
            "",
        ]
    )
    for item in report.results:
        rows.append(
            f"- `{item.run_id}` — {item.case_id} / {item.strategy.value}: "
            f"{'verified' if item.verified else item.failure or 'failed'}"
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


SchedulerFactory = Callable[[Path, float], Any]


class EvaluationRunner:
    """Run the same committed repair cases under comparable strategies."""

    def __init__(
        self,
        scheduler_factory: SchedulerFactory,
        *,
        python_executable: str = sys.executable,
        git_executable: str = "git",
    ):
        self.scheduler_factory = scheduler_factory
        self.python_executable = python_executable
        self.git_executable = git_executable

    async def run(
        self,
        manifest: Path | BenchmarkManifest,
        strategies: list[EvaluationStrategy],
        *,
        output_dir: Path,
        repetitions: int = 1,
    ) -> EvaluationReport:
        if repetitions < 1:
            raise ValueError("repetitions must be at least one")
        loaded = load_benchmark(manifest) if isinstance(manifest, Path) else manifest
        results: list[EvaluationCaseResult] = []
        workspace_root = output_dir / "workspaces"
        for strategy in strategies:
            for repetition in range(1, repetitions + 1):
                for case in loaded.cases:
                    workspace = workspace_root / f"{case.id}-{strategy.value}-{repetition}"
                    task_path = self._materialize(case.task, workspace)
                    started = monotonic()
                    if strategy is EvaluationStrategy.SCRIPTED:
                        result = await asyncio.to_thread(
                            self._run_scripted,
                            case.id,
                            task_path,
                            workspace,
                            repetition,
                        )
                    else:
                        spec = load_task(task_path)
                        spec.retrieval.enabled = strategy is EvaluationStrategy.MODEL_WITH_RETRIEVAL
                        state = await self.scheduler_factory(
                            spec.repo_path,
                            spec.budgets.verify_timeout_seconds,
                        ).start(spec)
                        result = EvaluationCaseResult(
                            case_id=case.id,
                            strategy=strategy,
                            repetition=repetition,
                            run_id=state.run_id,
                            verified=state.phase is Phase.COMPLETE,
                            duration_seconds=monotonic() - started,
                            tokens=state.budgets.total_tokens or 0,
                            repair_attempts=state.budgets.repair_attempts,
                            replan_attempts=state.budgets.replan_attempts,
                            changed_file_compliant=(
                                not spec.allowed_paths
                                or all(
                                    any(Path(path).match(pattern) for pattern in spec.allowed_paths)
                                    for path in state.changed_files
                                )
                            ),
                            failure=state.terminal_reason,
                            workspace=workspace,
                        )
                    result.duration_seconds = monotonic() - started
                    results.append(result)
        report = EvaluationReport(
            name=loaded.name,
            generated_at=datetime.now(UTC),
            results=results,
            aggregates=aggregate_evaluation(results),
        )
        write_evaluation_report(report, output_dir)
        return report

    def _materialize(self, task_path: Path, destination: Path) -> Path:
        source = task_path.parent
        if destination.exists():
            shutil.rmtree(destination, onerror=_clear_readonly_and_retry)
        shutil.copytree(source, destination)
        repo = destination / "repo"
        self._git(repo, "init", "-q")
        self._git(repo, "config", "user.email", "repopilot@example.invalid")
        self._git(repo, "config", "user.name", "RepoPilot Evaluation")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-qm", "evaluation baseline")
        return destination / task_path.name

    def _run_scripted(
        self,
        case_id: str,
        task_path: Path,
        workspace: Path,
        repetition: int,
    ) -> EvaluationCaseResult:
        spec = load_task(task_path)
        patch = workspace / "fix.patch"
        if not patch.exists():
            raise ValueError(f"scripted case has no fix.patch: {case_id}")
        completed = subprocess.run(
            [self.git_executable, "apply", str(patch)],
            cwd=spec.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            return EvaluationCaseResult(
                case_id=case_id,
                strategy=EvaluationStrategy.SCRIPTED,
                repetition=repetition,
                run_id=f"scripted-{case_id}-{repetition}",
                verified=False,
                duration_seconds=0,
                changed_file_compliant=False,
                failure="patch_failed",
                workspace=workspace,
            )
        verification = subprocess.run(
            [self.python_executable, "-m", *spec.verify_command],
            cwd=spec.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        changed = self._git(spec.repo_path, "diff", "--name-only").splitlines()
        compliant = not spec.allowed_paths or all(
            any(Path(path).match(pattern) for pattern in spec.allowed_paths) for path in changed
        )
        return EvaluationCaseResult(
            case_id=case_id,
            strategy=EvaluationStrategy.SCRIPTED,
            repetition=repetition,
            run_id=f"scripted-{case_id}-{repetition}",
            verified=verification.returncode == 0,
            duration_seconds=0,
            changed_file_compliant=compliant,
            failure=None if verification.returncode == 0 else "verification_failed",
            workspace=workspace,
        )

    def _git(self, cwd: Path, *args: str) -> str:
        completed = subprocess.run(
            [self.git_executable, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "git command failed")
        return completed.stdout


def _clear_readonly_and_retry(function, path: str, exc_info) -> None:
    del exc_info
    os.chmod(path, stat.S_IWRITE)
    function(path)
