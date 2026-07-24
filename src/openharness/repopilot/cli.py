from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from .benchmark import load_benchmark
from .phase_runner import OpenHarnessPhaseRunner
from .scheduler import RepoPilotScheduler
from .store import RunStore
from .task_loader import load_task
from .verifier import PythonPytestVerifier
from .workspace import WorkspaceManager

repopilot_app = typer.Typer(
    name="repopilot",
    help="Run deterministic local Python bug-repair workflows.",
    no_args_is_help=True,
)


def _scheduler(repo: Path, verify_timeout: float = 300) -> RepoPilotScheduler:
    return RepoPilotScheduler(
        store=RunStore(repo),
        workspace=WorkspaceManager(),
        verifier=PythonPytestVerifier(timeout_seconds=verify_timeout),
        phase_runner=OpenHarnessPhaseRunner(),
    )


def _task_or_bad_parameter(task_file: Path):
    try:
        return load_task(task_file)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="task") from exc


@repopilot_app.command("run")
def run_command(
    task: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Start a repair from a YAML task specification."""
    spec = _task_or_bad_parameter(task)
    state = asyncio.run(
        _scheduler(spec.repo_path, spec.budgets.verify_timeout_seconds).start(spec)
    )
    typer.echo(f"run_id: {state.run_id}")
    typer.echo(f"phase: {state.phase.value}")
    typer.echo(f"reason: {state.terminal_reason or 'verified'}")
    typer.echo(f"worktree: {state.worktree_path}")
    if state.phase.value != "COMPLETE":
        raise typer.Exit(code=1)


@repopilot_app.command("show")
def show_command(
    run_id: str,
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Original repository path."),
) -> None:
    """Print durable run state as JSON."""
    state = RunStore(repo).load_state(run_id)
    typer.echo(state.model_dump_json(indent=2))


@repopilot_app.command("resume")
def resume_command(
    run_id: str,
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Original repository path."),
) -> None:
    """Continue a non-terminal run from its last checkpoint."""
    store = RunStore(repo)
    state = store.load_state(run_id)
    scheduler = _scheduler(repo, state.task.budgets.verify_timeout_seconds)
    state = asyncio.run(scheduler.resume(run_id))
    typer.echo(f"run_id: {state.run_id}")
    typer.echo(f"phase: {state.phase.value}")
    if state.phase.value != "COMPLETE":
        raise typer.Exit(code=1)


@repopilot_app.command("report")
def report_command(
    run_id: str,
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Original repository path."),
) -> None:
    """Print the saved Markdown report."""
    report = RunStore(repo).run_dir(run_id) / "report.md"
    if not report.exists():
        raise typer.BadParameter(f"report does not exist: {report}")
    typer.echo(report.read_text(encoding="utf-8"))


@repopilot_app.command("benchmark")
def benchmark_command(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Run every RepoPilot case and print measured JSON results."""
    benchmark = load_benchmark(manifest)
    results = []
    for case in benchmark.cases:
        spec = _task_or_bad_parameter(case.task)
        state = asyncio.run(
            _scheduler(spec.repo_path, spec.budgets.verify_timeout_seconds).start(spec)
        )
        results.append(
            {
                "id": case.id,
                "strategy": "repopilot",
                "run_id": state.run_id,
                "phase": state.phase.value,
                "verified": state.phase.value == "COMPLETE",
                "phase_calls": state.budgets.phase_calls,
                "tokens": state.budgets.total_tokens,
                "failure": state.terminal_reason,
            }
        )
    typer.echo(
        json.dumps(
            {"name": benchmark.name, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
