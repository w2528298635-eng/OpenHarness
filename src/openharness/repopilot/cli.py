from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .benchmark import load_benchmark
from .evaluation import EvaluationRunner, EvaluationStrategy
from .insight import InsightRequest
from .phase_runner import OpenHarnessPhaseRunner
from .scheduler import RepoPilotScheduler
from .service import RepoPilotService
from .store import RunStore
from .swebench.cli import swebench_app
from .task_loader import load_task
from .verifier import PythonPytestVerifier
from .workspace import WorkspaceManager

repopilot_app = typer.Typer(
    name="repopilot",
    help="Run deterministic local Python bug-repair workflows.",
    no_args_is_help=True,
)
repopilot_app.add_typer(swebench_app)


def _scheduler(repo: Path, verify_timeout: float = 300) -> RepoPilotScheduler:
    api_key = os.environ.get("OPENHARNESS_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return RepoPilotScheduler(
        store=RunStore(repo),
        workspace=WorkspaceManager(),
        verifier=PythonPytestVerifier(timeout_seconds=verify_timeout),
        phase_runner=OpenHarnessPhaseRunner(
            model=os.environ.get("OPENHARNESS_MODEL"),
            base_url=os.environ.get("OPENHARNESS_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key,
            api_format=os.environ.get("OPENHARNESS_API_FORMAT"),
        ),
    )


def _task_or_bad_parameter(task_file: Path):
    try:
        return load_task(task_file)
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="task") from exc


def _service() -> RepoPilotService:
    return RepoPilotService(_scheduler)


@repopilot_app.command("run")
def run_command(
    task: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Start a repair from a YAML task specification."""
    spec = _task_or_bad_parameter(task)
    state = asyncio.run(_service().start(spec))
    typer.echo(f"run_id: {state.run_id}")
    typer.echo(f"phase: {state.phase.value}")
    typer.echo(f"reason: {state.terminal_reason or 'verified'}")
    typer.echo(f"worktree: {state.worktree_path}")
    if state.phase.value != "COMPLETE":
        raise typer.Exit(code=1)


@repopilot_app.command("show")
def show_command(
    run_id: str,
    repo: Annotated[Path | None, typer.Option("--repo", help="Original repository path.")] = None,
) -> None:
    """Print durable run state as JSON."""
    state = RunStore(repo or Path.cwd()).load_state(run_id)
    typer.echo(state.model_dump_json(indent=2))


@repopilot_app.command("resume")
def resume_command(
    run_id: str,
    repo: Annotated[Path | None, typer.Option("--repo", help="Original repository path.")] = None,
) -> None:
    """Continue a non-terminal run from its last checkpoint."""
    repo_path = repo or Path.cwd()
    store = RunStore(repo_path)
    state = store.load_state(run_id)
    state = asyncio.run(_service().resume(run_id, repo_path))
    typer.echo(f"run_id: {state.run_id}")
    typer.echo(f"phase: {state.phase.value}")
    if state.phase.value != "COMPLETE":
        raise typer.Exit(code=1)


@repopilot_app.command("report")
def report_command(
    run_id: str,
    repo: Annotated[Path | None, typer.Option("--repo", help="Original repository path.")] = None,
) -> None:
    """Print the saved Markdown report."""
    report = RunStore(repo or Path.cwd()).run_dir(run_id) / "report.md"
    if not report.exists():
        raise typer.BadParameter(f"report does not exist: {report}")
    typer.echo(report.read_text(encoding="utf-8"))


@repopilot_app.command("cleanup")
def cleanup_command(
    run_id: str,
    repo: Annotated[Path | None, typer.Option("--repo", help="Original repository path.")] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Remove a worktree even when it contains changes."),
    ] = False,
) -> None:
    """Remove one registered repair worktree without deleting run artifacts."""
    repo_path = (repo or Path.cwd()).resolve()
    state = RunStore(repo_path).load_state(run_id)
    if state.worktree_path is None:
        raise typer.BadParameter("run has no worktree")
    try:
        asyncio.run(
            WorkspaceManager().cleanup(
                repo_path,
                state.worktree_path,
                force=force,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_id") from exc
    typer.echo(f"removed worktree: {state.worktree_path}")


@repopilot_app.command("benchmark")
def benchmark_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
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


@repopilot_app.command("evaluate")
def evaluate_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    strategy: Annotated[
        list[EvaluationStrategy] | None,
        typer.Option("--strategy", help="Repeat to compare multiple strategies."),
    ] = None,
    repetitions: Annotated[int, typer.Option("--repetitions", min=1)] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Directory for reports and preserved workspaces."),
    ] = None,
    allow_live_matrix: Annotated[
        bool,
        typer.Option(
            "--allow-live-matrix",
            help="Acknowledge multiple paid model runs.",
        ),
    ] = False,
) -> None:
    """Run reproducible scripted or model repair strategies."""
    strategies = strategy or [EvaluationStrategy.SCRIPTED]
    live_runs = sum(item is not EvaluationStrategy.SCRIPTED for item in strategies)
    case_count = len(load_benchmark(manifest).cases)
    if live_runs * case_count * repetitions > case_count and not allow_live_matrix:
        raise typer.BadParameter(
            "multiple paid model matrices require --allow-live-matrix",
            param_hint="strategy",
        )
    destination = output or manifest.resolve().parent / "reports"
    report = asyncio.run(
        EvaluationRunner(_scheduler).run(
            manifest,
            strategies,
            output_dir=destination,
            repetitions=repetitions,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(f"reports: {destination.resolve()}")


@repopilot_app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    unsafe_local_demo: Annotated[
        bool,
        typer.Option(
            "--unsafe-local-demo",
            help="Allow binding this unauthenticated local demo API beyond loopback.",
        ),
    ] = False,
) -> None:
    """Serve the local RepoPilot HTTP adapter."""
    if host not in {"127.0.0.1", "localhost", "::1"} and not unsafe_local_demo:
        raise typer.BadParameter(
            "non-loopback binding requires --unsafe-local-demo",
            param_hint="host",
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "API dependencies are missing; install OpenHarness with the 'api' extra"
        ) from exc
    uvicorn.run(
        "openharness.repopilot.api:create_app",
        factory=True,
        host=host,
        port=port,
    )


@repopilot_app.command("insight")
def insight_command(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    question: Annotated[str, typer.Option("--question", help="Repository question.")],
) -> None:
    """Generate a read-only, source-cited repository insight report."""
    report = asyncio.run(
        _service().insight(
            InsightRequest(repo_path=repo, question=question),
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(
        f"report: {RunStore(repo).run_dir(report.run_id) / 'insight.md'}",
    )
