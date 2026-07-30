from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from .adapters import EvaluationArm
from .dataset import (
    HuggingFaceDatasetProvider,
    JsonDatasetProvider,
    prepare_manifest,
)
from .docker_runner import run_doctor
from .models import SampleManifest, SamplingConfig
from .orchestration import CheckpointStore, build_run_keys
from .reporting import ExperimentReport, render_report_markdown

swebench_app = typer.Typer(
    name="swebench",
    help="Run leakage-safe public SWE-bench evaluations.",
    no_args_is_help=True,
)


def _doctor_failure(report_json: str) -> typer.BadParameter:
    payload = json.loads(report_json)
    failures = [
        check["summary"]
        for check in payload["checks"]
        if check["status"] == "fail"
    ]
    return typer.BadParameter(
        "Docker environment is not ready: " + "; ".join(failures),
        param_hint="environment",
    )


@swebench_app.command("doctor")
def doctor_command(
    cache: Annotated[
        Path,
        typer.Option("--cache", help="Intended local SWE-bench cache path."),
    ] = Path(".openharness-swebench"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete machine-readable report."),
    ] = False,
) -> None:
    """Inspect Docker and local resources without modifying them."""
    report = run_doctor(cache)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    for check in report.checks:
        typer.echo(f"{check.status.upper():4} {check.name}: {check.summary}")
    typer.echo(f"formal_ready: {str(report.formal_ready).lower()}")


@swebench_app.command("prepare")
def prepare_command(
    output: Annotated[Path, typer.Option("--output")],
    source: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    dataset_name: Annotated[str, typer.Option("--dataset-name")] = (
        "SWE-bench/SWE-bench_Verified"
    ),
    easy: Annotated[int, typer.Option("--easy", min=0)] = 10,
    medium: Annotated[int, typer.Option("--medium", min=0)] = 15,
    hard: Annotated[int, typer.Option("--hard", min=0)] = 20,
    seed: Annotated[int, typer.Option("--seed")] = 20260730,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Stream public metadata or use an offline snapshot to freeze a manifest."""
    if source is None:
        provider = HuggingFaceDatasetProvider(
            dataset_name=dataset_name,
            revision=revision,
        )
    else:
        if revision is None:
            raise typer.BadParameter(
                "--revision is required for an offline source",
                param_hint="revision",
            )
        provider = JsonDatasetProvider(
            source,
            dataset_name=dataset_name,
            revision=revision,
        )
    manifest = prepare_manifest(
        provider,
        output,
        SamplingConfig(
            easy=easy,
            medium=medium,
            hard=hard,
            seed=seed,
            dataset_name=dataset_name,
        ),
        force=force,
    )
    typer.echo(f"manifest: {output.resolve()}")
    typer.echo(f"instances: {len(manifest.instances)}")
    typer.echo(f"sha256: {manifest.sha256}")


def _create_experiment(
    *,
    manifest_path: Path,
    output: Path,
    repetitions: int,
    confirm_paid_matrix: bool,
    pilot: bool,
) -> None:
    if not confirm_paid_matrix:
        raise typer.BadParameter(
            "paid model matrix requires --confirm-paid-matrix",
            param_hint="confirm_paid_matrix",
        )
    report = run_doctor(output / "cache")
    report_json = report.model_dump_json()
    if not report.formal_ready:
        raise _doctor_failure(report_json)
    manifest = SampleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    arms = tuple(EvaluationArm)
    keys = build_run_keys(manifest, arms, repetitions=repetitions)
    checkpoint_path = output / ("pilot-checkpoint.json" if pilot else "checkpoint.json")
    store = CheckpointStore(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = store.load()
        if checkpoint.manifest_sha256 != manifest.sha256:
            raise typer.BadParameter(
                "checkpoint manifest digest does not match requested manifest"
            )
    else:
        store.create(manifest)
    typer.echo(f"checkpoint: {checkpoint_path.resolve()}")
    typer.echo(f"planned_runs: {len(keys)}")
    typer.echo(
        "environment ready; use resume after provider adapters and repositories "
        "have been prepared"
    )


@swebench_app.command("pilot")
def pilot_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    confirm_paid_matrix: Annotated[
        bool,
        typer.Option("--confirm-paid-matrix"),
    ] = False,
) -> None:
    """Create the guarded three-instance calibration checkpoint."""
    _create_experiment(
        manifest_path=manifest,
        output=output,
        repetitions=1,
        confirm_paid_matrix=confirm_paid_matrix,
        pilot=True,
    )


@swebench_app.command("run")
def run_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    repetitions: Annotated[int, typer.Option("--repetitions", min=1)] = 3,
    confirm_paid_matrix: Annotated[
        bool,
        typer.Option("--confirm-paid-matrix"),
    ] = False,
) -> None:
    """Create or validate the guarded formal experiment checkpoint."""
    _create_experiment(
        manifest_path=manifest,
        output=output,
        repetitions=repetitions,
        confirm_paid_matrix=confirm_paid_matrix,
        pilot=False,
    )


@swebench_app.command("resume")
def resume_command(
    checkpoint: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Inspect resumable checkpoint status without rerunning completed entries."""
    state = CheckpointStore(checkpoint).load()
    counts = Counter(record.status.value for record in state.records.values())
    typer.echo(f"manifest_sha256: {state.manifest_sha256}")
    typer.echo(json.dumps(dict(sorted(counts.items())), indent=2))


@swebench_app.command("report")
def report_command(
    report_json: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Render a saved structured experiment report as Markdown."""
    report = ExperimentReport.model_validate_json(
        report_json.read_text(encoding="utf-8")
    )
    markdown = render_report_markdown(report)
    if output is None:
        typer.echo(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    typer.echo(f"report: {output.resolve()}")
