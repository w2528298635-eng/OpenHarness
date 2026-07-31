from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from .adapters import EvaluationArm, InferenceBudget
from .dataset import (
    HuggingFaceDatasetProvider,
    JsonDatasetProvider,
    prepare_manifest,
    write_manifest,
)
from .docker_runner import OfficialHarnessRunner, run_doctor
from .execution import evaluate_pending_matrix
from .experiment import build_experiment_adapters, run_inference_matrix
from .localization_execution import (
    LocalizationCheckpointStore,
    evaluate_localization_manifest,
)
from .localization_reporting import build_localization_report
from .models import SampleManifest, SamplingConfig
from .orchestration import CheckpointStore, RunRecord, build_run_keys
from .reporting import ExperimentReport, render_report_markdown
from .repositories import SelectedRepositoryCache
from .sampler import derive_pilot_manifest

_DEFAULT_REPOSITORY_CACHE = Path(tempfile.gettempdir()) / "repopilot-swebench-source"

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
    if not pilot and not report.formal_ready:
        raise _doctor_failure(report_json)
    manifest = SampleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if pilot:
        manifest = derive_pilot_manifest(manifest)
        write_manifest(output / "pilot-manifest.json", manifest)
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
        checkpoint = store.create(manifest)
    desired_keys = {key.value for key in keys}
    existing_keys = set(checkpoint.records)
    if existing_keys and existing_keys != desired_keys:
        raise typer.BadParameter(
            "checkpoint run matrix does not match requested arms and repetitions"
        )
    if not existing_keys:
        for key in keys:
            checkpoint = checkpoint.with_record(RunRecord(key=key))
        store.save(checkpoint)
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
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Run pending paid inference entries."),
    ] = False,
    evaluate: Annotated[
        bool,
        typer.Option("--evaluate", help="Run official SWE-bench Docker evaluation."),
    ] = False,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    repository_cache: Annotated[
        Path,
        typer.Option("--repository-cache"),
    ] = _DEFAULT_REPOSITORY_CACHE,
    legacy_source: Annotated[
        Path | None,
        typer.Option("--legacy-source", exists=True, file_okay=False),
    ] = None,
    model: Annotated[str, typer.Option("--model")] = "deepseek-v4-flash",
    max_model_calls: Annotated[
        int,
        typer.Option("--max-model-calls", min=1),
    ] = 8,
    max_total_tokens: Annotated[
        int,
        typer.Option("--max-total-tokens", min=1),
    ] = 50_000,
    max_wall_seconds: Annotated[
        float,
        typer.Option("--max-wall-seconds", min=1),
    ] = 900,
    max_infrastructure_retries: Annotated[
        int,
        typer.Option("--max-infrastructure-retries", min=0),
    ] = 1,
    base_url: Annotated[
        str,
        typer.Option("--base-url"),
    ] = "https://api.deepseek.com/v1",
    harness_python: Annotated[
        str,
        typer.Option("--harness-python"),
    ] = sys.executable,
    harness_workers: Annotated[
        int,
        typer.Option("--harness-workers", min=1),
    ] = 1,
    confirm_paid_matrix: Annotated[
        bool,
        typer.Option("--confirm-paid-matrix"),
    ] = False,
) -> None:
    """Inspect or execute a resumable, leakage-safe inference checkpoint."""
    store = CheckpointStore(checkpoint)
    state = store.load()
    selected_manifest: SampleManifest | None = None
    if execute or evaluate:
        if manifest is None:
            raise typer.BadParameter(
                "--manifest is required with --execute or --evaluate",
                param_hint="manifest",
            )
        selected_manifest = SampleManifest.model_validate_json(
            manifest.read_text(encoding="utf-8")
        )
    if execute:
        if not confirm_paid_matrix:
            raise typer.BadParameter(
                "paid model matrix requires --confirm-paid-matrix",
                param_hint="confirm_paid_matrix",
            )
        if legacy_source is None:
            raise typer.BadParameter(
                "--legacy-source is required with --execute",
                param_hint="legacy_source",
            )
        if not (
            os.environ.get("OPENHARNESS_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ):
            raise typer.BadParameter(
                "DeepSeek API credential is not available in this process; "
                "set OPENHARNESS_OPENAI_API_KEY securely before execution",
                param_hint="credential",
            )
        artifact_root = (output or checkpoint.parent).resolve()
        os.environ.setdefault("OPENHARNESS_BASE_URL", base_url)
        os.environ.setdefault("OPENHARNESS_API_FORMAT", "openai")
        state = asyncio.run(
            run_inference_matrix(
                manifest=selected_manifest,
                checkpoint_store=store,
                adapters=build_experiment_adapters(
                    model=model,
                    legacy_source=legacy_source.resolve(),
                    artifact_root=artifact_root,
                ),
                repository_cache=SelectedRepositoryCache(repository_cache),
                artifact_directory=artifact_root / "inference",
                budget=InferenceBudget(
                    max_model_calls=max_model_calls,
                    max_total_tokens=max_total_tokens,
                    max_wall_seconds=max_wall_seconds,
                ),
                max_infrastructure_retries=max_infrastructure_retries,
            )
        )
    if evaluate:
        assert selected_manifest is not None
        artifact_root = (output or checkpoint.parent).resolve()
        state = evaluate_pending_matrix(
            checkpoint_store=store,
            harness=OfficialHarnessRunner(
                python_executable=harness_python,
            ),
            dataset_name=selected_manifest.dataset_name,
            output_directory=artifact_root / "evaluation",
            max_workers=harness_workers,
        )
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


@swebench_app.command("localize")
def localize_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    repository_root: Annotated[Path, typer.Option("--repository-root")],
    strategy: Annotated[str, typer.Option("--strategy")] = "lexical",
    query_planning: Annotated[
        bool,
        typer.Option("--query-planning/--no-query-planning"),
    ] = True,
    structural_expansion: Annotated[
        bool,
        typer.Option("--structural-expansion/--no-structural-expansion"),
    ] = True,
    char_budget: Annotated[int, typer.Option("--char-budget", min=100)] = 12_000,
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 12,
) -> None:
    """Run resumable, post-hoc code-localization evaluation and ablations."""
    if strategy not in {"lexical", "hybrid"}:
        raise typer.BadParameter(
            "strategy must be lexical or hybrid",
            param_hint="strategy",
        )
    selected = SampleManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    state = evaluate_localization_manifest(
        instances=selected.instances,
        public_rows_with_gold=HuggingFaceDatasetProvider(
            dataset_name=selected.dataset_name,
            revision=selected.dataset_revision,
        ).rows(),
        repository_root=repository_root,
        store=LocalizationCheckpointStore(checkpoint),
        char_budget=char_budget,
        top_k=top_k,
        retrieval_strategy=strategy,
        query_planning=query_planning,
        structural_expansion=structural_expansion,
    )
    typer.echo(f"checkpoint: {checkpoint.resolve()}")
    typer.echo(f"completed: {len(state.records)}")


@swebench_app.command("localization-report")
def localization_report_command(
    checkpoint: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    manifest: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Aggregate a localization checkpoint overall and by difficulty."""
    selected = SampleManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    report = build_localization_report(
        LocalizationCheckpointStore(checkpoint).load(),
        selected,
    )
    payload = report.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    typer.echo(f"report: {output.resolve()}")
