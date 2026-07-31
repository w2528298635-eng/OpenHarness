from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from openharness.cli import app
from openharness.repopilot.swebench.docker_runner import (
    DoctorCheck,
    DoctorReport,
    SystemFacts,
)

runner = CliRunner()


def _rows() -> list[dict[str, str]]:
    return [
        {
            "instance_id": "alpha__repo-1",
            "repo": "alpha/repo",
            "base_commit": "a",
            "problem_statement": "Easy.",
            "difficulty": "<15 min fix",
        },
        {
            "instance_id": "beta__repo-2",
            "repo": "beta/repo",
            "base_commit": "b",
            "problem_statement": "Medium.",
            "difficulty": "15 min - 1 hour",
        },
        {
            "instance_id": "gamma__repo-3",
            "repo": "gamma/repo",
            "base_commit": "c",
            "problem_statement": "Hard.",
            "difficulty": "1-4 hours",
        },
    ]


def test_swebench_help_lists_workflow_commands() -> None:
    result = runner.invoke(app, ["repopilot", "swebench", "--help"])

    assert result.exit_code == 0
    for command in ["doctor", "prepare", "pilot", "run", "resume", "report", "localize"]:
        assert command in result.stdout


def test_prepare_builds_offline_frozen_manifest(tmp_path: Path) -> None:
    source = tmp_path / "public.json"
    source.write_text(json.dumps(_rows()), encoding="utf-8")
    output = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "repopilot",
            "swebench",
            "prepare",
            str(source),
            "--revision",
            "fixture-revision",
            "--output",
            str(output),
            "--easy",
            "1",
            "--medium",
            "1",
            "--hard",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["instances"]) == 3
    assert len(payload["sha256"]) == 64
    assert payload["dataset_revision"] == "fixture-revision"


def test_prepare_can_stream_public_dataset_without_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from openharness.repopilot.swebench import cli as swebench_cli

    class OnlineProvider:
        dataset_name = "SWE-bench/SWE-bench_Verified"
        revision = "resolved-dataset-sha"

        def rows(self):
            return _rows()

    monkeypatch.setattr(
        swebench_cli,
        "HuggingFaceDatasetProvider",
        lambda **kwargs: OnlineProvider(),
    )
    output = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "repopilot",
            "swebench",
            "prepare",
            "--output",
            str(output),
            "--easy",
            "1",
            "--medium",
            "1",
            "--hard",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset_revision"] == "resolved-dataset-sha"
    assert len(payload["instances"]) == 3


def test_doctor_can_emit_machine_readable_json(monkeypatch) -> None:
    from openharness.repopilot.swebench import cli as swebench_cli

    report = DoctorReport(
        facts=SystemFacts(
            system="Linux",
            machine="x86_64",
            logical_cpus=8,
            memory_bytes=16 * 1024**3,
            free_disk_bytes=120 * 1024**3,
        ),
        checks=(DoctorCheck(name="docker", status="pass", summary="ready"),),
    )
    monkeypatch.setattr(swebench_cli, "run_doctor", lambda path: report)

    result = runner.invoke(
        app,
        ["repopilot", "swebench", "doctor", "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["formal_ready"] is True


def test_formal_run_refuses_to_start_when_doctor_is_not_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from openharness.repopilot.swebench import cli as swebench_cli

    report = DoctorReport(
        facts=SystemFacts(
            system="Windows",
            machine="AMD64",
            logical_cpus=8,
            memory_bytes=16 * 1024**3,
            free_disk_bytes=10 * 1024**3,
        ),
        checks=(DoctorCheck(name="disk", status="fail", summary="10 GiB free"),),
    )
    monkeypatch.setattr(swebench_cli, "run_doctor", lambda path: report)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "repopilot",
            "swebench",
            "run",
            str(manifest),
            "--output",
            str(tmp_path / "output"),
            "--confirm-paid-matrix",
        ],
    )

    assert result.exit_code != 0
    assert "Docker environment is not ready" in result.output
    assert "10 GiB free" in result.output


def test_pilot_plans_inference_without_a_docker_ready_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from openharness.repopilot.swebench import cli as swebench_cli
    from openharness.repopilot.swebench.dataset import prepare_manifest
    from openharness.repopilot.swebench.models import SamplingConfig

    class StaticProvider:
        dataset_name = "fixture/verified"
        revision = "fixture-sha"

        def rows(self):
            return _rows()

    manifest_path = tmp_path / "formal.json"
    prepare_manifest(
        StaticProvider(),
        manifest_path,
        SamplingConfig(easy=1, medium=1, hard=1),
    )
    ready = DoctorReport(
        facts=SystemFacts(
            system="Windows",
            machine="AMD64",
            logical_cpus=16,
            memory_bytes=16 * 1024**3,
            free_disk_bytes=10 * 1024**3,
        ),
        checks=(DoctorCheck(name="docker", status="fail", summary="not ready"),),
    )
    monkeypatch.setattr(swebench_cli, "run_doctor", lambda path: ready)
    output = tmp_path / "pilot"

    result = runner.invoke(
        app,
        [
            "repopilot",
            "swebench",
            "pilot",
            str(manifest_path),
            "--output",
            str(output),
            "--confirm-paid-matrix",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "planned_runs: 12" in result.output
    pilot_payload = json.loads(
        (output / "pilot-manifest.json").read_text(encoding="utf-8")
    )
    assert len(pilot_payload["instances"]) == 3
    checkpoint_payload = json.loads(
        (output / "pilot-checkpoint.json").read_text(encoding="utf-8")
    )
    assert len(checkpoint_payload["records"]) == 12


def test_resume_execute_runs_sealed_inference_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from openharness.repopilot.swebench import cli as swebench_cli
    from openharness.repopilot.swebench.adapters import (
        AgentAdapter,
        EvaluationArm,
        RunnerOutcome,
        build_arm_configs,
    )
    from openharness.repopilot.swebench.dataset import prepare_manifest
    from openharness.repopilot.swebench.docker_runner import HarnessResult
    from openharness.repopilot.swebench.models import SamplingConfig
    from openharness.repopilot.swebench.orchestration import (
        CheckpointStore,
        RunKey,
        RunRecord,
    )

    class StaticProvider:
        dataset_name = "fixture/verified"
        revision = "fixture-sha"

        def rows(self):
            return [_rows()[0]]

    class StaticRunner:
        async def run(self, **kwargs):
            return RunnerOutcome(
                status="completed",
                run_id="cli-agent-run",
                model_patch="diff --git a/a.py b/a.py\n",
                duration_seconds=1,
            )

    class DisposableCache:
        def __init__(self, root: Path):
            self.root = root

        def prepare(self, instance, *, workspace_id: str) -> Path:
            path = self.root / workspace_id
            path.mkdir(parents=True)
            return path

        def release(self, instance, *, workspace_id: str) -> None:
            return None

    class ResolvingHarness:
        def __init__(self, **kwargs):
            return None

        def evaluate(self, **kwargs):
            return HarnessResult(
                status="completed",
                total=1,
                submitted=1,
                completed=1,
                resolved=1,
                resolved_instance_ids=("alpha__repo-1",),
            )

    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_manifest(
        StaticProvider(),
        manifest_path,
        SamplingConfig(easy=1, medium=0, hard=0),
    )
    key = RunKey(
        instance_id=manifest.instances[0].instance_id,
        arm=EvaluationArm.NATIVE,
        repetition=1,
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    store = CheckpointStore(checkpoint_path)
    store.save(store.create(manifest).with_record(RunRecord(key=key)))
    config = build_arm_configs(model="deepseek-v4-flash")[key.arm]
    monkeypatch.setattr(
        swebench_cli,
        "build_experiment_adapters",
        lambda **kwargs: {
            key.arm: AgentAdapter(config=config, runner=StaticRunner())
        },
    )
    monkeypatch.setattr(swebench_cli, "SelectedRepositoryCache", DisposableCache)
    monkeypatch.setattr(
        swebench_cli,
        "OfficialHarnessRunner",
        ResolvingHarness,
        raising=False,
    )
    monkeypatch.setenv("OPENHARNESS_OPENAI_API_KEY", "test-key")
    legacy_source = tmp_path / "legacy"
    legacy_source.mkdir()

    result = runner.invoke(
        app,
        [
            "repopilot",
            "swebench",
            "resume",
            str(checkpoint_path),
            "--execute",
            "--evaluate",
            "--manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "output"),
            "--repository-cache",
            str(tmp_path / "cache"),
            "--legacy-source",
            str(legacy_source),
            "--confirm-paid-matrix",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"completed": 1' in result.output
    assert list((tmp_path / "output" / "inference").glob("*.json"))
