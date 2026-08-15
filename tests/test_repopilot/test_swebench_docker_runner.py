from __future__ import annotations

import json
import subprocess
from pathlib import Path

from openharness.repopilot.swebench.docker_runner import (
    CommandResult,
    HarnessPrediction,
    OfficialHarnessRunner,
    SubprocessCommandRunner,
    SystemFacts,
    run_doctor,
    write_predictions_jsonl,
)

GIB = 1024**3


def test_subprocess_runner_normalizes_empty_streams(monkeypatch) -> None:
    received: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        received.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=None, stderr=None
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    result = SubprocessCommandRunner().run(["wsl", "--status"], timeout_seconds=1)

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert received["encoding"] == "utf-8"
    assert received["errors"] == "replace"


class RecordingRunner:
    def __init__(self, results: list[CommandResult]):
        self.results = list(results)
        self.commands: list[list[str]] = []

    def run(self, argv: list[str], *, timeout_seconds: float) -> CommandResult:
        self.commands.append(list(argv))
        return self.results.pop(0)


def _facts(
    *,
    system: str = "Windows",
    machine: str = "AMD64",
    cpus: int = 12,
    memory_gib: int = 32,
    free_gib: int = 150,
) -> SystemFacts:
    return SystemFacts(
        system=system,
        machine=machine,
        logical_cpus=cpus,
        memory_bytes=memory_gib * GIB,
        free_disk_bytes=free_gib * GIB,
    )


def test_doctor_reports_docker_daemon_unavailable() -> None:
    runner = RecordingRunner(
        [
            CommandResult(exit_code=0, stdout="Docker version 29", stderr=""),
            CommandResult(exit_code=1, stdout="", stderr="daemon unavailable"),
            CommandResult(exit_code=0, stdout="WSL 2", stderr=""),
        ]
    )

    report = run_doctor(
        Path("C:/cache"),
        command_runner=runner,
        system_facts=_facts(),
    )

    assert report.formal_ready is False
    daemon = next(check for check in report.checks if check.name == "docker_daemon")
    assert daemon.status == "fail"
    assert "daemon unavailable" in daemon.summary


def test_doctor_requires_wsl_but_treats_120_gib_as_recommendation_for_subset() -> None:
    runner = RecordingRunner(
        [
            CommandResult(exit_code=0, stdout="Docker version 29", stderr=""),
            CommandResult(exit_code=0, stdout='{"ServerVersion":"29"}', stderr=""),
            CommandResult(exit_code=1, stdout="", stderr="WSL denied"),
        ]
    )

    report = run_doctor(
        Path("C:/cache"),
        command_runner=runner,
        system_facts=_facts(cpus=4, memory_gib=8, free_gib=100),
    )

    statuses = {check.name: check.status for check in report.checks}
    assert statuses["cpu"] == "warn"
    assert statuses["memory"] == "warn"
    assert statuses["disk"] == "warn"
    assert statuses["wsl2"] == "fail"
    assert report.formal_ready is False


def test_doctor_fails_when_subset_has_less_than_minimum_safe_disk() -> None:
    runner = RecordingRunner(
        [
            CommandResult(exit_code=0, stdout="Docker version 29", stderr=""),
            CommandResult(exit_code=0, stdout='{"ServerVersion":"29"}', stderr=""),
            CommandResult(exit_code=0, stdout="WSL 2", stderr=""),
        ]
    )

    report = run_doctor(
        Path("C:/cache"),
        command_runner=runner,
        system_facts=_facts(free_gib=10),
    )

    disk = next(check for check in report.checks if check.name == "disk")
    assert disk.status == "fail"
    assert "20 GiB minimum" in disk.summary


def test_doctor_is_ready_on_supported_linux_without_wsl_check() -> None:
    runner = RecordingRunner(
        [
            CommandResult(exit_code=0, stdout="Docker version 29", stderr=""),
            CommandResult(exit_code=0, stdout='{"ServerVersion":"29"}', stderr=""),
        ]
    )

    report = run_doctor(
        Path("/cache"),
        command_runner=runner,
        system_facts=_facts(system="Linux", machine="x86_64"),
    )

    assert report.formal_ready is True
    assert "wsl2" not in {check.name for check in report.checks}


def test_prediction_jsonl_uses_official_field_names(tmp_path: Path) -> None:
    target = tmp_path / "predictions.jsonl"
    predictions = [
        HarnessPrediction(
            instance_id="django__django-1",
            model_name_or_path="repopilot/deepseek-v4-flash",
            model_patch="diff --git a/a.py b/a.py\n",
        )
    ]

    write_predictions_jsonl(target, predictions)

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "instance_id": "django__django-1",
        "model_name_or_path": "repopilot/deepseek-v4-flash",
        "model_patch": "diff --git a/a.py b/a.py\n",
    }


def test_official_harness_uses_argv_and_parses_report(tmp_path: Path) -> None:
    predictions = tmp_path / "directory with spaces" / "predictions.jsonl"
    predictions.parent.mkdir()
    predictions.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "results.json"
    report_path.write_text(
        json.dumps(
            {
                "total_instances": 2,
                "submitted_instances": 2,
                "completed_instances": 2,
                "resolved_instances": 1,
            }
        ),
        encoding="utf-8",
    )
    command_runner = RecordingRunner(
        [CommandResult(exit_code=0, stdout="complete", stderr="")]
    )
    runner = OfficialHarnessRunner(
        python_executable="python",
        command_runner=command_runner,
    )

    result = runner.evaluate(
        dataset_name="SWE-bench/SWE-bench_Verified",
        predictions_path=predictions,
        run_id="formal-1",
        result_path=report_path,
        max_workers=2,
        cache_level="env",
        clean=True,
        instance_ids=("django__django-1", "sympy__sympy-2"),
    )

    command = command_runner.commands[0]
    assert command[0:3] == ["python", "-m", "swebench.harness.run_evaluation"]
    assert str(predictions) in command
    assert command[command.index("--instance_ids") + 1 :] == [
        "django__django-1",
        "sympy__sympy-2",
    ]
    assert command[command.index("--clean") + 1] == "true"
    assert command[command.index("--report_dir") + 1] == str(report_path.parent)
    assert result.status == "completed"
    assert result.resolved == 1
    assert result.resolution_rate == 0.5


def test_official_harness_classifies_timeout_without_parsing_results(
    tmp_path: Path,
) -> None:
    command_runner = RecordingRunner(
        [
            CommandResult(
                exit_code=None,
                stdout="",
                stderr="timed out",
                timed_out=True,
            )
        ]
    )
    runner = OfficialHarnessRunner(
        python_executable="python",
        command_runner=command_runner,
    )

    result = runner.evaluate(
        dataset_name="SWE-bench/SWE-bench_Verified",
        predictions_path=tmp_path / "predictions.jsonl",
        run_id="formal-timeout",
        result_path=tmp_path / "missing.json",
        max_workers=1,
        cache_level="base",
        timeout_seconds=10,
    )

    assert result.status == "timeout"
    assert result.resolved == 0


def test_official_harness_finds_the_report_named_by_swebench(
    tmp_path: Path,
) -> None:
    generated_report = tmp_path / "RepoPilot__arm.run-1.json"
    generated_report.write_text(
        json.dumps(
            {
                "total_instances": 1,
                "submitted_instances": 1,
                "completed_instances": 1,
                "resolved_ids": ["django__django-1"],
            }
        ),
        encoding="utf-8",
    )
    runner = OfficialHarnessRunner(
        python_executable="python",
        command_runner=RecordingRunner(
            [
                CommandResult(
                    exit_code=0,
                    stdout=f"Report written to {generated_report}",
                    stderr="",
                )
            ]
        ),
    )

    result = runner.evaluate(
        dataset_name="SWE-bench/SWE-bench_Verified",
        predictions_path=tmp_path / "predictions.jsonl",
        run_id="run-1",
        result_path=tmp_path / "expected.json",
        max_workers=1,
        cache_level="env",
    )

    assert result.status == "completed"
    assert result.report_path == str(generated_report)
    assert result.resolved_instance_ids == ("django__django-1",)


def test_official_harness_preserves_resolved_instance_ids(tmp_path: Path) -> None:
    report_path = tmp_path / "results.json"
    report_path.write_text(
        json.dumps(
            {
                "total_instances": 2,
                "submitted_instances": 2,
                "completed_instances": 2,
                "resolved_ids": ["django__django-1"],
            }
        ),
        encoding="utf-8",
    )
    command_runner = RecordingRunner(
        [CommandResult(exit_code=0, stdout="complete", stderr="")]
    )

    result = OfficialHarnessRunner(
        python_executable="python",
        command_runner=command_runner,
    ).evaluate(
        dataset_name="SWE-bench/SWE-bench_Verified",
        predictions_path=tmp_path / "predictions.jsonl",
        run_id="ids",
        result_path=report_path,
        max_workers=1,
        cache_level="env",
    )

    assert result.resolved == 1
    assert result.resolved_instance_ids == ("django__django-1",)
