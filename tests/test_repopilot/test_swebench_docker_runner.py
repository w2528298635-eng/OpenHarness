from __future__ import annotations

import json
from pathlib import Path

from openharness.repopilot.swebench.docker_runner import (
    CommandResult,
    HarnessPrediction,
    OfficialHarnessRunner,
    SystemFacts,
    run_doctor,
    write_predictions_jsonl,
)

GIB = 1024**3


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


def test_doctor_requires_wsl_and_disk_but_only_warns_for_recommended_cpu_ram() -> None:
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
    assert statuses["disk"] == "fail"
    assert statuses["wsl2"] == "fail"
    assert report.formal_ready is False


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
    )

    command = command_runner.commands[0]
    assert command[0:3] == ["python", "-m", "swebench.harness.run_evaluation"]
    assert str(predictions) in command
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

