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


def test_swebench_help_lists_six_workflow_commands() -> None:
    result = runner.invoke(app, ["repopilot", "swebench", "--help"])

    assert result.exit_code == 0
    for command in ["doctor", "prepare", "pilot", "run", "resume", "report"]:
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

