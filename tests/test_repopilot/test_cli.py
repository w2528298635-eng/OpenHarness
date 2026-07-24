from typer.testing import CliRunner

from openharness.cli import app


runner = CliRunner()


def test_repopilot_help_lists_workflow_commands() -> None:
    result = runner.invoke(app, ["repopilot", "--help"])

    assert result.exit_code == 0
    for command in ["run", "show", "resume", "report", "benchmark"]:
        assert command in result.stdout


def test_run_rejects_missing_task_file() -> None:
    result = runner.invoke(app, ["repopilot", "run", "missing.yaml"])

    assert result.exit_code != 0
    assert "does not exist" in result.stderr
