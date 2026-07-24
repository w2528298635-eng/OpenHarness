from typer.testing import CliRunner

from openharness.cli import app
from openharness.repopilot.cli import _scheduler

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


def test_scheduler_passes_openai_compatible_environment_to_phase_runner(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENHARNESS_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("OPENHARNESS_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENHARNESS_API_FORMAT", "openai")
    monkeypatch.setenv("OPENHARNESS_OPENAI_API_KEY", "secret-value")

    scheduler = _scheduler(tmp_path)

    assert scheduler.phase_runner.runtime_options == {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": "secret-value",
        "api_format": "openai",
    }
