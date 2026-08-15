from pathlib import Path

import pytest

from openharness.repopilot.task_loader import load_task, validate_verify_command


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "-q"],
        ["py.test", "tests/test_x.py"],
        ["python", "-m", "pytest", "-q"],
        [r"C:\Python311\python.exe", "-m", "pytest"],
    ],
)
def test_accepts_pytest_argv(argv: list[str]) -> None:
    assert validate_verify_command(argv) == argv


@pytest.mark.parametrize(
    "argv",
    [[], ["bash", "-c", "pytest"], ["uv", "run", "pytest"], ["pytest && whoami"]],
)
def test_rejects_unsupported_or_shell_commands(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        validate_verify_command(argv)


def test_load_task_resolves_repo_relative_to_yaml(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    task_file = tmp_path / "task.yaml"
    task_file.write_text(
        "repo_path: repo\nissue: discount boundary is broken\n"
        "verify_command: [python, -m, pytest, -q]\n",
        encoding="utf-8",
    )

    task = load_task(task_file)

    assert task.repo_path == repo.resolve()
