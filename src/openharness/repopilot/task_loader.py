from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import RepoTaskSpec

_SHELL_META = re.compile(r"[;&|`<>]")


def validate_verify_command(argv: list[str]) -> list[str]:
    if not argv or any(not isinstance(part, str) or not part.strip() for part in argv):
        raise ValueError("verify_command must be a non-empty argv")
    if any(_SHELL_META.search(part) for part in argv):
        raise ValueError("shell metacharacters are not allowed in verify_command")

    executable = Path(argv[0]).name.lower()
    direct = executable in {"pytest", "pytest.exe", "py.test", "py.test.exe"}
    python_module = (
        executable.startswith("python") and len(argv) >= 3 and argv[1:3] == ["-m", "pytest"]
    )
    if not (direct or python_module):
        raise ValueError("verify_command must start with pytest, py.test, or <python> -m pytest")
    return list(argv)


def load_task(path: Path) -> RepoTaskSpec:
    task_path = path.expanduser().resolve()
    payload = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("task YAML must contain a mapping")
    repo_value = payload.get("repo_path")
    if not isinstance(repo_value, str):
        raise TypeError("repo_path must be a string")
    repo_path = Path(repo_value).expanduser()
    if not repo_path.is_absolute():
        repo_path = task_path.parent / repo_path
    repo_path = repo_path.resolve()
    if not repo_path.is_dir() or not (repo_path / ".git").exists():
        raise ValueError(f"repo_path is not a Git repository: {repo_path}")
    payload["repo_path"] = repo_path
    payload["verify_command"] = validate_verify_command(payload.get("verify_command", []))
    return RepoTaskSpec.model_validate(payload)
