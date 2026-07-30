from __future__ import annotations

from pathlib import Path

from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
)
from openharness.repopilot.swebench.repositories import (
    GitCommandResult,
    SelectedRepositoryCache,
)


def _instance(
    *,
    instance_id: str = "django__django-123",
    repo: str = "django/django",
    commit: str = "a" * 40,
) -> PublicInstance:
    return PublicInstance(
        instance_id=instance_id,
        repo=repo,
        base_commit=commit,
        problem_statement="Fix the public issue.",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )


class RecordingGitRunner:
    def __init__(self):
        self.commands: list[tuple[Path, list[str]]] = []
        self.known_commits: set[tuple[Path, str]] = set()

    def run(self, argv: list[str], *, cwd: Path) -> GitCommandResult:
        self.commands.append((cwd, list(argv)))
        if argv[:2] == ["git", "init"]:
            (cwd / ".git").mkdir(parents=True)
        if argv[:3] == ["git", "cat-file", "-e"]:
            commit = argv[3].removesuffix("^{commit}")
            exists = (cwd, commit) in self.known_commits
            return GitCommandResult(
                exit_code=0 if exists else 1,
                stdout="",
                stderr="" if exists else "missing",
            )
        if argv[:4] == ["git", "-c", "http.sslBackend=openssl", "fetch"]:
            commit = argv[-1]
            self.known_commits.add((cwd, commit))
        if argv[:3] == ["git", "worktree", "add"]:
            Path(argv[-2]).mkdir(parents=True)
        return GitCommandResult(exit_code=0, stdout="", stderr="")


def test_prepare_fetches_only_requested_commit_with_partial_shallow_clone(
    tmp_path: Path,
) -> None:
    runner = RecordingGitRunner()
    cache = SelectedRepositoryCache(tmp_path, command_runner=runner)
    instance = _instance()

    worktree = cache.prepare(instance, workspace_id="native-1")

    repo_path = tmp_path / "repositories" / "django__django"
    assert worktree == tmp_path / "worktrees" / "native-1"
    assert (
        repo_path,
        ["git", "config", "http.sslBackend", "openssl"],
    ) in runner.commands
    assert (
        repo_path,
        [
            "git",
            "-c",
            "http.sslBackend=openssl",
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "--no-tags",
            "origin",
            instance.base_commit,
        ],
    ) in runner.commands
    assert not any(
        command[:2] == ["git", "clone"] for _, command in runner.commands
    )
    assert (
        repo_path,
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(worktree),
            instance.base_commit,
        ],
    ) in runner.commands


def test_prepare_reuses_repo_and_fetched_commit_for_repetitions(tmp_path: Path) -> None:
    runner = RecordingGitRunner()
    cache = SelectedRepositoryCache(tmp_path, command_runner=runner)
    instance = _instance()

    cache.prepare(instance, workspace_id="native-1")
    cache.prepare(instance, workspace_id="native-2")

    commands = [command for _, command in runner.commands]
    assert sum(command[:2] == ["git", "init"] for command in commands) == 1
    assert (
        sum(
            command[:4] == ["git", "-c", "http.sslBackend=openssl", "fetch"]
            for command in commands
        )
        == 1
    )
    assert sum(command[:3] == ["git", "worktree", "add"] for command in commands) == 2


def test_prepare_does_not_touch_unselected_repository(tmp_path: Path) -> None:
    runner = RecordingGitRunner()
    cache = SelectedRepositoryCache(tmp_path, command_runner=runner)

    cache.prepare(_instance(), workspace_id="selected")

    command_text = "\n".join(
        f"{cwd} {' '.join(command)}" for cwd, command in runner.commands
    )
    assert "sympy" not in command_text
    assert "pytest-dev" not in command_text
