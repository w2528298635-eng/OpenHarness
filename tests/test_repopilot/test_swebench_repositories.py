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
        if argv == ["git", "rev-parse", "--is-inside-work-tree"]:
            return GitCommandResult(
                exit_code=0 if (cwd / ".git").exists() else 128,
                stdout="true\n" if (cwd / ".git").exists() else "",
                stderr="" if (cwd / ".git").exists() else "not a repository",
            )
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
        if argv[:3] == ["git", "worktree", "remove"]:
            Path(argv[-1]).rmdir()
        return GitCommandResult(exit_code=0, stdout="", stderr="")


class InvalidMetadataGitRunner(RecordingGitRunner):
    """Simulate an interrupted cache retaining objects but no usable Git metadata."""

    def run(self, argv: list[str], *, cwd: Path) -> GitCommandResult:
        self.commands.append((cwd, list(argv)))
        if argv[:2] == ["git", "init"]:
            (cwd / ".git").mkdir(parents=True, exist_ok=True)
            (cwd / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            return GitCommandResult(exit_code=0, stdout="", stderr="")
        if argv == ["git", "rev-parse", "--is-inside-work-tree"]:
            valid = (cwd / ".git" / "config").exists()
            return GitCommandResult(
                exit_code=0 if valid else 128,
                stdout="true\n" if valid else "",
                stderr="" if valid else "fatal: not a git repository",
            )
        return super().run(argv, cwd=cwd)


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


def test_prepare_recovers_interrupted_repository_metadata(tmp_path: Path) -> None:
    runner = InvalidMetadataGitRunner()
    repo_path = tmp_path / "repositories" / "django__django"
    (repo_path / ".git" / "objects").mkdir(parents=True)
    cache = SelectedRepositoryCache(tmp_path, command_runner=runner)

    cache.prepare(_instance(), workspace_id="recovered")

    commands = [command for _, command in runner.commands]
    assert ["git", "rev-parse", "--is-inside-work-tree"] in commands
    assert ["git", "init"] in commands
    assert ["git", "remote", "add", "origin", "https://github.com/django/django.git"] in commands
    assert ["git", "config", "remote.origin.promisor", "true"] in commands
    assert [
        "git",
        "config",
        "remote.origin.partialclonefilter",
        "blob:none",
    ] in commands


def test_prepare_does_not_touch_unselected_repository(tmp_path: Path) -> None:
    runner = RecordingGitRunner()
    cache = SelectedRepositoryCache(tmp_path, command_runner=runner)

    cache.prepare(_instance(), workspace_id="selected")

    command_text = "\n".join(
        f"{cwd} {' '.join(command)}" for cwd, command in runner.commands
    )
    assert "sympy" not in command_text
    assert "pytest-dev" not in command_text


def test_release_removes_only_requested_temporary_worktree(tmp_path: Path) -> None:
    runner = RecordingGitRunner()
    cache = SelectedRepositoryCache(tmp_path, command_runner=runner)
    instance = _instance()
    selected = cache.prepare(instance, workspace_id="native-1")
    other = cache.prepare(instance, workspace_id="native-2")

    cache.release(instance, workspace_id="native-1")

    repo_path = tmp_path / "repositories" / "django__django"
    assert not selected.exists()
    assert other.exists()
    assert (
        repo_path,
        ["git", "worktree", "remove", "--force", str(selected)],
    ) in runner.commands
