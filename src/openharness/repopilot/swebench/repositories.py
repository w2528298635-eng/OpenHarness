from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .models import PublicInstance

_REPOSITORY_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitCommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exit_code: int
    stdout: str
    stderr: str


class GitCommandRunner(Protocol):
    def run(self, argv: list[str], *, cwd: Path) -> GitCommandResult: ...


class SubprocessGitCommandRunner:
    def run(self, argv: list[str], *, cwd: Path) -> GitCommandResult:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return GitCommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class RepositoryPreparationError(RuntimeError):
    pass


def _repository_key(repo: str) -> str:
    parts = repo.split("/")
    if len(parts) != 2 or not all(
        _REPOSITORY_COMPONENT.fullmatch(part) for part in parts
    ):
        raise ValueError(f"invalid public GitHub repository name: {repo!r}")
    return "__".join(parts)


def _checked(
    runner: GitCommandRunner,
    argv: list[str],
    *,
    cwd: Path,
) -> GitCommandResult:
    result = runner.run(argv, cwd=cwd)
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise RepositoryPreparationError(f"{' '.join(argv)}: {detail}")
    return result


class SelectedRepositoryCache:
    """Fetch only commits named by the frozen public sample."""

    def __init__(
        self,
        root: Path,
        *,
        command_runner: GitCommandRunner | None = None,
    ):
        self.root = root.resolve()
        self.command_runner = command_runner or SubprocessGitCommandRunner()

    def prepare(self, instance: PublicInstance, *, workspace_id: str) -> Path:
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise ValueError(f"invalid workspace id: {workspace_id!r}")
        repository_key = _repository_key(instance.repo)
        repositories_root = self.root / "repositories"
        worktrees_root = self.root / "worktrees"
        repository_path = repositories_root / repository_key
        worktree_path = worktrees_root / workspace_id
        repositories_root.mkdir(parents=True, exist_ok=True)
        worktrees_root.mkdir(parents=True, exist_ok=True)
        if worktree_path.exists():
            raise RepositoryPreparationError(
                f"worktree already exists: {worktree_path}"
            )

        repository_is_usable = False
        if (repository_path / ".git").exists():
            repository_check = self.command_runner.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=repository_path,
            )
            repository_is_usable = (
                repository_check.exit_code == 0
                and repository_check.stdout.strip() == "true"
            )
        if not repository_is_usable:
            repository_path.mkdir(parents=True, exist_ok=True)
            _checked(
                self.command_runner,
                ["git", "init"],
                cwd=repository_path,
            )
            _checked(
                self.command_runner,
                [
                    "git",
                    "remote",
                    "add",
                    "origin",
                    f"https://github.com/{instance.repo}.git",
                ],
                cwd=repository_path,
            )

        # All selected repositories use blob-less fetches. These settings are
        # harmless for a healthy cache and repair a previously interrupted one:
        # Git can then obtain any missing blob during worktree checkout.
        _checked(
            self.command_runner,
            ["git", "config", "remote.origin.promisor", "true"],
            cwd=repository_path,
        )
        _checked(
            self.command_runner,
            [
                "git",
                "config",
                "remote.origin.partialclonefilter",
                "blob:none",
            ],
            cwd=repository_path,
        )
        _checked(
            self.command_runner,
            ["git", "config", "extensions.partialClone", "origin"],
            cwd=repository_path,
        )

        _checked(
            self.command_runner,
            ["git", "config", "http.sslBackend", "openssl"],
            cwd=repository_path,
        )
        commit_check = self.command_runner.run(
            ["git", "cat-file", "-e", f"{instance.base_commit}^{{commit}}"],
            cwd=repository_path,
        )
        if commit_check.exit_code != 0:
            _checked(
                self.command_runner,
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
                cwd=repository_path,
            )

        _checked(
            self.command_runner,
            [
                "git",
                "worktree",
                "add",
                "--detach",
                str(worktree_path),
                instance.base_commit,
            ],
            cwd=repository_path,
        )
        return worktree_path

    def release(self, instance: PublicInstance, *, workspace_id: str) -> None:
        """Remove one disposable task worktree while retaining shared Git objects."""
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise ValueError(f"invalid workspace id: {workspace_id!r}")
        repository_path = self.root / "repositories" / _repository_key(instance.repo)
        worktree_path = self.root / "worktrees" / workspace_id
        if not worktree_path.exists():
            return
        _checked(
            self.command_runner,
            [
                "git",
                "worktree",
                "remove",
                "--force",
                str(worktree_path),
            ],
            cwd=repository_path,
        )
