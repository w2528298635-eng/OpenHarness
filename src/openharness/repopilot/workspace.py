from __future__ import annotations

import asyncio
import difflib
import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.swarm.worktree import WorktreeManager


@dataclass(frozen=True)
class WorkspaceLease:
    run_id: str
    slug: str
    path: Path
    branch: str
    original_path: Path
    base_dir: Path


class WorkspaceManager:
    def __init__(
        self,
        worktrees: Any | None = None,
        *,
        base_path: Path | None = None,
    ):
        self.worktrees = worktrees
        configured = os.environ.get("OPENHARNESS_REPOPILOT_WORKTREE_ROOT")
        self.base_path = (
            base_path.resolve()
            if base_path is not None
            else Path(configured).expanduser().resolve()
            if configured
            else Path.home() / ".openharness" / "repopilot" / "worktrees"
        )
        self._managers: dict[str, Any] = {}

    async def create(self, repo_path: Path, run_id: str) -> WorkspaceLease:
        repo_path = repo_path.resolve()
        repo_key = hashlib.sha256(str(repo_path).encode()).hexdigest()[:10]
        base_dir = self.base_path / repo_key
        manager = self.worktrees
        if manager is None:
            manager = self._managers.get(repo_key)
            if manager is None:
                manager = WorktreeManager(base_dir)
                self._managers[repo_key] = manager
        slug = hashlib.sha256(run_id.encode()).hexdigest()[:12]
        info = await manager.create_worktree(
            repo_path,
            slug,
            branch=f"repopilot/{run_id}",
            agent_id=run_id,
        )
        return WorkspaceLease(
            run_id=run_id,
            slug=slug,
            path=Path(info.path).resolve(),
            branch=info.branch,
            original_path=repo_path,
            base_dir=Path(getattr(manager, "base_dir", base_dir)).resolve(),
        )

    async def cleanup(
        self,
        repo_path: Path,
        worktree_path: Path,
        *,
        force: bool = False,
    ) -> None:
        repo = repo_path.resolve()
        worktree = worktree_path.resolve()
        if worktree == repo:
            raise ValueError("refusing to remove the original repository")
        common_raw = (await self._git(worktree, "rev-parse", "--git-common-dir")).strip()
        common = Path(common_raw)
        if not common.is_absolute():
            common = worktree / common
        registered_repo = common.resolve().parent
        if registered_repo != repo:
            raise ValueError(
                f"worktree belongs to {registered_repo}, not requested repository {repo}"
            )
        changes = await self.changed_files(worktree)
        if changes and not force:
            raise RuntimeError("worktree has uncommitted changes; pass force=True to remove it")
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(worktree))
        await self._git(repo, *args)
        await self.prune(repo)

    async def prune(self, repo_path: Path) -> None:
        await self._git(repo_path.resolve(), "worktree", "prune")

    async def _git(self, cwd: Path, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        return stdout.decode(errors="replace")

    async def diff(self, worktree: Path) -> str:
        tracked = await self._git(worktree, "diff", "--binary", "--no-ext-diff")
        untracked_output = await self._git(worktree, "ls-files", "--others", "--exclude-standard")
        additions: list[str] = []
        for relative in untracked_output.splitlines():
            normalized = relative.replace("\\", "/")
            target = worktree / relative
            if not target.is_file():
                continue
            header = f"diff --git a/{normalized} b/{normalized}\nnew file mode 100644\n"
            try:
                content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                additions.append(header + f"Binary files /dev/null and b/{normalized} differ\n")
                continue
            patch = "".join(
                difflib.unified_diff(
                    [],
                    content.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=f"b/{normalized}",
                )
            )
            additions.append(header + patch)
        return tracked + "".join(additions)

    async def changed_files(self, worktree: Path) -> list[str]:
        output = await self._git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
        result: list[str] = []
        for line in output.splitlines():
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            result.append(path.replace("\\", "/").strip('"'))
        return sorted(set(result))

    def validate_changed_files(
        self, paths: list[str], allowed_paths: list[str] | None
    ) -> str | None:
        for raw_path in paths:
            path = raw_path.replace("\\", "/")
            while path.startswith("./"):
                path = path[2:]
            parts = Path(path).parts
            if ".." in parts or path == ".git" or path.startswith(".git/"):
                return f"sensitive path changed: {raw_path}"
            if path.startswith(".openharness/repopilot/"):
                return f"run artifact path changed: {raw_path}"
            if allowed_paths is not None and not any(
                fnmatch.fnmatchcase(path, pattern) for pattern in allowed_paths
            ):
                return f"path is outside allowed_paths: {raw_path}"
        return None

    @staticmethod
    def diff_signature(diff: str) -> str:
        return hashlib.sha256(diff.encode("utf-8")).hexdigest()
