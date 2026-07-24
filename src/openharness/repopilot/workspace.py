from __future__ import annotations

import asyncio
import fnmatch
import hashlib
from pathlib import Path
from typing import Any

from openharness.swarm.worktree import WorktreeManager


class WorkspaceManager:
    def __init__(self, worktrees: Any | None = None):
        self.worktrees = worktrees

    async def create(self, repo_path: Path, run_id: str):
        manager = self.worktrees
        if manager is None:
            manager = WorktreeManager(
                repo_path.resolve().parent / ".openharness-repopilot-worktrees" / repo_path.name
            )
            self.worktrees = manager
        return await manager.create_worktree(
            repo_path.resolve(),
            run_id,
            branch=f"repopilot/{run_id}",
            agent_id=run_id,
        )

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
        return await self._git(worktree, "diff", "--binary", "--no-ext-diff")

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
