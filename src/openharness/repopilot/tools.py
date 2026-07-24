from __future__ import annotations

import fnmatch
from pathlib import Path

from pydantic import BaseModel

from openharness.tools.base import (
    BaseTool,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)

from .models import Phase

_READ_TOOLS = {"read_file", "glob", "grep"}
_WRITE_TOOLS = _READ_TOOLS | {"edit_file", "write_file"}


class ConstrainedWriteTool(BaseTool):
    """Reject writes outside the RepoPilot worktree before delegation."""

    def __init__(
        self,
        delegate: BaseTool,
        cwd: Path,
        allowed_paths: list[str] | None,
    ):
        self.delegate = delegate
        self.cwd = cwd.resolve()
        self.allowed_paths = allowed_paths
        self.name = delegate.name
        self.description = delegate.description
        self.input_model = delegate.input_model

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        raw_path = getattr(arguments, "path", None)
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult(
                output="RepoPilot policy: editing tool requires a path",
                is_error=True,
            )
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        try:
            relative = candidate.resolve().relative_to(self.cwd).as_posix()
        except ValueError:
            return ToolResult(
                output=f"RepoPilot policy: path escapes worktree: {raw_path}",
                is_error=True,
            )
        if relative == ".git" or relative.startswith(".git/"):
            return ToolResult(
                output=f"RepoPilot policy: sensitive path is blocked: {raw_path}",
                is_error=True,
            )
        if relative.startswith(".openharness/repopilot/"):
            return ToolResult(
                output=f"RepoPilot policy: artifact path is blocked: {raw_path}",
                is_error=True,
            )
        if self.allowed_paths is not None and not any(
            fnmatch.fnmatchcase(relative, pattern) for pattern in self.allowed_paths
        ):
            return ToolResult(
                output=f"RepoPilot policy: path is outside allowed_paths: {raw_path}",
                is_error=True,
            )
        return await self.delegate.execute(arguments, context)

    def is_read_only(self, arguments: BaseModel) -> bool:
        return self.delegate.is_read_only(arguments)


class ScopedToolRegistry(ToolRegistry):
    @classmethod
    def from_registry(
        cls,
        source: ToolRegistry,
        phase: Phase,
        *,
        cwd: Path | None = None,
        allowed_paths: list[str] | None = None,
    ) -> ScopedToolRegistry:
        registry = cls()
        if phase in {Phase.ANALYZE, Phase.REPLAN}:
            names = _READ_TOOLS
        elif phase in {Phase.EXECUTE, Phase.REPAIR}:
            names = _WRITE_TOOLS
        else:
            names = set()
        lsp_allowed = phase in {
            Phase.ANALYZE,
            Phase.REPLAN,
            Phase.EXECUTE,
            Phase.REPAIR,
        }
        for tool in source.list_tools():
            if tool.name not in names and not (lsp_allowed and tool.name == "lsp"):
                continue
            if tool.name in {"edit_file", "write_file"}:
                if cwd is None:
                    continue
                registry.register(ConstrainedWriteTool(tool, cwd, allowed_paths))
            else:
                registry.register(tool)
        return registry
