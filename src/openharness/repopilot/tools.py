from __future__ import annotations

from openharness.tools.base import ToolRegistry

from .models import Phase

_READ_TOOLS = {"read_file", "glob", "grep"}
_WRITE_TOOLS = _READ_TOOLS | {"edit_file", "write_file"}


class ScopedToolRegistry(ToolRegistry):
    @classmethod
    def from_registry(cls, source: ToolRegistry, phase: Phase) -> ScopedToolRegistry:
        registry = cls()
        if phase in {Phase.ANALYZE, Phase.REPLAN}:
            names = _READ_TOOLS
        elif phase in {Phase.EXECUTE, Phase.REPAIR}:
            names = _WRITE_TOOLS
        else:
            names = set()
        lsp_allowed = phase in {Phase.ANALYZE, Phase.REPLAN, Phase.EXECUTE, Phase.REPAIR}
        for tool in source.list_tools():
            if tool.name in names or (lsp_allowed and tool.name == "lsp"):
                registry.register(tool)
        return registry
