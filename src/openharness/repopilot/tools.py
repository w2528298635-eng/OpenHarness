from __future__ import annotations

from .models import Phase
from openharness.tools.base import ToolRegistry

_READ_TOOLS = {"read_file", "glob", "grep"}
_WRITE_TOOLS = _READ_TOOLS | {"file_edit", "file_write"}


class ScopedToolRegistry(ToolRegistry):
    @classmethod
    def from_registry(cls, source: ToolRegistry, phase: Phase) -> "ScopedToolRegistry":
        registry = cls()
        if phase in {Phase.ANALYZE, Phase.REPLAN}:
            names = _READ_TOOLS
        elif phase in {Phase.EXECUTE, Phase.REPAIR}:
            names = _WRITE_TOOLS
        else:
            names = set()
        lsp_allowed = phase in {Phase.ANALYZE, Phase.REPLAN, Phase.EXECUTE, Phase.REPAIR}
        for tool in source.list_tools():
            if tool.name in names or (lsp_allowed and tool.name.startswith("lsp_")):
                registry.register(tool)
        return registry
