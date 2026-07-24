from pydantic import BaseModel

from openharness.repopilot.models import Phase
from openharness.repopilot.tools import ScopedToolRegistry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from openharness.tools import create_default_tool_registry


class Args(BaseModel):
    value: str = ""


class FakeTool(BaseTool):
    description = "fake"
    input_model = Args

    def __init__(self, name: str):
        self.name = name

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output="ok")


def test_phase_registries_are_filtered_without_mutating_source() -> None:
    source = ToolRegistry()
    for name in ["read_file", "glob", "grep", "edit_file", "write_file", "bash", "lsp"]:
        source.register(FakeTool(name))

    analyze = ScopedToolRegistry.from_registry(source, Phase.ANALYZE)
    execute = ScopedToolRegistry.from_registry(source, Phase.EXECUTE)
    plan = ScopedToolRegistry.from_registry(source, Phase.PLAN)

    assert {tool.name for tool in analyze.list_tools()} == {"read_file", "glob", "grep", "lsp"}
    assert {tool.name for tool in execute.list_tools()} == {
        "read_file", "glob", "grep", "edit_file", "write_file", "lsp"
    }
    assert plan.list_tools() == []
    assert source.get("bash") is not None
    assert execute.get("bash") is None


def test_real_default_registry_exposes_editing_but_not_shell() -> None:
    execute = ScopedToolRegistry.from_registry(
        create_default_tool_registry(), Phase.EXECUTE
    )

    assert {tool.name for tool in execute.list_tools()} == {
        "read_file", "glob", "grep", "edit_file", "write_file", "lsp"
    }
