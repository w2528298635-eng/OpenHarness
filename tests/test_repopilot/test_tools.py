import pytest
from pydantic import BaseModel

from openharness.repopilot.models import Phase
from openharness.repopilot.tools import ScopedToolRegistry
from openharness.tools import create_default_tool_registry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


class Args(BaseModel):
    path: str = ""


class FakeTool(BaseTool):
    description = "fake"
    input_model = Args

    def __init__(self, name: str):
        self.name = name

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output="ok")


def test_phase_registries_are_filtered_without_mutating_source(tmp_path) -> None:
    source = ToolRegistry()
    for name in ["read_file", "glob", "grep", "edit_file", "write_file", "bash", "lsp"]:
        source.register(FakeTool(name))

    analyze = ScopedToolRegistry.from_registry(source, Phase.ANALYZE)
    execute = ScopedToolRegistry.from_registry(source, Phase.EXECUTE, cwd=tmp_path)
    plan = ScopedToolRegistry.from_registry(source, Phase.PLAN)

    assert {tool.name for tool in analyze.list_tools()} == {"read_file", "glob", "grep", "lsp"}
    assert {tool.name for tool in execute.list_tools()} == {
        "read_file",
        "glob",
        "grep",
        "edit_file",
        "write_file",
        "lsp",
    }
    assert plan.list_tools() == []
    assert source.get("bash") is not None
    assert execute.get("bash") is None


def test_real_default_registry_exposes_editing_but_not_shell(tmp_path) -> None:
    execute = ScopedToolRegistry.from_registry(
        create_default_tool_registry(), Phase.EXECUTE, cwd=tmp_path
    )

    assert {tool.name for tool in execute.list_tools()} == {
        "read_file",
        "glob",
        "grep",
        "edit_file",
        "write_file",
        "lsp",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../outside.py", r"C:\outside.py", "tests/outside.py"])
async def test_editing_tools_block_paths_outside_worktree_policy(tmp_path, path: str) -> None:
    source = ToolRegistry()
    source.register(FakeTool("write_file"))
    execute = ScopedToolRegistry.from_registry(
        source,
        Phase.EXECUTE,
        cwd=tmp_path,
        allowed_paths=["src/**"],
    )
    tool = execute.get("write_file")

    result = await tool.execute(
        Args(path=path),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error
    assert "RepoPilot policy" in result.output


@pytest.mark.asyncio
async def test_editing_tools_allow_in_scope_worktree_path(tmp_path) -> None:
    source = ToolRegistry()
    source.register(FakeTool("write_file"))
    execute = ScopedToolRegistry.from_registry(
        source,
        Phase.EXECUTE,
        cwd=tmp_path,
        allowed_paths=["src/**"],
    )
    tool = execute.get("write_file")

    result = await tool.execute(
        Args(path="src/app.py"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert not result.is_error
