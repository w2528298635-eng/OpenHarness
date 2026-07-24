from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.stream_events import AssistantTurnComplete, ToolExecutionCompleted, ToolExecutionStarted
from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.phase_runner import OpenHarnessPhaseRunner
from openharness.tools.base import ToolRegistry


ANALYSIS_JSON = (
    '{"suspected_files":["app.py"],"root_cause":"off by one",'
    '"evidence":[{"file":"app.py","observation":"bad boundary"}],"confidence":0.9}'
)


class FakeEngine:
    def __init__(self, answers: list[str]):
        self.answers = answers
        self._tool_registry = None

    async def submit_message(self, prompt):
        answer = self.answers.pop(0)
        yield ToolExecutionStarted("read_file", {"path": "app.py"})
        yield ToolExecutionCompleted("read_file", "source")
        yield AssistantTurnComplete(
            ConversationMessage(role="assistant", content=[TextBlock(text=answer)]),
            UsageSnapshot(input_tokens=10, output_tokens=5),
        )


def _state(tmp_path: Path) -> RepoRunState:
    return RepoRunState(
        run_id="run",
        task=RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"]),
    )


@pytest.mark.asyncio
async def test_runner_parses_structured_output_and_captures_trace(tmp_path: Path) -> None:
    bundles = []

    async def factory(**kwargs):
        bundle = SimpleNamespace(engine=FakeEngine([ANALYSIS_JSON]), tool_registry=ToolRegistry())
        bundles.append(bundle)
        return bundle

    result = await OpenHarnessPhaseRunner(runtime_factory=factory).run(
        Phase.ANALYZE, _state(tmp_path), tmp_path
    )

    assert result.structured["root_cause"] == "off by one"
    assert result.tokens_used == 15
    assert result.actions[0].action_type == "read_file"
    assert result.observations[0].summary == "source"


@pytest.mark.asyncio
async def test_runner_retries_invalid_json_once_and_builds_fresh_runtime(tmp_path: Path) -> None:
    created = []

    async def factory(**kwargs):
        answers = ["not json", ANALYSIS_JSON] if not created else [ANALYSIS_JSON]
        bundle = SimpleNamespace(engine=FakeEngine(answers), tool_registry=ToolRegistry())
        created.append(bundle)
        return bundle

    runner = OpenHarnessPhaseRunner(runtime_factory=factory)
    first = await runner.run(Phase.ANALYZE, _state(tmp_path), tmp_path)
    second = await runner.run(Phase.ANALYZE, _state(tmp_path), tmp_path)

    assert first.structured and second.structured
    assert len(created) == 2
