from pathlib import Path

import pytest

from openharness.repopilot.handlers import RepairPhaseHandler
from openharness.repopilot.models import (
    Phase,
    RepoRunState,
    RepoTaskSpec,
    TransitionDecision,
)
from openharness.repopilot.workflow import RunContext


def _context(tmp_path: Path, phase: Phase) -> RunContext:
    task = RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"])
    return RunContext(
        state=RepoRunState(run_id="r1", task=task, phase=phase),
        workflow_name="repair",
        workflow_version="2",
    )


@pytest.mark.asyncio
async def test_repair_handler_translates_transition_decision(tmp_path: Path) -> None:
    calls: list[Phase] = []

    async def execute(state: RepoRunState, phase: Phase) -> TransitionDecision:
        calls.append(phase)
        assert state.phase is phase
        return TransitionDecision(next_phase=Phase.PLAN, detail="analyzed")

    handler = RepairPhaseHandler(phase=Phase.ANALYZE, execute=execute)

    result = await handler.handle(_context(tmp_path, Phase.ANALYZE))

    assert calls == [Phase.ANALYZE]
    assert result.next_phase is Phase.PLAN
    assert result.detail == "analyzed"


@pytest.mark.asyncio
async def test_repair_handler_converts_phase_exception_to_terminal_result(
    tmp_path: Path,
) -> None:
    async def execute(state: RepoRunState, phase: Phase) -> TransitionDecision:
        del state, phase
        raise ValueError("bad structured output")

    def classify(phase: Phase, error: Exception) -> TransitionDecision:
        assert phase is Phase.ANALYZE
        assert str(error) == "bad structured output"
        return TransitionDecision(
            next_phase=Phase.FAILED,
            terminal_reason="invalid_analysis",
            detail=str(error),
        )

    handler = RepairPhaseHandler(
        phase=Phase.ANALYZE,
        execute=execute,
        classify_exception=classify,
    )

    result = await handler.handle(_context(tmp_path, Phase.ANALYZE))

    assert result.next_phase is Phase.FAILED
    assert result.terminal_reason == "invalid_analysis"
