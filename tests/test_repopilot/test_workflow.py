from pathlib import Path

import pytest

from openharness.repopilot.events import RunEvent, RunEventKind
from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.workflow import (
    PhaseResult,
    RunContext,
    WorkflowDefinition,
    WorkflowRuntime,
)


class RecordingHandler:
    def __init__(self, name: str, next_phase: Phase, calls: list[str]):
        self.name = name
        self.next_phase = next_phase
        self.calls = calls

    async def handle(self, context: RunContext) -> PhaseResult:
        self.calls.append(self.name)
        return PhaseResult(
            next_phase=self.next_phase,
            updates={"changed_files": [*context.state.changed_files, self.name]},
        )


def _task(path: Path) -> RepoTaskSpec:
    return RepoTaskSpec(repo_path=path, issue="broken", verify_command=["pytest"])


def _state(task: RepoTaskSpec, phase: Phase = Phase.PRECHECK) -> RepoRunState:
    return RepoRunState(run_id="r1", task=task, phase=phase)


@pytest.mark.asyncio
async def test_runtime_executes_handlers_checkpoints_and_stops_at_terminal(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    checkpoints: list[Phase] = []
    events: list[RunEvent] = []
    definition = WorkflowDefinition(
        name="test",
        version="1",
        initial_phase=Phase.PRECHECK,
        terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
        handlers={
            Phase.PRECHECK: RecordingHandler("precheck", Phase.ANALYZE, calls),
            Phase.ANALYZE: RecordingHandler("analyze", Phase.COMPLETE, calls),
        },
    )
    runtime = WorkflowRuntime(
        definition=definition,
        create_state=lambda task: _state(task),
        checkpoint=lambda state: checkpoints.append(state.phase),
        emit=events.append,
    )

    state = await runtime.start(_task(tmp_path))

    assert calls == ["precheck", "analyze"]
    assert state.phase is Phase.COMPLETE
    assert state.changed_files == ["precheck", "analyze"]
    assert checkpoints == [Phase.ANALYZE, Phase.COMPLETE]
    assert [event.kind for event in events] == [
        RunEventKind.RUN_STARTED,
        RunEventKind.PHASE_STARTED,
        RunEventKind.PHASE_FINISHED,
        RunEventKind.TRANSITION,
        RunEventKind.CHECKPOINT,
        RunEventKind.PHASE_STARTED,
        RunEventKind.PHASE_FINISHED,
        RunEventKind.TRANSITION,
        RunEventKind.CHECKPOINT,
        RunEventKind.RUN_FINISHED,
    ]


@pytest.mark.asyncio
async def test_resume_starts_at_persisted_phase_without_replaying_handlers(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    definition = WorkflowDefinition(
        name="test",
        version="1",
        initial_phase=Phase.PRECHECK,
        terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
        handlers={
            Phase.PRECHECK: RecordingHandler("precheck", Phase.ANALYZE, calls),
            Phase.ANALYZE: RecordingHandler("analyze", Phase.COMPLETE, calls),
        },
    )
    runtime = WorkflowRuntime(
        definition=definition,
        create_state=lambda task: _state(task),
        checkpoint=lambda state: None,
    )

    state = await runtime.resume(_state(_task(tmp_path), phase=Phase.ANALYZE))

    assert state.phase is Phase.COMPLETE
    assert calls == ["analyze"]


@pytest.mark.asyncio
async def test_cancellation_stops_before_next_handler(tmp_path: Path) -> None:
    calls: list[str] = []
    checkpoints: list[Phase] = []
    definition = WorkflowDefinition(
        name="test",
        version="1",
        initial_phase=Phase.PRECHECK,
        terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
        handlers={
            Phase.PRECHECK: RecordingHandler("precheck", Phase.ANALYZE, calls),
            Phase.ANALYZE: RecordingHandler("analyze", Phase.COMPLETE, calls),
        },
    )
    runtime = WorkflowRuntime(
        definition=definition,
        create_state=lambda task: _state(task),
        checkpoint=lambda state: checkpoints.append(state.phase),
        is_cancelled=lambda state: state.phase is Phase.ANALYZE,
    )

    state = await runtime.start(_task(tmp_path))

    assert calls == ["precheck"]
    assert state.phase is Phase.FAILED
    assert state.terminal_reason == "cancelled"
    assert checkpoints == [Phase.ANALYZE, Phase.FAILED]


def test_definition_rejects_terminal_handlers_and_missing_initial_handler() -> None:
    with pytest.raises(ValueError, match="initial phase"):
        WorkflowDefinition(
            name="broken",
            version="1",
            initial_phase=Phase.PRECHECK,
            terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
            handlers={},
        )
    with pytest.raises(ValueError, match="terminal"):
        WorkflowDefinition(
            name="broken",
            version="1",
            initial_phase=Phase.PRECHECK,
            terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
            handlers={
                Phase.PRECHECK: RecordingHandler("precheck", Phase.COMPLETE, []),
                Phase.COMPLETE: RecordingHandler("complete", Phase.COMPLETE, []),
            },
        )
