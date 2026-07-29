from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .events import RunEvent, RunEventKind
from .models import Phase, RepoRunState, RepoTaskSpec, utc_now


@dataclass
class RunContext:
    state: RepoRunState
    workflow_name: str
    workflow_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PhaseResult(BaseModel):
    next_phase: Phase
    updates: dict[str, Any] = Field(default_factory=dict)
    terminal_reason: str | None = None
    detail: str = ""


class PhaseHandler(Protocol):
    async def handle(self, context: RunContext) -> PhaseResult: ...


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    version: str
    initial_phase: Phase
    terminal_phases: frozenset[Phase]
    handlers: Mapping[Phase, PhaseHandler]

    def __post_init__(self) -> None:
        if self.initial_phase in self.terminal_phases:
            raise ValueError("initial phase must not be terminal")
        if self.initial_phase not in self.handlers:
            raise ValueError("initial phase must have a handler")
        terminal_handlers = self.terminal_phases.intersection(self.handlers)
        if terminal_handlers:
            names = ", ".join(sorted(phase.value for phase in terminal_handlers))
            raise ValueError(f"terminal phases must not have handlers: {names}")
        object.__setattr__(self, "handlers", MappingProxyType(dict(self.handlers)))


StateFactory = Callable[[RepoTaskSpec], RepoRunState | Awaitable[RepoRunState]]
Checkpoint = Callable[[RepoRunState], None | Awaitable[None]]
EventSink = Callable[[RunEvent], Any | Awaitable[Any]]
StatePredicate = Callable[[RepoRunState], bool | Awaitable[bool]]
BudgetCheck = Callable[[RepoRunState], str | None | Awaitable[str | None]]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class WorkflowRuntime:
    def __init__(
        self,
        *,
        definition: WorkflowDefinition,
        create_state: StateFactory,
        checkpoint: Checkpoint,
        emit: EventSink | None = None,
        is_cancelled: StatePredicate | None = None,
        budget_check: BudgetCheck | None = None,
    ):
        self.definition = definition
        self.create_state = create_state
        self.checkpoint = checkpoint
        self.emit = emit
        self.is_cancelled = is_cancelled
        self.budget_check = budget_check

    async def start(self, task: RepoTaskSpec) -> RepoRunState:
        state = await _resolve(self.create_state(task))
        if state.phase != self.definition.initial_phase:
            raise ValueError(
                f"new state phase {state.phase.value} does not match "
                f"{self.definition.initial_phase.value}"
            )
        await self._emit(
            RunEvent.create(
                run_id=state.run_id,
                kind=RunEventKind.RUN_STARTED,
                phase=state.phase.value,
                data={
                    "workflow": self.definition.name,
                    "workflow_version": self.definition.version,
                },
            )
        )
        return await self._run(state)

    async def resume(self, state: RepoRunState) -> RepoRunState:
        return await self._run(state)

    async def _run(self, state: RepoRunState) -> RepoRunState:
        while state.phase not in self.definition.terminal_phases:
            if self.is_cancelled is not None and await _resolve(self.is_cancelled(state)):
                state = state.model_copy(
                    update={
                        "phase": Phase.FAILED,
                        "terminal_reason": "cancelled",
                        "completed_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                )
                await self._emit(
                    RunEvent.create(
                        run_id=state.run_id,
                        kind=RunEventKind.CANCELLATION,
                        phase=state.phase.value,
                    )
                )
                await self._checkpoint(state)
                break

            if self.budget_check is not None:
                exhausted_reason = await _resolve(self.budget_check(state))
                if exhausted_reason:
                    state = state.model_copy(
                        update={
                            "phase": Phase.FAILED,
                            "terminal_reason": exhausted_reason,
                            "completed_at": utc_now(),
                            "updated_at": utc_now(),
                        }
                    )
                    await self._checkpoint(state)
                    break

            handler = self.definition.handlers.get(state.phase)
            if handler is None:
                raise RuntimeError(
                    f"workflow {self.definition.name!r} has no handler for {state.phase.value}"
                )
            current_phase = state.phase
            await self._emit(
                RunEvent.create(
                    run_id=state.run_id,
                    kind=RunEventKind.PHASE_STARTED,
                    phase=current_phase.value,
                )
            )
            context = RunContext(
                state=state,
                workflow_name=self.definition.name,
                workflow_version=self.definition.version,
            )
            result = await handler.handle(context)
            await self._emit(
                RunEvent.create(
                    run_id=state.run_id,
                    kind=RunEventKind.PHASE_FINISHED,
                    phase=current_phase.value,
                    data={"detail": result.detail},
                )
            )
            updates = dict(result.updates)
            updates.update(
                {
                    "phase": result.next_phase,
                    "terminal_reason": result.terminal_reason,
                    "updated_at": utc_now(),
                }
            )
            if result.next_phase in self.definition.terminal_phases:
                updates["completed_at"] = utc_now()
            state = state.model_copy(update=updates)
            await self._emit(
                RunEvent.create(
                    run_id=state.run_id,
                    kind=RunEventKind.TRANSITION,
                    phase=current_phase.value,
                    data={
                        "from": current_phase.value,
                        "to": result.next_phase.value,
                        "reason": result.terminal_reason,
                    },
                )
            )
            await self._checkpoint(state)

        await self._emit(
            RunEvent.create(
                run_id=state.run_id,
                kind=RunEventKind.RUN_FINISHED,
                phase=state.phase.value,
                data={"terminal_reason": state.terminal_reason},
            )
        )
        return state

    async def _checkpoint(self, state: RepoRunState) -> None:
        await _resolve(self.checkpoint(state))
        await self._emit(
            RunEvent.create(
                run_id=state.run_id,
                kind=RunEventKind.CHECKPOINT,
                phase=state.phase.value,
            )
        )

    async def _emit(self, event: RunEvent) -> None:
        if self.emit is not None:
            await _resolve(self.emit(event))
