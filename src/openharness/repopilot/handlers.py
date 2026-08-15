from __future__ import annotations

from collections.abc import Awaitable, Callable

from .models import Phase, RepoRunState, TransitionDecision
from .workflow import PhaseResult, RunContext

PhaseExecutor = Callable[[RepoRunState, Phase], Awaitable[TransitionDecision]]
ExceptionClassifier = Callable[[Phase, Exception], TransitionDecision]


class RepairPhaseHandler:
    """Adapt one RepoPilot repair phase to the generic workflow contract."""

    def __init__(
        self,
        *,
        phase: Phase,
        execute: PhaseExecutor,
        classify_exception: ExceptionClassifier | None = None,
    ):
        self.phase = phase
        self.execute = execute
        self.classify_exception = classify_exception

    async def handle(self, context: RunContext) -> PhaseResult:
        if context.state.phase is not self.phase:
            raise RuntimeError(
                f"{self.phase.value} handler received {context.state.phase.value} state"
            )
        try:
            decision = await self.execute(context.state, self.phase)
        except Exception as exc:
            if self.classify_exception is None:
                raise
            decision = self.classify_exception(self.phase, exc)
        return PhaseResult(
            next_phase=decision.next_phase,
            terminal_reason=decision.terminal_reason,
            detail=decision.detail,
        )
