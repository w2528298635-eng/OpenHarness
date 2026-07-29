from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import Phase, PhaseRunResult, RepoRunState
from .phase_runner import PhaseAgentRunner


class PhaseExecutor(Protocol):
    async def run(
        self,
        phase: Phase,
        state: RepoRunState,
        cwd: Path,
        *,
        diff_summary: str = "",
        retrieved_context: str = "",
        prompt_version: str | None = None,
        context_ids: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> PhaseRunResult: ...


@dataclass(frozen=True)
class ExecutorCall:
    phase: Phase
    cwd: Path
    prompt_version: str | None
    context_ids: tuple[str, ...]
    tool_names: tuple[str, ...]
    has_retrieved_context: bool


class OpenHarnessPhaseExecutor:
    """Provider-backed executor that preserves the existing runner configuration."""

    def __init__(self, runner: PhaseAgentRunner):
        self.runner = runner

    async def run(
        self,
        phase: Phase,
        state: RepoRunState,
        cwd: Path,
        *,
        diff_summary: str = "",
        retrieved_context: str = "",
        prompt_version: str | None = None,
        context_ids: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> PhaseRunResult:
        del prompt_version, context_ids, tool_names
        return await self.runner.run(
            phase,
            state,
            cwd,
            diff_summary=diff_summary,
            retrieved_context=retrieved_context,
        )


class ScriptedPhaseExecutor:
    """Deterministic executor for tests and architecture evaluation."""

    def __init__(self, results: dict[Phase, list[PhaseRunResult]]):
        self._results = defaultdict(deque)
        for phase, phase_results in results.items():
            self._results[phase].extend(phase_results)
        self.calls: list[ExecutorCall] = []

    async def run(
        self,
        phase: Phase,
        state: RepoRunState,
        cwd: Path,
        *,
        diff_summary: str = "",
        retrieved_context: str = "",
        prompt_version: str | None = None,
        context_ids: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> PhaseRunResult:
        del state, diff_summary
        self.calls.append(
            ExecutorCall(
                phase=phase,
                cwd=cwd,
                prompt_version=prompt_version,
                context_ids=tuple(context_ids or ()),
                tool_names=tuple(tool_names or ()),
                has_retrieved_context=bool(retrieved_context),
            )
        )
        if not self._results[phase]:
            raise RuntimeError(f"no scripted result remains for phase {phase.value}")
        result = self._results[phase].popleft()
        if result.phase is not phase:
            raise RuntimeError(
                f"scripted result phase {result.phase.value} does not match {phase.value}"
            )
        return result
