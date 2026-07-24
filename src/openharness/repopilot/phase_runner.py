from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from openharness.engine.stream_events import (
    AssistantTurnComplete,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.ui.runtime import build_runtime, close_runtime

from .models import (
    ActionRecord,
    AnalysisResult,
    ObservationRecord,
    Phase,
    PhaseRunResult,
    RepairPlan,
    RepoRunState,
)
from .prompts import build_phase_prompt
from .tools import ScopedToolRegistry

RuntimeFactory = Callable[..., Awaitable[Any]]


class PhaseAgentRunner(Protocol):
    async def run(
        self, phase: Phase, state: RepoRunState, cwd: Path, *, diff_summary: str = ""
    ) -> PhaseRunResult: ...


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("phase output must be a JSON object")
    return value


class OpenHarnessPhaseRunner:
    def __init__(
        self,
        runtime_factory: RuntimeFactory = build_runtime,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_format: str | None = None,
    ):
        self.runtime_factory = runtime_factory
        self.runtime_options = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "api_format": api_format,
        }

    async def run(
        self, phase: Phase, state: RepoRunState, cwd: Path, *, diff_summary: str = ""
    ) -> PhaseRunResult:
        prompt = build_phase_prompt(phase, state, diff_summary=diff_summary)
        bundle = await self.runtime_factory(
            prompt=prompt,
            cwd=str(cwd),
            max_turns=12,
            permission_mode="full_auto",
            include_project_memory=False,
            **self.runtime_options,
        )
        scoped = ScopedToolRegistry.from_registry(bundle.tool_registry, phase)
        bundle.tool_registry = scoped
        bundle.engine._tool_registry = scoped
        try:
            return await self._consume(bundle.engine, phase, prompt)
        finally:
            if hasattr(bundle, "mcp_manager"):
                await close_runtime(bundle)

    async def _consume(self, engine: Any, phase: Phase, prompt: str) -> PhaseRunResult:
        schema: type[BaseModel] | None = None
        if phase is Phase.ANALYZE:
            schema = AnalysisResult
        elif phase in {Phase.PLAN, Phase.REPLAN}:
            schema = RepairPlan

        actions: list[ActionRecord] = []
        observations: list[ObservationRecord] = []
        total_tokens = 0
        final_text = ""
        pending_action_ids: list[str] = []

        for output_attempt in range(2):
            current_text = ""
            async for event in engine.submit_message(
                prompt
                if output_attempt == 0
                else "上一次输出不是有效的目标 JSON。只重新输出一个符合 schema 的 JSON 对象。"
            ):
                if isinstance(event, ToolExecutionStarted):
                    action_id = uuid4().hex
                    pending_action_ids.append(action_id)
                    actions.append(
                        ActionRecord(
                            action_id=action_id,
                            phase=phase,
                            action_type=event.tool_name,
                            parameters=event.tool_input,
                            source="model",
                        )
                    )
                elif isinstance(event, ToolExecutionCompleted):
                    action_id = (
                        pending_action_ids.pop(0) if pending_action_ids else uuid4().hex
                    )
                    observations.append(
                        ObservationRecord(
                            action_id=action_id,
                            status="failure" if event.is_error else "success",
                            summary=event.output,
                            metadata=event.metadata or {},
                        )
                    )
                elif isinstance(event, AssistantTurnComplete):
                    current_text = event.message.text
                    total_tokens += event.usage.total_tokens
                elif isinstance(event, ErrorEvent) and not event.recoverable:
                    raise RuntimeError(event.message)
            final_text = current_text
            if schema is None:
                return PhaseRunResult(
                    phase=phase,
                    final_text=final_text,
                    tokens_used=total_tokens,
                    actions=actions,
                    observations=observations,
                )
            try:
                validated = schema.model_validate(_extract_json(final_text))
                return PhaseRunResult(
                    phase=phase,
                    structured=validated.model_dump(mode="json"),
                    final_text=final_text,
                    tokens_used=total_tokens,
                    actions=actions,
                    observations=observations,
                )
            except (ValueError, json.JSONDecodeError, ValidationError):
                if output_attempt == 1:
                    raise ValueError(f"invalid structured output for {phase.value}")
        raise AssertionError("unreachable")
