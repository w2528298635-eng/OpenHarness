from __future__ import annotations

import re
from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .models import utc_now

_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_OPENAI_STYLE_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")
_SENSITIVE_KEYS = frozenset({"authorization", "api_key", "apikey", "password", "secret", "token"})


class RunEventKind(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    PHASE_STARTED = "phase_started"
    PHASE_FINISHED = "phase_finished"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_ACTION = "tool_action"
    OBSERVATION = "observation"
    VERIFICATION = "verification"
    TRANSITION = "transition"
    CHECKPOINT = "checkpoint"
    RECOVERY = "recovery"
    CANCELLATION = "cancellation"
    LEGACY = "legacy"


def _sanitize_string(value: str, max_chars: int) -> str:
    sanitized = _ASSIGNMENT_SECRET.sub(r"\1\2[REDACTED]", value)
    sanitized = _BEARER_SECRET.sub("Bearer [REDACTED]", sanitized)
    sanitized = _OPENAI_STYLE_SECRET.sub("[REDACTED]", sanitized)
    if len(sanitized) > max_chars:
        return sanitized[:max_chars] + "...[TRUNCATED]"
    return sanitized


def redact_and_bound(value: Any, max_chars: int = 4000) -> Any:
    """Return a sanitized copy suitable for durable event storage."""
    copied = deepcopy(value)
    if isinstance(copied, str):
        return _sanitize_string(copied, max_chars)
    if isinstance(copied, dict):
        return {
            key: (
                "[REDACTED]"
                if str(key).casefold() in _SENSITIVE_KEYS
                else redact_and_bound(item, max_chars=max_chars)
            )
            for key, item in copied.items()
        }
    if isinstance(copied, list):
        return [redact_and_bound(item, max_chars=max_chars) for item in copied]
    if isinstance(copied, tuple):
        return tuple(redact_and_bound(item, max_chars=max_chars) for item in copied)
    return copied


class RunEvent(BaseModel):
    schema_version: int = 1
    run_id: str
    kind: RunEventKind
    timestamp: Any = Field(default_factory=utc_now)
    phase: str | None = None
    correlation_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        kind: RunEventKind,
        phase: str | None = None,
        correlation_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        return cls(
            run_id=run_id,
            kind=kind,
            phase=phase,
            correlation_id=correlation_id,
            data=redact_and_bound(data or {}),
        )
