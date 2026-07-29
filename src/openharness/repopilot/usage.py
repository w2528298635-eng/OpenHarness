from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .models import Phase, TokenUsage

__all__ = [
    "CostEstimate",
    "ProviderPrice",
    "RunSummary",
    "TokenUsage",
    "estimate_cost",
]


class ProviderPrice(BaseModel):
    provider: str
    model: str
    currency: str
    input_per_million: float = Field(ge=0)
    cache_hit_input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    version: str


class CostEstimate(BaseModel):
    provider: str
    model: str
    currency: str
    amount: float = Field(ge=0)
    price_version: str
    is_estimate: bool = True


def estimate_cost(usage: TokenUsage, price: ProviderPrice) -> CostEstimate:
    cache_tokens = min(usage.cache_hit_tokens, usage.input_tokens)
    regular_input_tokens = usage.input_tokens - cache_tokens
    million = Decimal(1_000_000)
    amount = (
        Decimal(regular_input_tokens) * Decimal(str(price.input_per_million))
        + Decimal(cache_tokens) * Decimal(str(price.cache_hit_input_per_million))
        + Decimal(usage.output_tokens) * Decimal(str(price.output_per_million))
    ) / million
    return CostEstimate(
        provider=price.provider,
        model=price.model,
        currency=price.currency,
        amount=float(amount),
        price_version=price.version,
    )


class RunSummary(BaseModel):
    schema_version: int = 1
    run_id: str
    workflow: str
    workflow_version: str
    phase: Phase
    terminal_reason: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    model_calls: int = 0
    phase_calls: int = 0
    tool_calls: int = 0
    verification_checks: int = 0
    repair_attempts: int = 0
    replan_attempts: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_estimate: CostEstimate | None = None
    changed_files: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def derive_duration(self) -> RunSummary:
        if (
            self.duration_seconds is None
            and self.completed_at is not None
            and self.started_at is not None
        ):
            self.duration_seconds = max(0.0, (self.completed_at - self.started_at).total_seconds())
        return self
