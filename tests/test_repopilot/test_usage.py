from datetime import UTC, datetime, timedelta

import pytest

from openharness.repopilot.models import Phase
from openharness.repopilot.usage import (
    ProviderPrice,
    RunSummary,
    TokenUsage,
    estimate_cost,
)


def test_estimate_cost_separates_cache_input_and_output() -> None:
    usage = TokenUsage(
        input_tokens=800_000,
        output_tokens=200_000,
        cache_hit_tokens=300_000,
    )
    price = ProviderPrice(
        provider="deepseek",
        model="deepseek-v4-flash",
        currency="CNY",
        input_per_million=1.0,
        cache_hit_input_per_million=0.02,
        output_per_million=2.0,
        version="2026-07-30",
    )

    result = estimate_cost(usage, price)

    assert usage.total_tokens == 1_000_000
    assert result.amount == pytest.approx(0.906)
    assert result.is_estimate is True


def test_token_usage_aggregates_partial_provider_results() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=3)

    combined = usage + TokenUsage(total_tokens=7)

    assert combined.input_tokens == 10
    assert combined.output_tokens == 3
    assert combined.total_tokens == 20


def test_run_summary_derives_duration() -> None:
    started = datetime(2026, 7, 30, tzinfo=UTC)
    summary = RunSummary(
        run_id="r1",
        workflow="repair",
        workflow_version="2",
        phase=Phase.COMPLETE,
        started_at=started,
        completed_at=started + timedelta(seconds=2.5),
    )

    assert summary.duration_seconds == 2.5
