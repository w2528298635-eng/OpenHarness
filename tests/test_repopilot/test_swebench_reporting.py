from __future__ import annotations

from openharness.repopilot.swebench.adapters import EvaluationArm
from openharness.repopilot.swebench.models import DifficultyStratum
from openharness.repopilot.swebench.reporting import (
    EvaluationObservation,
    build_experiment_report,
    render_report_markdown,
)


def _observation(
    task: str,
    arm: EvaluationArm,
    *,
    resolved: bool,
    repetition: int = 1,
) -> EvaluationObservation:
    return EvaluationObservation(
        instance_id=task,
        difficulty=DifficultyStratum.EASY,
        arm=arm,
        repetition=repetition,
        attempted=True,
        completed=True,
        patch_applied=resolved,
        resolved=resolved,
        scope_compliant=True,
        total_tokens=100,
        duration_seconds=2,
    )


def test_report_uses_explicit_denominators_and_difficulty_breakdowns() -> None:
    observations = [
        _observation("a", EvaluationArm.NATIVE, resolved=False),
        _observation("b", EvaluationArm.NATIVE, resolved=True),
        _observation("a", EvaluationArm.UPGRADED_NO_RETRIEVAL, resolved=True),
        _observation("b", EvaluationArm.UPGRADED_NO_RETRIEVAL, resolved=True),
    ]

    report = build_experiment_report(observations, bootstrap_iterations=200)

    native = report.arms[EvaluationArm.NATIVE]
    assert native.attempted == 2
    assert native.resolved == 1
    assert native.resolution_rate == 0.5
    assert native.by_difficulty[DifficultyStratum.EASY].attempted == 2


def test_report_does_not_claim_significance_when_interval_crosses_zero() -> None:
    observations = [
        _observation("a", EvaluationArm.NATIVE, resolved=False),
        _observation("b", EvaluationArm.NATIVE, resolved=True),
        _observation("a", EvaluationArm.UPGRADED_NO_RETRIEVAL, resolved=True),
        _observation("b", EvaluationArm.UPGRADED_NO_RETRIEVAL, resolved=False),
    ]

    report = build_experiment_report(observations, bootstrap_iterations=500)
    comparison = next(
        item for item in report.comparisons if item.name == "upgraded_vs_native"
    )

    assert comparison.claim == "null_or_inconclusive"
    markdown = render_report_markdown(report)
    assert "null_or_inconclusive" in markdown
    assert "2 attempted" in markdown

