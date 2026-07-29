from datetime import UTC, datetime
from pathlib import Path

from openharness.repopilot.evaluation import (
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationStrategy,
    aggregate_evaluation,
    write_evaluation_report,
)


def _result(
    case: str,
    strategy: EvaluationStrategy,
    *,
    verified: bool,
    duration: float,
    tokens: int = 0,
    failure: str | None = None,
) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        case_id=case,
        strategy=strategy,
        repetition=1,
        run_id=f"{strategy.value}-{case}",
        verified=verified,
        duration_seconds=duration,
        tokens=tokens,
        estimated_cost=0.01 if tokens else None,
        repair_attempts=1,
        replan_attempts=0,
        changed_file_compliant=verified,
        failure=failure,
    )


def test_aggregate_keeps_scripted_and_model_quality_separate() -> None:
    results = [
        _result("a", EvaluationStrategy.SCRIPTED, verified=True, duration=1),
        _result(
            "a",
            EvaluationStrategy.MODEL_NO_RETRIEVAL,
            verified=False,
            duration=3,
            tokens=100,
            failure="verification_failed",
        ),
        _result(
            "b",
            EvaluationStrategy.MODEL_NO_RETRIEVAL,
            verified=True,
            duration=5,
            tokens=300,
        ),
    ]

    aggregates = aggregate_evaluation(results)

    assert aggregates[EvaluationStrategy.SCRIPTED].success_rate == 1
    model = aggregates[EvaluationStrategy.MODEL_NO_RETRIEVAL]
    assert model.success_rate == 0.5
    assert model.median_duration_seconds == 4
    assert model.total_tokens == 400
    assert model.failure_distribution == {"verification_failed": 1}


def test_report_writers_preserve_every_run(tmp_path: Path) -> None:
    result = _result("a", EvaluationStrategy.SCRIPTED, verified=True, duration=1)
    report = EvaluationReport(
        name="repair-suite",
        generated_at=datetime.now(UTC),
        results=[result],
        aggregates=aggregate_evaluation([result]),
    )

    json_path, markdown_path = write_evaluation_report(report, tmp_path)

    assert result.run_id in json_path.read_text(encoding="utf-8")
    assert "scripted" in markdown_path.read_text(encoding="utf-8")
