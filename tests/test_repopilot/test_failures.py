import pytest

from openharness.repopilot.failures import (
    FailureCategory,
    FailurePolicy,
    FailureRecord,
    RecoveryAction,
    classify_failure,
)
from openharness.repopilot.models import Phase


@pytest.mark.parametrize(
    ("stdout", "stderr", "category"),
    [
        ("SyntaxError: invalid syntax", "", FailureCategory.SYNTAX),
        ("ERROR collecting tests/test_x.py", "", FailureCategory.COLLECTION),
        ("FAILED tests/test_x.py::test_x - AssertionError", "", FailureCategory.ASSERTION),
        ("ModuleNotFoundError: No module named 'missing'", "", FailureCategory.DEPENDENCY),
        ("", "PermissionError: access denied", FailureCategory.PERMISSION),
        ("request timed out", "", FailureCategory.TIMEOUT),
    ],
)
def test_classify_failure_has_stable_categories(
    stdout: str, stderr: str, category: FailureCategory
) -> None:
    record = classify_failure(
        stdout=stdout,
        stderr=stderr,
        exit_code=1,
        source_phase=Phase.VERIFY,
    )

    assert record.category is category
    assert record.signature


def test_failure_signature_is_independent_of_volatile_values() -> None:
    left = classify_failure(
        stdout=r"C:\tmp\a.py:12 failed in 0.13s at 0x7FFABC",
        stderr="",
        exit_code=1,
        source_phase=Phase.VERIFY,
    )
    right = classify_failure(
        stdout=r"D:\other\a.py:99 failed in 3.27s at 0x1AAEEE",
        stderr="",
        exit_code=1,
        source_phase=Phase.VERIFY,
    )

    assert left.signature == right.signature


@pytest.mark.parametrize(
    ("phase", "category", "repeated", "expected"),
    [
        (
            Phase.ANALYZE,
            FailureCategory.STRUCTURED_OUTPUT,
            False,
            RecoveryAction.RETRY,
        ),
        (Phase.VERIFY, FailureCategory.ASSERTION, False, RecoveryAction.REPAIR),
        (Phase.VERIFY, FailureCategory.ASSERTION, True, RecoveryAction.REPLAN),
        (
            Phase.EXECUTE,
            FailureCategory.OUT_OF_SCOPE_DIFF,
            False,
            RecoveryAction.STOP,
        ),
        (Phase.EXECUTE, FailureCategory.NO_DIFF, False, RecoveryAction.REPLAN),
        (Phase.REPAIR, FailureCategory.SYNTAX, False, RecoveryAction.REPAIR),
    ],
)
def test_recovery_matrix(
    phase: Phase,
    category: FailureCategory,
    repeated: bool,
    expected: RecoveryAction,
) -> None:
    failure = FailureRecord(
        category=category,
        source_phase=phase,
        signature="same",
        summary=category.value,
    )

    decision = FailurePolicy().decide(failure, repeated=repeated)

    assert decision.action is expected


def test_budget_exhaustion_overrides_recovery() -> None:
    failure = FailureRecord(
        category=FailureCategory.ASSERTION,
        source_phase=Phase.VERIFY,
        signature="same",
        summary="failed",
    )

    decision = FailurePolicy().decide(
        failure,
        repeated=False,
        budget_exhausted_reason="repair_budget_exhausted",
    )

    assert decision.action is RecoveryAction.STOP
    assert decision.terminal_reason == "repair_budget_exhausted"
