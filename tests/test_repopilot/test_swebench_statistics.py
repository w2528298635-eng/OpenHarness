from __future__ import annotations

import pytest

from openharness.repopilot.swebench.statistics import (
    claim_classification,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
    paired_permutation_test,
)


def test_paired_bootstrap_is_seeded_and_uses_task_level_differences() -> None:
    baseline = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 0.0}
    candidate = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.5}

    first = paired_bootstrap(candidate, baseline, iterations=2000, seed=17)
    second = paired_bootstrap(candidate, baseline, iterations=2000, seed=17)

    assert first == second
    assert first.pairs == 4
    assert first.difference == 0.5
    assert first.lower <= first.difference <= first.upper


def test_paired_statistics_reject_mismatched_task_sets() -> None:
    with pytest.raises(ValueError, match="paired task IDs differ"):
        paired_bootstrap({"a": 1.0}, {"b": 0.0}, iterations=10)

    with pytest.raises(ValueError, match="paired task IDs differ"):
        paired_permutation_test({"a": 1.0}, {"b": 0.0})


def test_paired_permutation_all_zero_differences_has_unit_p_value() -> None:
    values = {"a": 0.0, "b": 0.5, "c": 1.0}

    result = paired_permutation_test(values, values)

    assert result.p_value == 1.0
    assert result.exact is True


def test_paired_permutation_detects_consistent_improvement() -> None:
    baseline = {f"task-{index}": 0.0 for index in range(10)}
    candidate = {f"task-{index}": 1.0 for index in range(10)}

    result = paired_permutation_test(candidate, baseline)

    assert result.p_value == pytest.approx(2 / (2**10))
    assert result.exact is True


def test_mcnemar_exact_uses_only_discordant_pairs() -> None:
    baseline = {
        **{f"gain-{index}": False for index in range(10)},
        "both-pass": True,
        "both-fail": False,
    }
    candidate = {
        **{f"gain-{index}": True for index in range(10)},
        "both-pass": True,
        "both-fail": False,
    }

    result = mcnemar_exact(candidate, baseline)

    assert result.candidate_only == 10
    assert result.baseline_only == 0
    assert result.p_value == pytest.approx(2 / (2**10))


def test_holm_adjust_is_monotone_in_sorted_p_value_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize(
    ("adjusted_p", "lower", "upper", "effect", "expected"),
    [
        (0.01, 0.02, 0.20, 0.10, "significant_improvement"),
        (0.01, -0.20, -0.02, -0.10, "significant_regression"),
        (0.01, -0.01, 0.20, 0.10, "directional_improvement"),
        (0.20, 0.01, 0.20, 0.10, "directional_improvement"),
        (0.20, -0.20, -0.01, -0.10, "directional_regression"),
        (0.50, -0.10, 0.10, 0.0, "null_or_inconclusive"),
    ],
)
def test_claim_classification_requires_p_value_and_interval(
    adjusted_p: float,
    lower: float,
    upper: float,
    effect: float,
    expected: str,
) -> None:
    assert (
        claim_classification(
            adjusted_p=adjusted_p,
            lower=lower,
            upper=upper,
            effect=effect,
        )
        == expected
    )

