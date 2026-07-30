from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PairedEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs: int = Field(ge=1)
    difference: float
    lower: float
    upper: float
    confidence: float
    iterations: int


class PermutationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1)
    exact: bool
    permutations: int = Field(ge=1)


class McNemarResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs: int = Field(ge=1)
    candidate_only: int = Field(ge=0)
    baseline_only: int = Field(ge=0)
    p_value: float = Field(ge=0, le=1)


ClaimClassification = Literal[
    "significant_improvement",
    "significant_regression",
    "directional_improvement",
    "directional_regression",
    "null_or_inconclusive",
]


def _paired_differences(
    candidate: Mapping[str, float | bool],
    baseline: Mapping[str, float | bool],
) -> list[float]:
    candidate_ids = set(candidate)
    baseline_ids = set(baseline)
    if candidate_ids != baseline_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "paired task IDs differ: "
            f"missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    if not candidate_ids:
        raise ValueError("paired statistics require at least one task")
    return [
        float(candidate[task_id]) - float(baseline[task_id])
        for task_id in sorted(candidate_ids)
    ]


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    iterations: int = 10_000,
    seed: int = 20260730,
    confidence: float = 0.95,
) -> PairedEstimate:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    differences = _paired_differences(candidate, baseline)
    rng = random.Random(seed)
    bootstrap_means = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(iterations)
    ]
    tail = (1 - confidence) / 2
    return PairedEstimate(
        pairs=len(differences),
        difference=sum(differences) / len(differences),
        lower=_quantile(bootstrap_means, tail),
        upper=_quantile(bootstrap_means, 1 - tail),
        confidence=confidence,
        iterations=iterations,
    )


def paired_permutation_test(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    iterations: int = 100_000,
    seed: int = 20260730,
    exact_limit: int = 20,
) -> PermutationResult:
    differences = _paired_differences(candidate, baseline)
    observed = abs(sum(differences) / len(differences))
    tolerance = 1e-12
    if len(differences) <= exact_limit:
        sign_vectors = itertools.product((-1, 1), repeat=len(differences))
        extreme = 0
        permutations = 0
        for signs in sign_vectors:
            permutations += 1
            permuted = abs(
                sum(sign * value for sign, value in zip(signs, differences, strict=True))
                / len(differences)
            )
            if permuted + tolerance >= observed:
                extreme += 1
        p_value = extreme / permutations
        exact = True
    else:
        if iterations < 1:
            raise ValueError("permutation iterations must be positive")
        rng = random.Random(seed)
        extreme = 0
        for _ in range(iterations):
            permuted = abs(
                sum(rng.choice((-1, 1)) * value for value in differences)
                / len(differences)
            )
            if permuted + tolerance >= observed:
                extreme += 1
        permutations = iterations
        p_value = (extreme + 1) / (iterations + 1)
        exact = False
    return PermutationResult(
        pairs=len(differences),
        p_value=p_value,
        exact=exact,
        permutations=permutations,
    )


def _binomial_cdf_half(successes: int, trials: int) -> float:
    denominator = 2**trials
    return sum(math.comb(trials, value) for value in range(successes + 1)) / denominator


def mcnemar_exact(
    candidate: Mapping[str, bool],
    baseline: Mapping[str, bool],
) -> McNemarResult:
    _paired_differences(candidate, baseline)
    candidate_only = sum(
        bool(candidate[task_id]) and not bool(baseline[task_id])
        for task_id in candidate
    )
    baseline_only = sum(
        bool(baseline[task_id]) and not bool(candidate[task_id])
        for task_id in candidate
    )
    discordant = candidate_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2 * _binomial_cdf_half(min(candidate_only, baseline_only), discordant),
        )
    return McNemarResult(
        pairs=len(candidate),
        candidate_only=candidate_only,
        baseline_only=baseline_only,
        p_value=p_value,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    for value in p_values:
        if not 0 <= value <= 1:
            raise ValueError("p-values must be between zero and one")
    count = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (original_index, value) in enumerate(indexed):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def claim_classification(
    *,
    adjusted_p: float,
    lower: float,
    upper: float,
    effect: float,
    alpha: float = 0.05,
) -> ClaimClassification:
    if adjusted_p < alpha and lower > 0 and effect > 0:
        return "significant_improvement"
    if adjusted_p < alpha and upper < 0 and effect < 0:
        return "significant_regression"
    if effect > 0:
        return "directional_improvement"
    if effect < 0:
        return "directional_regression"
    return "null_or_inconclusive"

