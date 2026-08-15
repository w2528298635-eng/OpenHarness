from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from .adapters import EvaluationArm
from .models import DifficultyStratum
from .statistics import (
    PairedEstimate,
    claim_classification,
    holm_adjust,
    paired_bootstrap,
    paired_permutation_test,
)


class EvaluationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    difficulty: DifficultyStratum
    arm: EvaluationArm
    repetition: int = Field(ge=1)
    attempted: bool
    completed: bool
    patch_applied: bool
    resolved: bool
    scope_compliant: bool
    total_tokens: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)


class OutcomeAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempted: int
    completed: int
    patch_applied: int
    resolved: int
    resolution_rate: float
    scope_compliance_rate: float
    total_tokens: int
    median_duration_seconds: float


class ArmAggregate(OutcomeAggregate):
    by_difficulty: dict[DifficultyStratum, OutcomeAggregate]


class PrimaryComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    candidate: EvaluationArm
    baseline: EvaluationArm
    estimate: PairedEstimate
    raw_p: float
    adjusted_p: float
    claim: str


class ExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observations: int
    arms: dict[EvaluationArm, ArmAggregate]
    comparisons: tuple[PrimaryComparison, ...]


def _aggregate(items: list[EvaluationObservation]) -> OutcomeAggregate:
    attempted = sum(item.attempted for item in items)
    completed = sum(item.completed for item in items)
    patch_applied = sum(item.patch_applied for item in items)
    resolved = sum(item.resolved for item in items)
    scope_compliant = sum(item.scope_compliant for item in items)
    return OutcomeAggregate(
        attempted=attempted,
        completed=completed,
        patch_applied=patch_applied,
        resolved=resolved,
        resolution_rate=resolved / attempted if attempted else 0.0,
        scope_compliance_rate=scope_compliant / attempted if attempted else 0.0,
        total_tokens=sum(item.total_tokens for item in items),
        median_duration_seconds=(
            statistics.median(item.duration_seconds for item in items)
            if items
            else 0.0
        ),
    )


def _task_success(
    observations: list[EvaluationObservation],
    arm: EvaluationArm,
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for item in observations:
        if item.arm is arm and item.attempted:
            grouped[item.instance_id].append(item.resolved)
    return {
        instance_id: sum(values) / len(values)
        for instance_id, values in grouped.items()
    }


def build_experiment_report(
    observations: list[EvaluationObservation],
    *,
    bootstrap_iterations: int = 10_000,
) -> ExperimentReport:
    by_arm: dict[EvaluationArm, list[EvaluationObservation]] = defaultdict(list)
    for observation in observations:
        by_arm[observation.arm].append(observation)

    arms: dict[EvaluationArm, ArmAggregate] = {}
    for arm, items in by_arm.items():
        overall = _aggregate(items)
        by_difficulty = {
            difficulty: _aggregate(
                [item for item in items if item.difficulty is difficulty]
            )
            for difficulty in DifficultyStratum
            if any(item.difficulty is difficulty for item in items)
        }
        arms[arm] = ArmAggregate(
            **overall.model_dump(),
            by_difficulty=by_difficulty,
        )

    declared = (
        (
            "upgraded_vs_native",
            EvaluationArm.UPGRADED_NO_RETRIEVAL,
            EvaluationArm.NATIVE,
        ),
        (
            "upgraded_vs_legacy",
            EvaluationArm.UPGRADED_NO_RETRIEVAL,
            EvaluationArm.LEGACY,
        ),
        (
            "rag_vs_no_rag",
            EvaluationArm.UPGRADED_WITH_RETRIEVAL,
            EvaluationArm.UPGRADED_NO_RETRIEVAL,
        ),
    )
    provisional: list[tuple[str, EvaluationArm, EvaluationArm, PairedEstimate, float]] = []
    for name, candidate_arm, baseline_arm in declared:
        candidate = _task_success(observations, candidate_arm)
        baseline = _task_success(observations, baseline_arm)
        if not candidate or not baseline or set(candidate) != set(baseline):
            continue
        estimate = paired_bootstrap(
            candidate,
            baseline,
            iterations=bootstrap_iterations,
        )
        permutation = paired_permutation_test(candidate, baseline)
        provisional.append(
            (name, candidate_arm, baseline_arm, estimate, permutation.p_value)
        )
    adjusted = holm_adjust([item[4] for item in provisional])
    comparisons = tuple(
        PrimaryComparison(
            name=name,
            candidate=candidate_arm,
            baseline=baseline_arm,
            estimate=estimate,
            raw_p=raw_p,
            adjusted_p=adjusted_p,
            claim=claim_classification(
                adjusted_p=adjusted_p,
                lower=estimate.lower,
                upper=estimate.upper,
                effect=estimate.difference,
            ),
        )
        for (name, candidate_arm, baseline_arm, estimate, raw_p), adjusted_p in zip(
            provisional,
            adjusted,
            strict=True,
        )
    )
    return ExperimentReport(
        observations=len(observations),
        arms=arms,
        comparisons=comparisons,
    )


def render_report_markdown(report: ExperimentReport) -> str:
    lines = [
        "# RepoPilot SWE-bench report",
        "",
        "## Outcomes",
        "",
        "| arm | denominator | resolved | rate | tokens | median seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, aggregate in report.arms.items():
        lines.append(
            f"| {arm.value} | {aggregate.attempted} attempted | "
            f"{aggregate.resolved} | {aggregate.resolution_rate:.1%} | "
            f"{aggregate.total_tokens} | {aggregate.median_duration_seconds:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered paired comparisons",
            "",
            "| comparison | difference | 95% CI | adjusted p | claim |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for comparison in report.comparisons:
        lines.append(
            f"| {comparison.name} | {comparison.estimate.difference:.3f} | "
            f"[{comparison.estimate.lower:.3f}, {comparison.estimate.upper:.3f}] | "
            f"{comparison.adjusted_p:.4f} | {comparison.claim} |"
        )
    return "\n".join(lines) + "\n"

