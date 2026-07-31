"""Aggregate machine-readable reports for localization checkpoints."""

from __future__ import annotations

from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field

from .localization_execution import (
    LocalizationCheckpoint,
    LocalizationRecord,
    LocalizationRunConfig,
)
from .models import SampleManifest


class LocalizationAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: int = Field(ge=1)
    recall_at: dict[int, float]
    hit_at: dict[int, float]
    mrr: float
    irrelevant_context_rate: float
    mean_index_seconds: float
    mean_retrieval_seconds: float
    mean_estimated_context_tokens: float


class LocalizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str
    configuration: LocalizationRunConfig
    overall: LocalizationAggregate
    by_difficulty: dict[str, LocalizationAggregate]
    completed_instance_ids: tuple[str, ...]


def _mean(values) -> float:
    return round(fmean(values), 6)


def _aggregate(records: list[LocalizationRecord]) -> LocalizationAggregate:
    if not records:
        raise ValueError("cannot aggregate an empty localization record set")
    cutoffs = sorted(records[0].metrics.recall_at)
    return LocalizationAggregate(
        tasks=len(records),
        recall_at={
            cutoff: _mean(record.metrics.recall_at[cutoff] for record in records)
            for cutoff in cutoffs
        },
        hit_at={
            cutoff: _mean(
                1.0 if record.metrics.hit_at[cutoff] else 0.0
                for record in records
            )
            for cutoff in cutoffs
        },
        mrr=_mean(record.metrics.mrr for record in records),
        irrelevant_context_rate=_mean(
            record.metrics.irrelevant_context_rate for record in records
        ),
        mean_index_seconds=_mean(record.index_seconds for record in records),
        mean_retrieval_seconds=_mean(record.retrieval_seconds for record in records),
        mean_estimated_context_tokens=_mean(
            record.metrics.estimated_context_tokens for record in records
        ),
    )


def build_localization_report(
    checkpoint: LocalizationCheckpoint,
    manifest: SampleManifest,
) -> LocalizationReport:
    """Summarize completed records overall and by frozen difficulty stratum."""
    if checkpoint.configuration is None:
        raise ValueError("checkpoint does not record a localization configuration")
    difficulty_by_id = {
        instance.instance_id: instance.difficulty.value for instance in manifest.instances
    }
    unknown = sorted(set(checkpoint.records) - set(difficulty_by_id))
    if unknown:
        raise ValueError(f"checkpoint contains instances outside manifest: {unknown}")
    records = [
        record
        for record in checkpoint.records.values()
        if record.status == "completed"
    ]
    if not records:
        raise ValueError("checkpoint has no completed localization records")
    grouped: dict[str, list[LocalizationRecord]] = {}
    for record in records:
        grouped.setdefault(difficulty_by_id[record.instance_id], []).append(record)
    return LocalizationReport(
        manifest_sha256=manifest.sha256,
        configuration=checkpoint.configuration,
        overall=_aggregate(records),
        by_difficulty={
            difficulty: _aggregate(group)
            for difficulty, group in sorted(grouped.items())
        },
        completed_instance_ids=tuple(sorted(record.instance_id for record in records)),
    )
