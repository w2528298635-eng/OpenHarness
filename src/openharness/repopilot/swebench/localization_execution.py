"""Post-inference-only, resumable retrieval-localization evaluation."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..context import ContextBuilder
from ..retrieval import RepositoryIndex
from .gold import extract_gold_files, extract_gold_labels
from .localization import LocalizationMetrics, RetrievedLocation, score_localization
from .models import PublicInstance


class LocalizationRecord(BaseModel):
    """A persisted post-hoc metric record; it never feeds the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    status: str
    index_seconds: float = Field(ge=0)
    retrieval_seconds: float = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    metrics: LocalizationMetrics


class LocalizationCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    records: dict[str, LocalizationRecord] = {}


class LocalizationCheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> LocalizationCheckpoint:
        if not self.path.exists():
            return LocalizationCheckpoint()
        return LocalizationCheckpoint.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )

    def save(self, checkpoint: LocalizationCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=self.path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(checkpoint.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


def _base_sources(repository: Path, patch: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in extract_gold_files(patch):
        candidate = repository / relative
        try:
            result[relative] = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return result


def evaluate_localization_instance(
    *,
    instance: PublicInstance,
    repository: Path,
    gold_patch: str,
    store: LocalizationCheckpointStore,
    char_budget: int = 12_000,
    top_k: int = 12,
) -> LocalizationRecord:
    """Score a task after retrieval output is generated and checkpoint it atomically."""
    checkpoint = store.load()
    existing = checkpoint.records.get(instance.instance_id)
    if existing is not None and existing.status == "completed":
        return existing

    from time import perf_counter

    index_start = perf_counter()
    index = RepositoryIndex.build(repository)
    index_seconds = perf_counter() - index_start
    retrieval_start = perf_counter()
    selection = ContextBuilder(char_budget=char_budget, top_k=top_k).build(
        index=index,
        query=instance.problem_statement,
    )
    retrieval_seconds = perf_counter() - retrieval_start
    labels = extract_gold_labels(gold_patch, base_sources=_base_sources(repository, gold_patch))
    ranking = [
        RetrievedLocation(
            file=chunk.chunk.path,
            symbol=chunk.chunk.symbol or None,
            rank=rank,
            characters=len(chunk.chunk.text),
        )
        for rank, chunk in enumerate(selection.selected_chunks, start=1)
    ]
    record = LocalizationRecord(
        instance_id=instance.instance_id,
        status="completed",
        index_seconds=index_seconds,
        retrieval_seconds=retrieval_seconds,
        indexed_chunks=len(index.chunks),
        metrics=score_localization(labels, ranking),
    )
    store.save(
        LocalizationCheckpoint(
            records={**checkpoint.records, instance.instance_id: record}
        )
    )
    return record


def evaluate_localization_manifest(
    *,
    instances: Iterable[PublicInstance],
    public_rows_with_gold: Iterable[Mapping[str, object]],
    repository_root: Path,
    store: LocalizationCheckpointStore,
    char_budget: int = 12_000,
    top_k: int = 12,
) -> LocalizationCheckpoint:
    """Evaluate a frozen public manifest; gold patches are consumed only here."""
    patches: dict[str, str] = {}
    for row in public_rows_with_gold:
        instance_id = row.get("instance_id")
        patch = row.get("patch")
        if isinstance(instance_id, str) and isinstance(patch, str):
            patches[instance_id] = patch
    for instance in instances:
        if instance.instance_id not in patches:
            raise ValueError(f"gold patch missing for {instance.instance_id}")
        evaluate_localization_instance(
            instance=instance,
            repository=repository_root / f"formal-{instance.instance_id}",
            gold_patch=patches[instance.instance_id],
            store=store,
            char_budget=char_budget,
            top_k=top_k,
        )
    return store.load()
