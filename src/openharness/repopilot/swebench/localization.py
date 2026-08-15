from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .gold import GoldLabels


class RetrievedLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(min_length=1)
    rank: int = Field(ge=1)
    symbol: str | None = None
    characters: int = Field(default=0, ge=0)

    @field_validator("file")
    @classmethod
    def normalize_file(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized


class LocalizationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_file_denominator: int
    symbol_denominator: int
    recall_at: dict[int, float]
    hit_at: dict[int, bool]
    precision_at: dict[int, float]
    ndcg_at: dict[int, float]
    symbol_recall_at: dict[int, float] | None
    mrr: float
    first_relevant_file_rank: int | None
    first_relevant_symbol_rank: int | None
    context_characters: int
    estimated_context_tokens: int
    irrelevant_context_rate: float
    relevant_file_hits_per_1000_tokens: float


def _ndcg(relevance: list[int], *, gold_count: int, k: int) -> float:
    observed = sum(
        value / math.log2(index + 2)
        for index, value in enumerate(relevance[:k])
    )
    ideal_relevant = min(gold_count, k)
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_relevant))
    return observed / ideal if ideal else 0.0


def score_localization(
    labels: GoldLabels,
    ranking: list[RetrievedLocation],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> LocalizationMetrics:
    if not labels.files:
        raise ValueError("localization metrics require at least one gold file")
    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("localization cutoffs must be positive")

    ordered = sorted(ranking, key=lambda item: item.rank)
    unique_files: list[str] = []
    for item in ordered:
        if item.file not in unique_files:
            unique_files.append(item.file)

    gold_files = set(labels.files)
    file_relevance = [int(path in gold_files) for path in unique_files]
    recall_at: dict[int, float] = {}
    hit_at: dict[int, bool] = {}
    precision_at: dict[int, float] = {}
    ndcg_at: dict[int, float] = {}
    for k in normalized_ks:
        relevant = sum(file_relevance[:k])
        recall_at[k] = relevant / len(gold_files)
        hit_at[k] = relevant > 0
        precision_at[k] = relevant / k
        ndcg_at[k] = _ndcg(file_relevance, gold_count=len(gold_files), k=k)

    first_file = next(
        (index for index, relevant in enumerate(file_relevance, start=1) if relevant),
        None,
    )
    mrr = 1 / first_file if first_file is not None else 0.0

    gold_symbols = {
        (path, symbol)
        for path, names in labels.symbols.items()
        for symbol in names
    }
    symbol_relevance: list[int] = []
    seen_symbols: set[tuple[str, str]] = set()
    for item in ordered:
        if item.symbol is None:
            continue
        key = (item.file, item.symbol)
        if key in seen_symbols:
            continue
        seen_symbols.add(key)
        symbol_relevance.append(int(key in gold_symbols))
    if gold_symbols:
        symbol_recall_at = {
            k: sum(symbol_relevance[:k]) / len(gold_symbols) for k in normalized_ks
        }
        first_symbol = next(
            (
                index
                for index, relevant in enumerate(symbol_relevance, start=1)
                if relevant
            ),
            None,
        )
    else:
        symbol_recall_at = None
        first_symbol = None

    context_characters = sum(item.characters for item in ordered)
    irrelevant_characters = sum(
        item.characters for item in ordered if item.file not in gold_files
    )
    estimated_tokens = math.ceil(context_characters / 4)
    relevant_file_hits = len(gold_files.intersection(unique_files))
    return LocalizationMetrics(
        gold_file_denominator=len(gold_files),
        symbol_denominator=len(gold_symbols),
        recall_at=recall_at,
        hit_at=hit_at,
        precision_at=precision_at,
        ndcg_at=ndcg_at,
        symbol_recall_at=symbol_recall_at,
        mrr=mrr,
        first_relevant_file_rank=first_file,
        first_relevant_symbol_rank=first_symbol,
        context_characters=context_characters,
        estimated_context_tokens=estimated_tokens,
        irrelevant_context_rate=(
            irrelevant_characters / context_characters if context_characters else 0.0
        ),
        relevant_file_hits_per_1000_tokens=(
            relevant_file_hits * 1000 / estimated_tokens if estimated_tokens else 0.0
        ),
    )

