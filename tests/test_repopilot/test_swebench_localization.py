from __future__ import annotations

import math

from openharness.repopilot.swebench.gold import GoldLabels
from openharness.repopilot.swebench.localization import (
    RetrievedLocation,
    score_localization,
)


def test_file_metrics_deduplicate_chunks_by_best_file_rank() -> None:
    labels = GoldLabels(files=("a.py", "b.py"))
    ranking = [
        RetrievedLocation(file="x.py", rank=1, characters=100),
        RetrievedLocation(file="a.py", rank=2, characters=100),
        RetrievedLocation(file="a.py", rank=3, characters=100),
        RetrievedLocation(file="b.py", rank=4, characters=100),
    ]

    result = score_localization(labels, ranking, ks=(1, 2, 3))

    assert result.recall_at == {1: 0.0, 2: 0.5, 3: 1.0}
    assert result.hit_at == {1: False, 2: True, 3: True}
    assert result.precision_at == {1: 0.0, 2: 0.5, 3: 2 / 3}
    assert result.mrr == 0.5
    expected_dcg = (1 / math.log2(3)) + (1 / math.log2(4))
    expected_ideal = 1 + (1 / math.log2(3))
    assert result.ndcg_at[3] == expected_dcg / expected_ideal
    assert result.first_relevant_file_rank == 2


def test_context_efficiency_uses_original_chunks_and_explicit_token_estimate() -> None:
    labels = GoldLabels(files=("a.py",))
    ranking = [
        RetrievedLocation(file="x.py", rank=1, characters=300),
        RetrievedLocation(file="a.py", rank=2, characters=100),
    ]

    result = score_localization(labels, ranking, ks=(1, 2))

    assert result.context_characters == 400
    assert result.estimated_context_tokens == 100
    assert result.irrelevant_context_rate == 0.75
    assert result.relevant_file_hits_per_1000_tokens == 10


def test_symbol_metrics_report_their_own_denominator() -> None:
    labels = GoldLabels(
        files=("src/a.py", "src/b.py"),
        symbols={
            "src/a.py": ("Parser.parse",),
            "src/b.py": ("normalize",),
        },
    )
    ranking = [
        RetrievedLocation(
            file="src/a.py",
            symbol="Parser.other",
            rank=1,
            characters=100,
        ),
        RetrievedLocation(
            file="src/a.py",
            symbol="Parser.parse",
            rank=2,
            characters=100,
        ),
        RetrievedLocation(
            file="src/b.py",
            symbol="normalize",
            rank=3,
            characters=100,
        ),
    ]

    result = score_localization(labels, ranking, ks=(1, 2, 3))

    assert result.symbol_denominator == 2
    assert result.symbol_recall_at == {1: 0.0, 2: 0.5, 3: 1.0}
    assert result.first_relevant_symbol_rank == 2


def test_symbol_metrics_are_not_reported_without_eligible_gold_symbols() -> None:
    labels = GoldLabels(files=("src/config.py",))
    ranking = [
        RetrievedLocation(file="src/config.py", rank=1, characters=100),
    ]

    result = score_localization(labels, ranking, ks=(1,))

    assert result.symbol_denominator == 0
    assert result.symbol_recall_at is None
    assert result.first_relevant_symbol_rank is None


def test_empty_retrieval_has_zero_scores_without_division_errors() -> None:
    result = score_localization(
        GoldLabels(files=("src/a.py",)),
        [],
        ks=(1, 3),
    )

    assert result.recall_at == {1: 0.0, 3: 0.0}
    assert result.precision_at == {1: 0.0, 3: 0.0}
    assert result.mrr == 0.0
    assert result.context_characters == 0
    assert result.irrelevant_context_rate == 0.0
    assert result.relevant_file_hits_per_1000_tokens == 0.0

