from pathlib import Path

from openharness.repopilot.context import ContextBuilder
from openharness.repopilot.retrieval import RepositoryIndex


def test_context_builder_prioritizes_failure_and_suspected_file_within_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "pricing.py").write_text(
        "def final_price(total, discount_rate):\n"
        "    if discount_rate >= 1:\n"
        "        raise ValueError('discount')\n"
        "    return total * (1 - discount_rate)\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text(
        "def send_email():\n    return True\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)

    selection = ContextBuilder(char_budget=500).build(
        index=index,
        query="full discount should return zero",
        failure_text="FAILED test_full_discount: ValueError discount",
        priority_paths=["pricing.py"],
    )

    assert "FAILED test_full_discount" in selection.rendered
    assert "pricing.py" in selection.rendered
    assert selection.total_chars <= 500
    assert selection.selected_chunks
    assert all(item.reason for item in selection.selected_chunks)


def test_context_builder_deduplicates_chunks(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def parse_value(value):\n    return value.strip()\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)

    selection = ContextBuilder(char_budget=1000).build(
        index=index,
        query="parse value",
        priority_paths=["app.py", "app.py"],
    )

    ids = [item.chunk.chunk_id for item in selection.selected_chunks]
    assert len(ids) == len(set(ids))


def test_context_builder_uses_hybrid_retrieval_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from openharness.repopilot import context as context_module

    (tmp_path / "mask.py").write_text("def combine_masks(a, b):\n return a | b\n", encoding="utf-8")
    monkeypatch.setenv("REPOPILOT_HYBRID_RETRIEVAL", "1")

    class FakeEncoder:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(context_module, "LocalEmbeddingEncoder", FakeEncoder)
    selection = ContextBuilder(char_budget=500).build(
        index=RepositoryIndex.build(tmp_path), query="mask failure"
    )

    assert "semantic" in selection.selected_chunks[0].reason


def test_context_builder_includes_structural_neighbors(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(
        "def helper(value):\n return value\n\ndef target(value):\n return helper(value)\n",
        encoding="utf-8",
    )
    selection = ContextBuilder(
        char_budget=2000,
        top_k=1,
        structural_expansion=True,
    ).build(
        index=RepositoryIndex.build(tmp_path), query="target"
    )

    assert {item.chunk.symbol for item in selection.selected_chunks} >= {"target", "helper"}
    assert any("structure" in item.reason for item in selection.selected_chunks)


def test_context_builder_reranks_only_fused_candidates_before_top_k(
    tmp_path: Path, monkeypatch
) -> None:
    from openharness.repopilot import context as context_module

    (tmp_path / "first.py").write_text(
        "def generic_mask():\n return 'mask mask mask'\n", encoding="utf-8"
    )
    (tmp_path / "target.py").write_text(
        "def combine_masks(left, right):\n return left | right\n", encoding="utf-8"
    )
    captured = {}

    class FakeEncoder:
        def __init__(self, **kwargs):
            captured["embedding_config"] = kwargs

        def rank_many(self, _queries, chunks, *, top_k):
            assert top_k == 100
            return [(index, 0.9 - index * 0.1) for index, _chunk in enumerate(chunks)]

    class FakeReranker:
        def __init__(self, **kwargs):
            captured["reranker_config"] = kwargs

        def rank(self, query, candidates, *, top_k):
            captured["query"] = query
            captured["candidate_ids"] = [chunk.chunk_id for chunk in candidates]
            captured["top_k"] = top_k
            return [(1, 8.5), (0, 1.0)]

    monkeypatch.setattr(context_module, "LocalEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr(context_module, "LocalCrossEncoderReranker", FakeReranker)
    selection = ContextBuilder(
        char_budget=500,
        top_k=1,
        retrieval_strategy="hybrid",
        embedding_model="custom/code-encoder",
        embedding_revision="embedding-revision",
        embedding_max_seq_length=768,
        reranker="cross_encoder",
        reranker_model="custom/code-reranker",
        reranker_revision="reranker-revision",
        reranker_max_length=1024,
        reranker_candidate_k=40,
        reranker_weight=0.75,
    ).build(
        index=RepositoryIndex.build(tmp_path),
        query="mask propagation fails for None",
    )

    assert len(captured["candidate_ids"]) == 2
    assert captured["embedding_config"] == {
        "model": "custom/code-encoder",
        "revision": "embedding-revision",
        "max_seq_length": 768,
    }
    assert captured["reranker_config"] == {
        "model": "custom/code-reranker",
        "revision": "reranker-revision",
        "max_length": 1024,
    }
    assert captured["top_k"] == 2
    assert selection.selected_chunks[0].chunk.path == "target.py"
    assert "reranker" in selection.selected_chunks[0].reason


def test_context_builder_allows_query_planner_ablation(tmp_path: Path, monkeypatch) -> None:
    from openharness.repopilot.query_planner import QueryPlanner

    (tmp_path / "service.py").write_text(
        "def combine_masks(left, right):\n return left | right\n",
        encoding="utf-8",
    )

    def unexpected_plan(_self, _text):
        raise AssertionError("query planner should be disabled")

    monkeypatch.setattr(QueryPlanner, "plan", unexpected_plan)
    selection = ContextBuilder(char_budget=500, query_planning=False).build(
        index=RepositoryIndex.build(tmp_path),
        query="combine_masks",
    )

    assert selection.selected_chunks[0].chunk.path == "service.py"


def test_context_builder_reserves_tight_budget_for_structural_neighbor(
    tmp_path: Path,
) -> None:
    (tmp_path / "service.py").write_text(
        "def helper(value):\n return value\n\n"
        "def target(value):\n return helper(value) + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "distractor.py").write_text(
        "def target_documentation():\n return 'target target target target target'\n",
        encoding="utf-8",
    )

    selection = ContextBuilder(
        char_budget=180,
        top_k=2,
        structural_expansion=True,
    ).build(
        index=RepositoryIndex.build(tmp_path),
        query="target",
    )

    assert selection.selected_chunks[0].chunk.symbol == "target"
    assert selection.selected_chunks[1].chunk.symbol == "helper"
    assert selection.selected_chunks[1].reason == "structure"
