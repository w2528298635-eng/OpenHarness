from pathlib import Path

from openharness.repopilot.query_planner import QueryPlanner
from openharness.repopilot.retrieval import RepositoryIndex, RetrievalQuery


def test_repository_index_builds_python_symbol_chunks_and_ignores_noise(
    tmp_path: Path,
) -> None:
    (tmp_path / "pricing.py").write_text(
        '"""Pricing helpers."""\n'
        "import decimal\n\n"
        "def final_price(total, discount_rate):\n"
        "    return total * (1 - discount_rate)\n\n"
        "class Cart:\n"
        "    def total(self):\n"
        "        return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("discount policy", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
    ignored = tmp_path / ".git"
    ignored.mkdir()
    (ignored / "config").write_text("discount secret", encoding="utf-8")

    index = RepositoryIndex.build(tmp_path)

    assert {chunk.symbol for chunk in index.chunks} >= {"final_price", "Cart"}
    assert any(chunk.path == "notes.txt" for chunk in index.chunks)
    assert not any(chunk.path.startswith(".git") for chunk in index.chunks)
    assert not any(chunk.path == "binary.bin" for chunk in index.chunks)
    assert len({chunk.chunk_id for chunk in index.chunks}) == len(index.chunks)


def test_lexical_search_explains_symbol_and_path_relevance(tmp_path: Path) -> None:
    (tmp_path / "pricing.py").write_text(
        "def final_price(total, discount_rate):\n    return total * (1 - discount_rate)\n",
        encoding="utf-8",
    )
    (tmp_path / "shipping.py").write_text(
        "def shipping_cost(weight):\n    return weight * 2\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)

    results = index.search(RetrievalQuery(text="final_price discount boundary", top_k=3))

    assert results.matches[0].chunk.path == "pricing.py"
    assert results.matches[0].score > 0
    assert "symbol" in results.matches[0].reasons


def test_index_respects_allowed_paths_and_file_size_limit(tmp_path: Path) -> None:
    (tmp_path / "allowed.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "blocked.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "large.py").write_text("x" * 1000, encoding="utf-8")

    index = RepositoryIndex.build(
        tmp_path,
        allowed_paths=["allowed.py", "large.py"],
        max_file_bytes=100,
    )

    assert [chunk.path for chunk in index.chunks] == ["allowed.py"]


def test_hybrid_search_can_promote_semantically_relevant_code(tmp_path: Path) -> None:
    (tmp_path / "mask.py").write_text(
        "def combine_masks(left, right):\n    return left | right\n", encoding="utf-8"
    )
    (tmp_path / "other.py").write_text("def discount(price):\n    return price\n", encoding="utf-8")
    index = RepositoryIndex.build(tmp_path)

    def encode(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "mask" in text or "NoneType" in text else [0.0, 1.0] for text in texts]

    results = index.hybrid_search(RetrievalQuery(text="NoneType mask failure"), encoder=encode)

    assert results.matches[0].chunk.path == "mask.py"
    assert "semantic" in results.matches[0].reasons


def test_hybrid_search_returns_empty_without_invoking_encoder(tmp_path: Path) -> None:
    class UnexpectedEncoder:
        def rank_many(self, *_args, **_kwargs):
            raise AssertionError("empty indexes should not invoke the encoder")

    result = RepositoryIndex.build(tmp_path).hybrid_search(
        RetrievalQuery(text="missing code", top_k=5),
        encoder=UnexpectedEncoder(),
    )

    assert result.matches == []
    assert result.indexed_chunks == 0


def test_query_planner_extracts_error_symbols_paths_and_multiple_queries() -> None:
    plan = QueryPlanner().plan(
        "TypeError: unsupported operand in `combine_masks`; see astropy.nddata.mixins.ndarithmetic"
    )

    assert "TypeError" in plan.errors
    assert "combine_masks" in plan.identifiers
    assert "astropy.nddata.mixins.ndarithmetic" in plan.paths
    assert len(plan.queries) >= 2


def test_query_planner_extracts_python_paths_and_camel_case_symbols() -> None:
    plan = QueryPlanner().plan(
        "ScalarFormatter loses offsets in sklearn/pipeline.py and raises ValueError"
    )

    assert "ScalarFormatter" in plan.identifiers
    assert "sklearn/pipeline.py" in plan.paths
    assert any(query == "ScalarFormatter ValueError" for query in plan.queries)
    assert "sklearn/pipeline.py" in plan.queries


def test_lexical_search_can_disable_query_planning(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "service.py").write_text(
        "def combine_masks(left, right):\n return left | right\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)

    def unexpected_plan(_self, _text):
        raise AssertionError("query planner should be bypassed")

    monkeypatch.setattr(QueryPlanner, "plan", unexpected_plan)

    result = index.search(
        RetrievalQuery(text="combine_masks", top_k=1),
        query_planning=False,
    )

    assert result.matches[0].chunk.path == "service.py"


def test_hybrid_search_merges_independent_dense_candidates(tmp_path: Path) -> None:
    (tmp_path / "literal.py").write_text("def requested_name():\n return 1\n", encoding="utf-8")
    (tmp_path / "semantic.py").write_text("def unrelated_words():\n return 2\n", encoding="utf-8")
    index = RepositoryIndex.build(tmp_path)

    class DenseRanker:
        def rank(self, query, chunks, *, top_k):
            return [(next(i for i, chunk in enumerate(chunks) if chunk.path == "semantic.py"), 0.99)]

    result = index.hybrid_search(
        RetrievalQuery(text="requested_name", top_k=2), encoder=DenseRanker()
    )

    assert {match.chunk.path for match in result.matches} == {"literal.py", "semantic.py"}
    assert any("dense" in match.reasons for match in result.matches)


def test_query_planner_drives_dense_multi_query_retrieval(tmp_path: Path) -> None:
    (tmp_path / "mask.py").write_text(
        "def combine_masks(left, right):\n return left | right\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)

    class DenseRanker:
        def __init__(self):
            self.queries = ()

        def rank_many(self, queries, chunks, *, top_k):
            self.queries = tuple(queries)
            return [(0, 0.9)]

    ranker = DenseRanker()
    index.hybrid_search(
        RetrievalQuery(
            text="TypeError while calling `combine_masks` in astropy.nddata.mask",
            top_k=1,
        ),
        encoder=ranker,
    )

    assert len(ranker.queries) >= 2
    assert any("combine_masks" in query for query in ranker.queries[1:])


def test_structure_expansion_adds_same_file_and_symbol_reference_neighbors(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(
        "def normalize_mask(mask):\n return mask\n\n"
        "def combine_masks(left, right):\n return normalize_mask(left) | normalize_mask(right)\n",
        encoding="utf-8",
    )
    (tmp_path / "api.py").write_text(
        "from service import combine_masks\n\ndef apply_mask(a, b):\n return combine_masks(a, b)\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)
    seed = next(chunk for chunk in index.chunks if chunk.symbol == "combine_masks")

    expanded = index.expand_structure([seed], limit=4)

    assert {chunk.symbol for chunk in expanded} >= {"combine_masks", "normalize_mask", "apply_mask"}


def test_structure_expansion_reserves_space_for_cross_file_callers(tmp_path: Path) -> None:
    helpers = "\n\n".join(
        f"def helper_{index}():\n return {index}" for index in range(12)
    )
    (tmp_path / "service.py").write_text(
        f"{helpers}\n\ndef target(value):\n return value\n",
        encoding="utf-8",
    )
    (tmp_path / "api.py").write_text(
        "from service import target\n\ndef call_target(value):\n return target(value)\n",
        encoding="utf-8",
    )
    index = RepositoryIndex.build(tmp_path)
    seed = next(chunk for chunk in index.chunks if chunk.symbol == "target")

    expanded = index.expand_structure([seed], limit=5)

    assert any(chunk.path == "service.py" and chunk.chunk_id != seed.chunk_id for chunk in expanded)
    assert any(chunk.path == "api.py" and chunk.symbol == "call_target" for chunk in expanded)
