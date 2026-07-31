from pathlib import Path

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
