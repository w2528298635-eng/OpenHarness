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
