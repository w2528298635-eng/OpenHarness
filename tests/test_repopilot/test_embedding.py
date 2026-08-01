import json
from pathlib import Path
from types import SimpleNamespace

from openharness.repopilot.embedding import LocalEmbeddingEncoder
from openharness.repopilot.retrieval import CodeChunk


def test_local_encoder_rank_uses_independent_dense_worker_and_stable_cache(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["payload"] = json.loads(kwargs["input"])
        return SimpleNamespace(
            stdout='{"ranked": [[1, 0.91]], "cache_hits": 1, "cache_misses": 1}'
        )

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", fake_run)
    encoder = LocalEmbeddingEncoder(cache_directory=tmp_path)
    chunks = [
        CodeChunk(chunk_id="a", path="a.py", start_line=1, end_line=1, kind="text", text="alpha"),
        CodeChunk(chunk_id="b", path="b.py", start_line=1, end_line=1, kind="text", text="beta"),
    ]

    result = encoder.rank("semantic request", chunks, top_k=1)

    assert result == [(1, 0.91)]
    assert encoder.last_stats == {"cache_hits": 1, "cache_misses": 1}
    assert captured["payload"]["texts"] == [
        "File: a.py\nKind: text\nCode:\nalpha",
        "File: b.py\nKind: text\nCode:\nbeta",
    ]
    assert captured["payload"]["queries"] == ["semantic request"]
    assert captured["payload"]["top_k"] == 1
    assert captured["payload"]["max_seq_length"] == 512
    assert captured["payload"]["chunk_ids"] == ["a", "b"]
    assert len(captured["payload"]["embedding_ids"]) == 2
    assert captured["payload"]["embedding_ids"] != ["a", "b"]
    assert captured["payload"]["model"] == "nomic-ai/CodeRankEmbed"
    assert captured["payload"]["revision"] == (
        "3c4b60807d71f79b43f3c4363786d9493691f8b1"
    )
    assert captured["payload"]["query_prefix"] == (
        "Represent this query for searching relevant code: "
    )
    assert "CodeRankEmbed" in captured["payload"]["model_key"]
    assert "legacy_model_key" not in captured["payload"]
    assert captured["payload"]["vector_store"].endswith("embeddings-v2.sqlite3")
    assert str(tmp_path) in captured["payload"]["cache_file"]


def test_local_encoder_keeps_model_configurations_in_separate_caches(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = []

    def fake_run(_argv, **kwargs):
        payloads.append(json.loads(kwargs["input"]))
        return SimpleNamespace(
            stdout='{"ranked": [[0, 0.8]], "cache_hits": 0, "cache_misses": 1}'
        )

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", fake_run)
    chunk = CodeChunk(
        chunk_id="chunk",
        path="service.py",
        start_line=1,
        end_line=2,
        symbol="target",
        kind="functiondef",
        text="def target():\n return 1\n",
    )

    LocalEmbeddingEncoder(cache_directory=tmp_path).rank("target", [chunk], top_k=1)
    LocalEmbeddingEncoder(
        cache_directory=tmp_path,
        model="custom/code-model",
        revision="revision-2",
        query_prefix="code search: ",
        max_seq_length=768,
    ).rank("target", [chunk], top_k=1)

    assert payloads[0]["model_key"] != payloads[1]["model_key"]
    assert payloads[0]["cache_file"] != payloads[1]["cache_file"]
    assert payloads[1]["model"] == "custom/code-model"
    assert payloads[1]["revision"] == "revision-2"
    assert payloads[1]["query_prefix"] == "code search: "
    assert payloads[1]["max_seq_length"] == 768


def test_embedding_cache_identity_ignores_chunk_line_identity(tmp_path: Path, monkeypatch) -> None:
    payloads = []

    def fake_run(_argv, **kwargs):
        payloads.append(json.loads(kwargs["input"]))
        return SimpleNamespace(
            stdout='{"ranked": [[0, 0.8]], "cache_hits": 0, "cache_misses": 1}'
        )

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", fake_run)
    encoder = LocalEmbeddingEncoder(cache_directory=tmp_path)
    first = CodeChunk(
        chunk_id="revision-a",
        path="service.py",
        start_line=10,
        end_line=11,
        symbol="target",
        kind="functiondef",
        text="def target():\n return 1\n",
    )
    second = first.model_copy(
        update={"chunk_id": "revision-b", "start_line": 40, "end_line": 41}
    )

    encoder.rank("target", [first], top_k=1)
    encoder.rank("target", [second], top_k=1)

    assert payloads[0]["embedding_ids"] == payloads[1]["embedding_ids"]
    assert payloads[0]["cache_file"] == payloads[1]["cache_file"]


def test_local_encoder_ranks_multiple_queries_in_one_worker_call(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_run(_argv, **kwargs):
        captured.update(json.loads(kwargs["input"]))
        return SimpleNamespace(
            stdout='{"ranked": [[0, 0.92]], "cache_hits": 1, "cache_misses": 0}'
        )

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", fake_run)
    chunk = CodeChunk(
        chunk_id="a",
        path="service.py",
        start_line=1,
        end_line=2,
        symbol="combine_masks",
        kind="functiondef",
        text="def combine_masks(a, b):\n return a | b\n",
    )

    result = LocalEmbeddingEncoder(cache_directory=tmp_path).rank_many(
        ["mask TypeError", "combine_masks"],
        [chunk],
        top_k=1,
    )

    assert result == [(0, 0.92)]
    assert captured["queries"] == ["mask TypeError", "combine_masks"]


def test_local_encoder_surfaces_worker_stderr(tmp_path: Path, monkeypatch) -> None:
    def fake_run(_argv, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="database is locked")

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", fake_run)
    chunk = CodeChunk(
        chunk_id="a",
        path="service.py",
        start_line=1,
        end_line=1,
        kind="text",
        text="value = 1",
    )

    try:
        LocalEmbeddingEncoder(cache_directory=tmp_path).rank("value", [chunk], top_k=1)
    except RuntimeError as error:
        assert "database is locked" in str(error)
    else:
        raise AssertionError("worker failure should have been surfaced")


def test_local_encoder_returns_empty_for_empty_chunk_set(tmp_path: Path, monkeypatch) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("worker should not start for an empty chunk set")

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", unexpected_run)
    encoder = LocalEmbeddingEncoder(cache_directory=tmp_path)

    assert encoder.rank_many(["query"], [], top_k=5) == []
    assert encoder.last_stats == {"cache_hits": 0, "cache_misses": 0}
