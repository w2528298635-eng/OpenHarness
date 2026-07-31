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
        return SimpleNamespace(stdout='[[1, 0.91]]')

    monkeypatch.setattr("openharness.repopilot.embedding.subprocess.run", fake_run)
    encoder = LocalEmbeddingEncoder(cache_directory=tmp_path)
    chunks = [
        CodeChunk(chunk_id="a", path="a.py", start_line=1, end_line=1, kind="text", text="alpha"),
        CodeChunk(chunk_id="b", path="b.py", start_line=1, end_line=1, kind="text", text="beta"),
    ]

    result = encoder.rank("semantic request", chunks, top_k=1)

    assert result == [(1, 0.91)]
    assert captured["payload"]["texts"] == ["a.py \nalpha", "b.py \nbeta"]
    assert captured["payload"]["top_k"] == 1
    assert str(tmp_path) in captured["payload"]["cache_file"]
