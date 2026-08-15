import json
from pathlib import Path
from types import SimpleNamespace

from openharness.repopilot.reranker import LocalCrossEncoderReranker, blend_rankings
from openharness.repopilot.retrieval import CodeChunk


def test_cross_encoder_ranks_only_supplied_candidates_and_uses_stable_cache(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["payload"] = json.loads(kwargs["input"])
        return SimpleNamespace(
            returncode=0,
            stdout='{"ranked": [[1, 8.4], [0, 1.2]], "cache_hits": 1, "cache_misses": 1}',
            stderr="",
        )

    monkeypatch.setattr("openharness.repopilot.reranker.subprocess.run", fake_run)
    candidates = [
        CodeChunk(
            chunk_id="first",
            path="first.py",
            start_line=1,
            end_line=1,
            kind="text",
            text="unrelated = True",
        ),
        CodeChunk(
            chunk_id="second",
            path="service.py",
            start_line=4,
            end_line=5,
            symbol="combine_masks",
            kind="functiondef",
            text="def combine_masks(a, b):\n return a | b\n",
        ),
    ]

    reranker = LocalCrossEncoderReranker(cache_directory=tmp_path)
    ranked = reranker.rank("mask propagation fails for None", candidates, top_k=2)

    assert ranked == [(1, 8.4), (0, 1.2)]
    assert reranker.last_stats == {"cache_hits": 1, "cache_misses": 1}
    assert captured["payload"]["texts"] == [
        "File: first.py\nKind: text\nCode:\nunrelated = True",
        "File: service.py\nSymbol: combine_masks\nKind: functiondef\nCode:\ndef combine_masks(a, b):\n return a | b\n",
    ]
    assert captured["payload"]["query"] == "mask propagation fails for None"
    assert captured["payload"]["candidate_ids"] == ["first", "second"]
    assert len(captured["payload"]["pair_ids"]) == 2
    assert captured["payload"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["payload"]["revision"] == (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )
    assert captured["payload"]["max_length"] == 512
    assert captured["payload"]["top_k"] == 2
    assert captured["payload"]["score_store"].endswith("reranker-v1.sqlite3")


def test_cross_encoder_cache_identity_changes_with_model_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = []

    def fake_run(_argv, **kwargs):
        payloads.append(json.loads(kwargs["input"]))
        return SimpleNamespace(
            returncode=0,
            stdout='{"ranked": [[0, 0.5]], "cache_hits": 0, "cache_misses": 1}',
            stderr="",
        )

    monkeypatch.setattr("openharness.repopilot.reranker.subprocess.run", fake_run)
    chunk = CodeChunk(
        chunk_id="candidate",
        path="service.py",
        start_line=1,
        end_line=1,
        kind="text",
        text="value = 1",
    )

    LocalCrossEncoderReranker(cache_directory=tmp_path).rank("value", [chunk], top_k=1)
    LocalCrossEncoderReranker(
        cache_directory=tmp_path,
        model="custom/reranker",
        revision="revision-2",
        max_length=768,
    ).rank("value", [chunk], top_k=1)

    assert payloads[0]["model_key"] != payloads[1]["model_key"]
    assert payloads[1]["model"] == "custom/reranker"
    assert payloads[1]["revision"] == "revision-2"
    assert payloads[1]["max_length"] == 768


def test_cross_encoder_returns_empty_without_starting_worker(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("worker should not start for an empty candidate set")

    monkeypatch.setattr("openharness.repopilot.reranker.subprocess.run", unexpected_run)
    reranker = LocalCrossEncoderReranker(cache_directory=tmp_path)

    assert reranker.rank("query", [], top_k=12) == []
    assert reranker.last_stats == {"cache_hits": 0, "cache_misses": 0}


def test_cross_encoder_surfaces_worker_failure(tmp_path: Path, monkeypatch) -> None:
    def fake_run(_argv, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="model load failed")

    monkeypatch.setattr("openharness.repopilot.reranker.subprocess.run", fake_run)
    chunk = CodeChunk(
        chunk_id="candidate",
        path="service.py",
        start_line=1,
        end_line=1,
        kind="text",
        text="value = 1",
    )

    try:
        LocalCrossEncoderReranker(cache_directory=tmp_path).rank(
            "value", [chunk], top_k=1
        )
    except RuntimeError as error:
        assert "model load failed" in str(error)
    else:
        raise AssertionError("worker failure should have been surfaced")


def test_blended_ranking_normalizes_both_signals_before_weighting() -> None:
    """A marginal reranker disagreement must not erase a strong fused lead."""
    ranked = blend_rankings(
        base_scores=[0.9, 0.8, 0.1],
        reranked=[(1, 8.0), (0, 7.0), (2, -3.0)],
        reranker_weight=0.5,
        top_k=2,
    )

    assert [index for index, _score in ranked] == [0, 1]
    assert ranked[0][1] > ranked[1][1]


def test_explicit_reranker_identity_cannot_be_overridden_by_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("REPOPILOT_RERANKER_MODEL", "environment/model")
    monkeypatch.setenv("REPOPILOT_RERANKER_REVISION", "environment-revision")

    reranker = LocalCrossEncoderReranker(
        cache_directory=tmp_path,
        model="locked/model",
        revision="locked-revision",
    )

    assert reranker.model == "locked/model"
    assert reranker.revision == "locked-revision"
