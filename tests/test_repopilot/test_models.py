from pathlib import Path

import pytest
from pydantic import ValidationError

from openharness.repopilot.models import (
    AnalysisResult,
    Phase,
    RepoRunState,
    RepoTaskSpec,
    RetrievalConfig,
)


def test_task_requires_non_empty_issue_and_command(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RepoTaskSpec(repo_path=tmp_path, issue=" ", verify_command=["pytest"])
    with pytest.raises(ValidationError):
        RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=[])


def test_analysis_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult(suspected_files=[], root_cause="x", evidence=[], confidence=1.1)


def test_structural_expansion_is_opt_in_by_default() -> None:
    assert RetrievalConfig().structural_expansion is False


def test_cross_encoder_reranker_is_explicit_and_bounded() -> None:
    default = RetrievalConfig()

    assert default.reranker == "none"
    assert default.embedding_model == "nomic-ai/CodeRankEmbed"
    assert default.embedding_revision == "3c4b60807d71f79b43f3c4363786d9493691f8b1"
    assert default.embedding_max_seq_length == 512
    assert default.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert default.reranker_revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert default.reranker_max_length == 512
    assert default.reranker_candidate_k == 40
    assert default.reranker_weight == 0.5
    configured = RetrievalConfig(
        reranker="cross_encoder",
        embedding_model="custom/code-encoder",
        embedding_revision="embedding-revision",
        embedding_max_seq_length=768,
        reranker_model="custom/code-reranker",
        reranker_revision="reranker-revision",
        reranker_max_length=1024,
        reranker_candidate_k=60,
        reranker_weight=0.7,
        reranker_strict=True,
    )
    assert configured.reranker == "cross_encoder"
    assert configured.embedding_model == "custom/code-encoder"
    assert configured.reranker_model == "custom/code-reranker"
    assert configured.reranker_strict is True

    with pytest.raises(ValidationError):
        RetrievalConfig(reranker_candidate_k=101)
    with pytest.raises(ValidationError):
        RetrievalConfig(reranker_weight=1.01)


def test_run_state_round_trips_json(tmp_path: Path) -> None:
    task = RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"])
    state = RepoRunState(run_id="run-1", task=task, phase=Phase.PRECHECK)

    restored = RepoRunState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.budgets.phase_calls == 0
