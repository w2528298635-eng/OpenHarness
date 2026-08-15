from pathlib import Path

import pytest


def test_evaluator_records_one_completed_instance_and_can_resume(tmp_path: Path) -> None:
    from openharness.repopilot.swebench.localization_execution import (
        LocalizationCheckpointStore,
        evaluate_localization_instance,
    )
    from openharness.repopilot.swebench.models import DifficultyStratum, PublicInstance

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pricing.py").write_text(
        "def final_price(total, discount):\n    return total - discount\n",
        encoding="utf-8",
    )
    instance = PublicInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="abc",
        problem_statement="final_price discounts incorrectly",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )
    patch = (
        "diff --git a/pricing.py b/pricing.py\n"
        "--- a/pricing.py\n"
        "+++ b/pricing.py\n"
        "@@ -1 +1 @@\n"
        "-def final_price(total, discount):\n"
        "+def final_price(total, discount, tax=0):\n"
    )
    store = LocalizationCheckpointStore(tmp_path / "localization.json")

    result = evaluate_localization_instance(
        instance=instance,
        repository=repository,
        gold_patch=patch,
        store=store,
    )

    assert result.instance_id == instance.instance_id
    assert result.status == "completed"
    assert result.metrics.recall_at[1] == 1.0
    assert store.load().records[instance.instance_id].status == "completed"
    assert evaluate_localization_instance(
        instance=instance,
        repository=repository,
        gold_patch=patch,
        store=store,
    ).model_dump() == result.model_dump()


def test_manifest_evaluator_uses_gold_only_after_public_instance_is_loaded(
    tmp_path: Path,
) -> None:
    from openharness.repopilot.swebench.localization_execution import (
        LocalizationCheckpointStore,
        evaluate_localization_manifest,
    )
    from openharness.repopilot.swebench.models import DifficultyStratum, PublicInstance

    instance = PublicInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="abc",
        problem_statement="discount calculation",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )
    repository = tmp_path / "sources" / "formal-owner__repo-1"
    repository.mkdir(parents=True)
    (repository / "pricing.py").write_text("DISCOUNT = 1\n", encoding="utf-8")
    checkpoint = evaluate_localization_manifest(
        instances=(instance,),
        public_rows_with_gold=[
            {
                "instance_id": instance.instance_id,
                "patch": (
                    "diff --git a/pricing.py b/pricing.py\n"
                    "--- a/pricing.py\n+++ b/pricing.py\n@@ -1 +1 @@\n"
                    "-DISCOUNT = 1\n+DISCOUNT = 2\n"
                ),
            }
        ],
        repository_root=tmp_path / "sources",
        store=LocalizationCheckpointStore(tmp_path / "checkpoint.json"),
    )

    assert checkpoint.records[instance.instance_id].metrics.recall_at[1] == 1.0


def test_localization_evaluator_exposes_query_and_structure_ablation(
    tmp_path: Path, monkeypatch
) -> None:
    from openharness.repopilot.query_planner import QueryPlanner
    from openharness.repopilot.swebench.localization_execution import (
        LocalizationCheckpointStore,
        evaluate_localization_instance,
    )
    from openharness.repopilot.swebench.models import DifficultyStratum, PublicInstance

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "service.py").write_text("TARGET = 1\n", encoding="utf-8")
    instance = PublicInstance(
        instance_id="owner__repo-2",
        repo="owner/repo",
        base_commit="abc",
        problem_statement="TARGET is incorrect",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )

    def unexpected_plan(_self, _text):
        raise AssertionError("query planning should be disabled")

    monkeypatch.setattr(QueryPlanner, "plan", unexpected_plan)
    result = evaluate_localization_instance(
        instance=instance,
        repository=repository,
        gold_patch=(
            "diff --git a/service.py b/service.py\n"
            "--- a/service.py\n+++ b/service.py\n@@ -1 +1 @@\n"
            "-TARGET = 1\n+TARGET = 2\n"
        ),
        store=LocalizationCheckpointStore(tmp_path / "ablation.json"),
        query_planning=False,
        structural_expansion=False,
    )

    assert result.status == "completed"
    with pytest.raises(ValueError, match="configuration does not match"):
        evaluate_localization_instance(
            instance=instance,
            repository=repository,
            gold_patch=(
                "diff --git a/service.py b/service.py\n"
                "--- a/service.py\n+++ b/service.py\n@@ -1 +1 @@\n"
                "-TARGET = 1\n+TARGET = 2\n"
            ),
            store=LocalizationCheckpointStore(tmp_path / "ablation.json"),
            query_planning=True,
            structural_expansion=False,
        )


def test_localization_evaluator_rejects_empty_repository(tmp_path: Path) -> None:
    from openharness.repopilot.swebench.localization_execution import (
        LocalizationCheckpointStore,
        evaluate_localization_instance,
    )
    from openharness.repopilot.swebench.models import DifficultyStratum, PublicInstance

    repository = tmp_path / "empty"
    repository.mkdir()
    instance = PublicInstance(
        instance_id="owner__empty-1",
        repo="owner/empty",
        base_commit="abc",
        problem_statement="missing implementation",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )

    with pytest.raises(ValueError, match="no indexable source chunks"):
        evaluate_localization_instance(
            instance=instance,
            repository=repository,
            gold_patch=(
                "diff --git a/service.py b/service.py\n"
                "--- a/service.py\n+++ b/service.py\n@@ -0,0 +1 @@\n+VALUE = 1\n"
            ),
            store=LocalizationCheckpointStore(tmp_path / "empty.json"),
        )


def test_localization_run_configuration_tracks_reranker_ablation() -> None:
    from openharness.repopilot.swebench.localization_execution import (
        LocalizationRunConfig,
    )

    configuration = LocalizationRunConfig(
        retrieval_strategy="hybrid",
        embedding_model="nomic-ai/CodeRankEmbed",
        embedding_revision="3c4b60807d71f79b43f3c4363786d9493691f8b1",
        embedding_max_seq_length=512,
        reranker="cross_encoder",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        reranker_max_length=512,
        reranker_candidate_k=40,
        reranker_weight=0.5,
        reranker_strict=True,
    )

    assert configuration.reranker == "cross_encoder"
    assert configuration.embedding_model == "nomic-ai/CodeRankEmbed"
    assert configuration.embedding_max_seq_length == 512
    assert configuration.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert configuration.reranker_max_length == 512
    assert configuration.reranker_candidate_k == 40
    assert configuration.reranker_weight == 0.5
    assert configuration.reranker_strict is True
