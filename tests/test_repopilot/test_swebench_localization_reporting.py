from openharness.repopilot.swebench.localization import LocalizationMetrics
from openharness.repopilot.swebench.localization_execution import (
    LocalizationCheckpoint,
    LocalizationRecord,
    LocalizationRunConfig,
)
from openharness.repopilot.swebench.localization_reporting import (
    build_localization_report,
)
from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
    SampleManifest,
    SamplingConfig,
)


def _record(instance_id: str, *, recall5: float, rank: int | None) -> LocalizationRecord:
    return LocalizationRecord(
        instance_id=instance_id,
        status="completed",
        index_seconds=2.0,
        retrieval_seconds=4.0,
        indexed_chunks=100,
        metrics=LocalizationMetrics(
            gold_file_denominator=1,
            symbol_denominator=0,
            recall_at={1: 1.0 if rank == 1 else 0.0, 3: recall5, 5: recall5, 10: recall5},
            hit_at={1: rank == 1, 3: bool(recall5), 5: bool(recall5), 10: bool(recall5)},
            precision_at={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
            ndcg_at={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
            symbol_recall_at=None,
            mrr=1.0 / rank if rank else 0.0,
            first_relevant_file_rank=rank,
            first_relevant_symbol_rank=None,
            context_characters=4000,
            estimated_context_tokens=1000,
            irrelevant_context_rate=0.5 if recall5 else 1.0,
            relevant_file_hits_per_1000_tokens=1.0 if recall5 else 0.0,
        ),
    )


def test_localization_report_aggregates_overall_and_difficulty() -> None:
    easy = PublicInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="a",
        problem_statement="easy",
        source_difficulty="<15 min fix",
        difficulty=DifficultyStratum.EASY,
    )
    hard = PublicInstance(
        instance_id="owner__repo-2",
        repo="owner/repo",
        base_commit="b",
        problem_statement="hard",
        source_difficulty="1-4 hours",
        difficulty=DifficultyStratum.HARD,
    )
    manifest = SampleManifest(
        dataset_name="fixture/verified",
        dataset_revision="sha",
        sampling=SamplingConfig(easy=1, medium=0, hard=1, seed=1),
        instances=(easy, hard),
        sha256="a" * 64,
    )
    checkpoint = LocalizationCheckpoint(
        configuration=LocalizationRunConfig(retrieval_strategy="hybrid"),
        records={
            easy.instance_id: _record(easy.instance_id, recall5=1.0, rank=1),
            hard.instance_id: _record(hard.instance_id, recall5=0.0, rank=None),
        },
    )

    report = build_localization_report(checkpoint, manifest)

    assert report.overall.tasks == 2
    assert report.overall.recall_at[5] == 0.5
    assert report.overall.hit_at[5] == 0.5
    assert report.overall.mrr == 0.5
    assert report.by_difficulty["easy"].recall_at[5] == 1.0
    assert report.by_difficulty["hard"].recall_at[5] == 0.0
    assert report.configuration.retrieval_strategy == "hybrid"
