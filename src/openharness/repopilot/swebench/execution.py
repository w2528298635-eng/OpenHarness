from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from openharness.repopilot.swebench.docker_runner import (
    HarnessPrediction,
    HarnessResult,
    write_predictions_jsonl,
)
from openharness.repopilot.swebench.inference import (
    InferenceArtifact,
    verify_artifact_seal,
)
from openharness.repopilot.swebench.orchestration import (
    CheckpointStore,
    ExperimentCheckpoint,
    InfrastructureRunError,
    RunKey,
    RunStatus,
)


class BatchHarnessEvaluator:
    """Evaluate a homogeneous batch of sealed inference artifacts."""

    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStore,
        harness: object,
        dataset_name: str,
        output_directory: Path,
        max_workers: int = 1,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._harness = harness
        self._dataset_name = dataset_name
        self._output_directory = output_directory
        self._max_workers = max_workers

    def evaluate(self, keys: Sequence[RunKey]) -> HarnessResult:
        if not keys:
            raise ValueError("At least one run key is required for evaluation.")

        first = keys[0]
        if any(
            key.arm != first.arm or key.repetition != first.repetition
            for key in keys[1:]
        ):
            raise ValueError(
                "A harness batch must use one evaluation arm and repetition."
            )

        checkpoint = self._checkpoint_store.load()
        predictions: list[HarnessPrediction] = []

        for key in keys:
            record = checkpoint.record_for(key)
            if record.status not in {
                RunStatus.EVALUATION_PENDING,
                RunStatus.COMPLETED,
            }:
                raise InfrastructureRunError(
                    f"Inference is not ready for official evaluation: {key.value}"
                )
            if record.artifact_path is None or record.artifact_sha256 is None:
                raise InfrastructureRunError(
                    f"Missing sealed inference artifact: {key.value}"
                )

            artifact_path = Path(record.artifact_path)
            if not verify_artifact_seal(artifact_path):
                raise InfrastructureRunError(
                    f"Inference artifact seal is invalid: {key.value}"
                )
            persisted_sha256 = artifact_path.with_suffix(
                ".json.sha256"
            ).read_text(encoding="ascii").strip()
            if persisted_sha256 != record.artifact_sha256:
                raise InfrastructureRunError(
                    f"Checkpoint artifact digest does not match seal: {key.value}"
                )
            artifact = InferenceArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            if (
                artifact.instance_id != key.instance_id
                or artifact.arm != key.arm
                or artifact.repetition != key.repetition
            ):
                raise InfrastructureRunError(
                    f"Inference artifact does not match run key: {key.value}"
                )

            predictions.append(
                HarnessPrediction(
                    instance_id=key.instance_id,
                    model_name_or_path=(
                        f"repopilot/{artifact.model}/{key.arm.value}/"
                        f"r{key.repetition}"
                    ),
                    model_patch=artifact.model_patch,
                )
            )

        self._output_directory.mkdir(parents=True, exist_ok=True)
        batch_slug = f"{first.arm.value}-r{first.repetition}"
        predictions_path = self._output_directory / f"{batch_slug}.predictions.jsonl"
        result_path = self._output_directory / f"{batch_slug}.result.json"
        write_predictions_jsonl(predictions_path, predictions)

        result = self._harness.evaluate(
            dataset_name=self._dataset_name,
            predictions_path=predictions_path,
            run_id=batch_slug,
            result_path=result_path,
            max_workers=self._max_workers,
            cache_level="env",
            clean=True,
            instance_ids=tuple(key.instance_id for key in keys),
        )
        if result.status != "completed":
            raise InfrastructureRunError(
                f"Official SWE-bench evaluation failed: {result.status}"
            )
        if result.resolved and not result.resolved_instance_ids:
            raise InfrastructureRunError(
                "Official report omitted resolved instance IDs; aggregate results "
                "cannot be mapped back to checkpoint records."
            )

        resolved_ids = set(result.resolved_instance_ids)
        for key in keys:
            checkpoint = checkpoint.with_evaluation(
                key,
                resolved=key.instance_id in resolved_ids,
            )
        self._checkpoint_store.save(checkpoint)
        return result


def evaluate_pending_matrix(
    *,
    checkpoint_store: CheckpointStore,
    harness: object,
    dataset_name: str,
    output_directory: Path,
    max_workers: int = 1,
) -> ExperimentCheckpoint:
    """Evaluate each arm/repetition batch without repeating completed records."""
    checkpoint = checkpoint_store.load()
    groups: dict[tuple[str, int], list[RunKey]] = {}
    for record in checkpoint.records.values():
        if record.status is not RunStatus.EVALUATION_PENDING:
            continue
        group = (record.key.arm.value, record.key.repetition)
        groups.setdefault(group, []).append(record.key)

    evaluator = BatchHarnessEvaluator(
        checkpoint_store=checkpoint_store,
        harness=harness,
        dataset_name=dataset_name,
        output_directory=output_directory,
        max_workers=max_workers,
    )
    for group in sorted(groups):
        keys = sorted(groups[group], key=lambda key: key.instance_id)
        evaluator.evaluate(keys)
    return checkpoint_store.load()
