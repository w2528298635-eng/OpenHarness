from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from .adapters import (
    AgentAdapter,
    EvaluationArm,
    InferenceBudget,
    build_arm_configs,
)
from .inference import InferenceArtifact, InferenceRequest, InferenceRunner
from .models import PublicInstance, SampleManifest
from .orchestration import (
    AgentRunError,
    CheckpointStore,
    ExperimentCheckpoint,
    ExperimentOrchestrator,
    InfrastructureRunError,
    RunCompletion,
    RunKey,
)
from .runners import (
    CurrentRepoPilotRunner,
    LegacyRepoPilotRunner,
    NativeOpenHarnessRunner,
)


class RepositoryCache(Protocol):
    def prepare(self, instance: PublicInstance, *, workspace_id: str) -> Path: ...

    def release(self, instance: PublicInstance, *, workspace_id: str) -> None: ...


def _workspace_id(key: RunKey) -> str:
    digest = hashlib.sha256(key.value.encode("utf-8")).hexdigest()[:20]
    return f"run-{digest}"


def build_experiment_adapters(
    *,
    model: str,
    legacy_source: Path,
    artifact_root: Path,
) -> dict[EvaluationArm, AgentAdapter]:
    configs = build_arm_configs(model=model)
    runners = {
        EvaluationArm.NATIVE: NativeOpenHarnessRunner(),
        EvaluationArm.LEGACY: LegacyRepoPilotRunner(
            legacy_source=legacy_source,
            artifact_root=artifact_root,
        ),
        EvaluationArm.UPGRADED_NO_RETRIEVAL: CurrentRepoPilotRunner(),
        EvaluationArm.UPGRADED_WITH_RETRIEVAL: CurrentRepoPilotRunner(),
    }
    return {
        arm: AgentAdapter(config=configs[arm], runner=runners[arm])
        for arm in EvaluationArm
    }


class InferenceExperimentExecutor:
    """Turn checkpoint run keys into sealed, evaluation-ready patches."""

    def __init__(
        self,
        *,
        manifest: SampleManifest,
        adapters: Mapping[EvaluationArm, AgentAdapter],
        repository_cache: RepositoryCache,
        inference_runner: InferenceRunner,
        artifact_directory: Path,
        budget: InferenceBudget,
    ) -> None:
        self._instances = {
            instance.instance_id: instance for instance in manifest.instances
        }
        self._adapters = dict(adapters)
        self._repository_cache = repository_cache
        self._inference_runner = inference_runner
        self._artifact_directory = artifact_directory
        self._budget = budget

    async def worker(self, key: RunKey) -> RunCompletion:
        try:
            instance = self._instances[key.instance_id]
            adapter = self._adapters[key.arm]
        except KeyError as exc:
            raise InfrastructureRunError(
                f"Run key is not configured in this experiment: {key.value}"
            ) from exc

        workspace_id = _workspace_id(key)
        prepared = False
        try:
            try:
                repository_path = self._repository_cache.prepare(
                    instance,
                    workspace_id=workspace_id,
                )
                prepared = True
            except Exception as exc:
                raise InfrastructureRunError(
                    f"Could not prepare repository for {key.value}: {exc}"
                ) from exc

            artifact_path = await self._inference_runner.run(
                request=InferenceRequest(
                    instance=instance,
                    arm=key.arm,
                    repetition=key.repetition,
                    budget=self._budget,
                ),
                adapter=adapter,
                repository_path=repository_path,
                output_directory=self._artifact_directory,
            )
            artifact = InferenceArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            if artifact.status != "completed" or not artifact.model_patch.strip():
                raise AgentRunError(
                    artifact.error
                    or f"{key.value} produced no evaluation-ready patch"
                )
            seal = artifact_path.with_suffix(".json.sha256").read_text(
                encoding="ascii"
            ).strip()
            return RunCompletion(
                artifact_path=str(artifact_path),
                artifact_sha256=seal,
                resolved=None,
            )
        finally:
            if prepared:
                try:
                    self._repository_cache.release(
                        instance,
                        workspace_id=workspace_id,
                    )
                except Exception as exc:
                    raise InfrastructureRunError(
                        f"Could not release repository for {key.value}: {exc}"
                    ) from exc


async def run_inference_matrix(
    *,
    manifest: SampleManifest,
    checkpoint_store: CheckpointStore,
    adapters: Mapping[EvaluationArm, AgentAdapter],
    repository_cache: RepositoryCache,
    artifact_directory: Path,
    budget: InferenceBudget,
    max_infrastructure_retries: int,
    inference_runner: InferenceRunner | None = None,
) -> ExperimentCheckpoint:
    checkpoint = checkpoint_store.load()
    if checkpoint.manifest_sha256 != manifest.sha256:
        raise InfrastructureRunError(
            "Checkpoint manifest digest does not match the inference manifest."
        )
    if not checkpoint.records:
        raise InfrastructureRunError(
            "Checkpoint contains no run keys; recreate it with pilot or run."
        )
    executor = InferenceExperimentExecutor(
        manifest=manifest,
        adapters=adapters,
        repository_cache=repository_cache,
        inference_runner=inference_runner or InferenceRunner(),
        artifact_directory=artifact_directory,
        budget=budget,
    )
    keys = [record.key for record in checkpoint.records.values()]
    return await ExperimentOrchestrator(
        checkpoint_store,
        max_infrastructure_retries=max_infrastructure_retries,
    ).run(keys, executor.worker)
