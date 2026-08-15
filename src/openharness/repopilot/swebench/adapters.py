from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .localization import RetrievedLocation
from .models import PublicInstance

LEGACY_REPOPILOT_COMMIT = "15fb5947bff15fccb2faea186240fcd76ec0e2ab"
_DEPRECATED_MODELS = {"deepseek-chat", "deepseek-reasoner"}


class EvaluationArm(str, Enum):
    NATIVE = "native_openharness"
    LEGACY = "legacy_repopilot"
    UPGRADED_NO_RETRIEVAL = "upgraded_no_retrieval"
    UPGRADED_WITH_RETRIEVAL = "upgraded_with_retrieval"


class InferenceBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_model_calls: int = Field(ge=1)
    max_total_tokens: int = Field(ge=1)
    max_wall_seconds: float = Field(gt=0)


class ArmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: EvaluationArm
    provider: str = "deepseek"
    model: str
    temperature: float = 0
    prompt_version: str = "swebench-v1"
    retrieval_enabled: bool = False
    legacy_commit: str | None = None

    @field_validator("model")
    @classmethod
    def reject_deprecated_model_aliases(cls, value: str) -> str:
        if value in _DEPRECATED_MODELS:
            raise ValueError(
                f"deprecated DeepSeek model alias is not valid for formal runs: {value}"
            )
        return value

    @model_validator(mode="after")
    def validate_arm_contract(self) -> ArmConfig:
        if self.arm is EvaluationArm.LEGACY:
            if self.legacy_commit != LEGACY_REPOPILOT_COMMIT:
                raise ValueError("legacy arm must use the frozen pre-upgrade commit")
        elif self.legacy_commit is not None:
            raise ValueError("legacy_commit is valid only for the legacy arm")
        expected_retrieval = self.arm is EvaluationArm.UPGRADED_WITH_RETRIEVAL
        if self.retrieval_enabled != expected_retrieval:
            raise ValueError(
                f"retrieval setting does not match arm {self.arm.value}"
            )
        return self


def build_arm_configs(*, model: str) -> dict[EvaluationArm, ArmConfig]:
    configs = [
        ArmConfig(arm=EvaluationArm.NATIVE, model=model),
        ArmConfig(
            arm=EvaluationArm.LEGACY,
            model=model,
            legacy_commit=LEGACY_REPOPILOT_COMMIT,
        ),
        ArmConfig(
            arm=EvaluationArm.UPGRADED_NO_RETRIEVAL,
            model=model,
        ),
        ArmConfig(
            arm=EvaluationArm.UPGRADED_WITH_RETRIEVAL,
            model=model,
            retrieval_enabled=True,
        ),
    ]
    return {config.arm: config for config in configs}


class RunnerOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "failed", "timeout", "cancelled"]
    run_id: str
    model_patch: str = ""
    duration_seconds: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_hit_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    retrieval: tuple[RetrievedLocation, ...] = ()
    error: str | None = None


class ArmRunner(Protocol):
    async def run(
        self,
        *,
        instance: PublicInstance,
        repository_path: Path,
        config: ArmConfig,
        budget: InferenceBudget,
    ) -> RunnerOutcome: ...


class AgentAdapter:
    def __init__(self, *, config: ArmConfig, runner: ArmRunner):
        self.config = config
        self.runner = runner

    async def run(
        self,
        *,
        instance: PublicInstance,
        repository_path: Path,
        budget: InferenceBudget,
    ) -> RunnerOutcome:
        return await self.runner.run(
            instance=instance,
            repository_path=repository_path,
            config=self.config,
            budget=budget,
        )

