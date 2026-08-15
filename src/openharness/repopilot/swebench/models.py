from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_GOLD_ONLY_FIELDS = frozenset(
    {
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "gold_files",
        "gold_symbols",
    }
)


class DifficultyStratum(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def from_source(cls, value: str) -> DifficultyStratum:
        normalized = " ".join(value.strip().casefold().replace("–", "-").split())
        aliases = {
            "<15 min fix": cls.EASY,
            "<15 minute fix": cls.EASY,
            "15 min - 1 hour": cls.MEDIUM,
            "15 min-1 hour": cls.MEDIUM,
            "15 minutes - 1 hour": cls.MEDIUM,
            "1-4 hours": cls.HARD,
            "1 - 4 hours": cls.HARD,
            ">4 hours": cls.HARD,
            "4 hours": cls.HARD,
            "> 4 hours": cls.HARD,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported SWE-bench difficulty: {value!r}") from exc


class PublicInstance(BaseModel):
    """Inference-safe task data; gold evaluation fields are forbidden."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    source_difficulty: str = Field(min_length=1)
    difficulty: DifficultyStratum

    @model_validator(mode="before")
    @classmethod
    def reject_gold_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            forbidden = sorted(_GOLD_ONLY_FIELDS.intersection(value))
            if forbidden:
                joined = ", ".join(forbidden)
                raise ValueError(f"gold-only fields are forbidden in inference data: {joined}")
        return value


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    easy: int = Field(default=10, ge=0)
    medium: int = Field(default=15, ge=0)
    hard: int = Field(default=20, ge=0)
    seed: int = 20260730
    dataset_name: str = "SWE-bench/SWE-bench_Verified"

    @model_validator(mode="after")
    def require_nonempty_sample(self) -> SamplingConfig:
        if self.easy + self.medium + self.hard == 0:
            raise ValueError("sampling configuration must request at least one instance")
        return self

    def requested(self, stratum: DifficultyStratum) -> int:
        return {
            DifficultyStratum.EASY: self.easy,
            DifficultyStratum.MEDIUM: self.medium,
            DifficultyStratum.HARD: self.hard,
        }[stratum]


class SampleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    dataset_name: str
    dataset_revision: str = Field(min_length=1)
    sampling: SamplingConfig
    instances: tuple[PublicInstance, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

