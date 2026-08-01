from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class Phase(str, Enum):
    PRECHECK = "PRECHECK"
    ANALYZE = "ANALYZE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    REPAIR = "REPAIR"
    REPLAN = "REPLAN"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INSIGHT_SCAN = "INSIGHT_SCAN"
    INSIGHT_RETRIEVE = "INSIGHT_RETRIEVE"
    INSIGHT_ANALYZE = "INSIGHT_ANALYZE"
    INSIGHT_REPORT = "INSIGHT_REPORT"


class BudgetConfig(BaseModel):
    max_phase_calls: int = Field(default=8, ge=1)
    max_repair_attempts: int = Field(default=3, ge=0)
    max_replan_attempts: int = Field(default=2, ge=0)
    max_wall_seconds: float = Field(default=1800, gt=0)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_changed_files: int = Field(default=12, ge=1)
    max_repeated_actions: int = Field(default=3, ge=1)
    max_repeated_diffs: int = Field(default=2, ge=1)
    verify_timeout_seconds: float = Field(default=300, gt=0)


class BudgetUsage(BaseModel):
    phase_calls: int = 0
    repair_attempts: int = 0
    replan_attempts: int = 0
    total_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    repeated_actions: int = 0
    repeated_diffs: int = 0


class RetrievalConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["lexical", "hybrid"] = "lexical"
    query_planning: bool = True
    structural_expansion: bool = False
    max_file_bytes: int = Field(default=200_000, ge=1024)
    max_chunk_chars: int = Field(default=4000, ge=200)
    context_char_budget: int = Field(default=12_000, ge=500)
    top_k: int = Field(default=12, ge=1, le=100)


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_hit_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def derive_total(self) -> TokenUsage:
        if self.total_tokens is None:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_hit_tokens=self.cache_hit_tokens + other.cache_hit_tokens,
            total_tokens=(self.total_tokens or 0) + (other.total_tokens or 0),
        )


class RepoTaskSpec(BaseModel):
    repo_path: Path
    issue: str
    verify_command: list[str]
    allowed_paths: list[str] | None = None
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @field_validator("issue")
    @classmethod
    def validate_issue(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("issue must not be empty")
        return value

    @field_validator("verify_command")
    @classmethod
    def validate_command_not_empty(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("verify_command must be a non-empty argv")
        return value


class CodeEvidence(BaseModel):
    file: str
    symbol: str = ""
    line: int | None = Field(default=None, ge=1)
    observation: str


class AnalysisResult(BaseModel):
    suspected_files: list[str]
    root_cause: str
    evidence: list[CodeEvidence]
    affected_symbols: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class PlanStep(BaseModel):
    id: str
    description: str
    target_files: list[str]
    expected_behavior: str


class RepairPlan(BaseModel):
    hypothesis: str
    steps: list[PlanStep]
    expected_files: list[str]
    expected_behavior: str


class ActionRecord(BaseModel):
    action_id: str
    phase: Phase
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: Literal["model", "scheduler"]
    timestamp: datetime = Field(default_factory=utc_now)


class ObservationRecord(BaseModel):
    action_id: str
    status: Literal["success", "failure", "blocked"]
    summary: str
    raw_artifact_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    failure_signature: str | None = None


class VerificationResult(BaseModel):
    attempt: int
    command: list[str]
    passed: bool
    exit_code: int | None = None
    category: Literal[
        "passed",
        "test_failure",
        "collection_error",
        "missing_executable",
        "timeout",
        "infrastructure_error",
    ]
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0
    failure_signature: str | None = None


class TransitionDecision(BaseModel):
    next_phase: Phase
    terminal_reason: str | None = None
    detail: str = ""


class PhaseRunResult(BaseModel):
    phase: Phase
    structured: dict[str, Any] | None = None
    final_text: str = ""
    tokens_used: int | None = None
    token_usage: TokenUsage | None = None
    actions: list[ActionRecord] = Field(default_factory=list)
    observations: list[ObservationRecord] = Field(default_factory=list)


class RepoRunState(BaseModel):
    run_id: str
    task: RepoTaskSpec
    phase: Phase = Phase.PRECHECK
    original_repo: Path | None = None
    worktree_path: Path | None = None
    worktree_branch: str | None = None
    worktree_slug: str | None = None
    worktree_root: Path | None = None
    analysis: AnalysisResult | None = None
    plan: RepairPlan | None = None
    verification_history: list[VerificationResult] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    budgets: BudgetUsage = Field(default_factory=BudgetUsage)
    action_ids: list[str] = Field(default_factory=list)
    failure_signatures: list[str] = Field(default_factory=list)
    diff_signatures: list[str] = Field(default_factory=list)
    action_signatures: list[str] = Field(default_factory=list)
    terminal_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
