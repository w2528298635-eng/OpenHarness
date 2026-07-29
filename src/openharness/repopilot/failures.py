from __future__ import annotations

import hashlib
import re
from enum import Enum

from pydantic import BaseModel

from .models import Phase

_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s:]+(?:\\[^\s:]+)*")
_POSIX_PATH = re.compile(r"/(?:[^/\s:]+/)*[^/\s:]+")
_LINE = re.compile(r":\d+")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?s\b")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


class FailureCategory(str, Enum):
    PASSED = "passed"
    CONFIGURATION = "configuration"
    BASELINE_NOT_REPRODUCIBLE = "baseline_not_reproducible"
    TIMEOUT = "timeout"
    SYNTAX = "syntax"
    COLLECTION = "collection"
    ASSERTION = "assertion"
    DEPENDENCY = "dependency"
    PERMISSION = "permission"
    PATH = "path"
    NO_DIFF = "no_diff"
    OUT_OF_SCOPE_DIFF = "out_of_scope_diff"
    REPEATED_FAILURE = "repeated_failure"
    PROVIDER = "provider"
    STRUCTURED_OUTPUT = "structured_output"
    CANCELLATION = "cancellation"
    INTERNAL = "internal"


class FailureRecord(BaseModel):
    category: FailureCategory
    source_phase: Phase
    signature: str
    summary: str
    retryable: bool = False
    artifact_path: str | None = None


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPAIR = "repair"
    REPLAN = "replan"
    STOP = "stop"


class RecoveryDecision(BaseModel):
    action: RecoveryAction
    next_phase: Phase | None = None
    reason: str
    terminal_reason: str | None = None


def normalize_failure_signature(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    text = _WINDOWS_PATH.sub("<path>", text)
    text = _POSIX_PATH.sub("<path>", text)
    text = _LINE.sub(":<line>", text)
    text = _DURATION.sub("<time>", text)
    text = _ADDRESS.sub("<address>", text)
    text = " ".join(text.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_failure(
    *,
    stdout: str,
    stderr: str,
    exit_code: int | None,
    source_phase: Phase,
    artifact_path: str | None = None,
) -> FailureRecord:
    combined = f"{stdout}\n{stderr}".casefold()
    if exit_code == 0:
        category = FailureCategory.PASSED
    elif "timed out" in combined or combined.strip() == "timeout":
        category = FailureCategory.TIMEOUT
    elif "permissionerror" in combined or "access is denied" in combined:
        category = FailureCategory.PERMISSION
    elif "syntaxerror" in combined or "indentationerror" in combined:
        category = FailureCategory.SYNTAX
    elif "error collecting" in combined or "error during collection" in combined:
        category = FailureCategory.COLLECTION
    elif (
        "modulenotfounderror" in combined
        or "no module named" in combined
        or "importerror" in combined
    ):
        category = FailureCategory.DEPENDENCY
    elif "assertionerror" in combined or "failed " in combined:
        category = FailureCategory.ASSERTION
    else:
        category = FailureCategory.INTERNAL
    retryable = category in {
        FailureCategory.TIMEOUT,
        FailureCategory.PROVIDER,
        FailureCategory.STRUCTURED_OUTPUT,
    }
    summary_source = stderr.strip() or stdout.strip() or category.value
    return FailureRecord(
        category=category,
        source_phase=source_phase,
        signature=normalize_failure_signature(stdout, stderr),
        summary=summary_source[:1000],
        retryable=retryable,
        artifact_path=artifact_path,
    )


class FailurePolicy:
    def decide(
        self,
        failure: FailureRecord,
        *,
        repeated: bool,
        budget_exhausted_reason: str | None = None,
    ) -> RecoveryDecision:
        if budget_exhausted_reason:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=budget_exhausted_reason,
                terminal_reason=budget_exhausted_reason,
            )
        category = failure.category
        phase = failure.source_phase
        if category is FailureCategory.OUT_OF_SCOPE_DIFF:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=category.value,
                terminal_reason="policy_violation",
            )
        if category in {FailureCategory.PERMISSION, FailureCategory.PATH}:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=category.value,
                terminal_reason=f"unsafe_{category.value}",
            )
        if category is FailureCategory.NO_DIFF:
            return RecoveryDecision(
                action=RecoveryAction.REPLAN,
                next_phase=Phase.REPLAN,
                reason=category.value,
            )
        if repeated or category is FailureCategory.REPEATED_FAILURE:
            return RecoveryDecision(
                action=RecoveryAction.REPLAN,
                next_phase=Phase.REPLAN,
                reason="repeated_failure",
            )
        if category is FailureCategory.STRUCTURED_OUTPUT and phase in {
            Phase.ANALYZE,
            Phase.PLAN,
            Phase.REPLAN,
        }:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                next_phase=phase,
                reason=category.value,
            )
        if category in {FailureCategory.PROVIDER, FailureCategory.TIMEOUT}:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                next_phase=phase,
                reason=category.value,
            )
        if category is FailureCategory.SYNTAX and phase in {Phase.EXECUTE, Phase.REPAIR}:
            return RecoveryDecision(
                action=RecoveryAction.REPAIR,
                next_phase=Phase.REPAIR,
                reason=category.value,
            )
        if category is FailureCategory.ASSERTION and phase is Phase.VERIFY:
            return RecoveryDecision(
                action=RecoveryAction.REPAIR,
                next_phase=Phase.REPAIR,
                reason=category.value,
            )
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason=category.value,
            terminal_reason=f"unrecoverable_{category.value}",
        )
