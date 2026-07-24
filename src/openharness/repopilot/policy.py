from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from .models import Phase, RepoRunState, TransitionDecision, VerificationResult


class TransitionPolicy:
    def after_precheck(self, result: VerificationResult) -> TransitionDecision:
        if result.passed:
            return TransitionDecision(next_phase=Phase.FAILED, terminal_reason="bug_not_reproduced")
        if result.category == "test_failure":
            return TransitionDecision(next_phase=Phase.ANALYZE)
        return TransitionDecision(
            next_phase=Phase.FAILED,
            terminal_reason="invalid_verification_environment",
            detail=result.category,
        )

    def after_execute(
        self, state: RepoRunState, *, has_diff: bool, policy_violation: str | None = None
    ) -> TransitionDecision:
        del state
        if policy_violation:
            return TransitionDecision(
                next_phase=Phase.FAILED,
                terminal_reason="policy_violation",
                detail=policy_violation,
            )
        return TransitionDecision(next_phase=Phase.VERIFY if has_diff else Phase.REPLAN)

    def after_verify(self, state: RepoRunState, result: VerificationResult) -> TransitionDecision:
        if result.passed:
            return TransitionDecision(next_phase=Phase.COMPLETE)
        if result.category in {"missing_executable", "infrastructure_error", "timeout"}:
            return TransitionDecision(
                next_phase=Phase.FAILED,
                terminal_reason=f"verification_{result.category}",
            )
        signatures = state.failure_signatures + (
            [result.failure_signature] if result.failure_signature else []
        )
        if result.failure_signature and len(signatures) >= 2 and signatures[-1] == signatures[-2]:
            return TransitionDecision(next_phase=Phase.REPLAN)
        return TransitionDecision(next_phase=Phase.REPAIR)

    def after_repair(self, state: RepoRunState, *, diff_changed: bool) -> TransitionDecision:
        del state
        return TransitionDecision(next_phase=Phase.VERIFY if diff_changed else Phase.REPLAN)

    def after_replan(self, *, valid: bool) -> TransitionDecision:
        return TransitionDecision(
            next_phase=Phase.EXECUTE if valid else Phase.FAILED,
            terminal_reason=None if valid else "invalid_plan",
        )


class BudgetController:
    _MODEL_PHASES: ClassVar[set[Phase]] = {
        Phase.ANALYZE,
        Phase.PLAN,
        Phase.EXECUTE,
        Phase.REPAIR,
        Phase.REPLAN,
    }

    def check(
        self,
        state: RepoRunState,
        *,
        next_phase: Phase,
        changed_files: list[str] | None = None,
        now: datetime | None = None,
    ) -> TransitionDecision:
        limits = state.task.budgets
        usage = state.budgets
        current = now or datetime.now(UTC)
        elapsed = (current - state.started_at).total_seconds()
        reason: str | None = None
        if elapsed >= limits.max_wall_seconds:
            reason = "wall_clock_budget_exhausted"
        elif next_phase in self._MODEL_PHASES and usage.phase_calls >= limits.max_phase_calls:
            reason = "phase_call_budget_exhausted"
        elif next_phase is Phase.REPAIR and usage.repair_attempts >= limits.max_repair_attempts:
            reason = "repair_budget_exhausted"
        elif next_phase is Phase.REPLAN and usage.replan_attempts >= limits.max_replan_attempts:
            reason = "replan_budget_exhausted"
        elif changed_files is not None and len(changed_files) > limits.max_changed_files:
            reason = "changed_file_budget_exhausted"
        elif limits.max_total_tokens is not None:
            if usage.total_tokens is None:
                reason = "usage_unavailable"
            elif usage.total_tokens >= limits.max_total_tokens:
                reason = "token_budget_exhausted"
        if reason:
            return TransitionDecision(next_phase=Phase.FAILED, terminal_reason=reason)
        return TransitionDecision(next_phase=next_phase)
