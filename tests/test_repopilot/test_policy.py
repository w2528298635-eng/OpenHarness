from datetime import timedelta
from pathlib import Path

import pytest

from openharness.repopilot.models import (
    Phase,
    RepoRunState,
    RepoTaskSpec,
    VerificationResult,
    utc_now,
)
from openharness.repopilot.policy import BudgetController, TransitionPolicy


def _state(tmp_path: Path) -> RepoRunState:
    return RepoRunState(
        run_id="run",
        task=RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"]),
    )


def _verification(category: str, passed: bool = False) -> VerificationResult:
    return VerificationResult(
        attempt=1, command=["pytest"], passed=passed, exit_code=0 if passed else 1, category=category
    )


def test_only_passing_verification_completes(tmp_path: Path) -> None:
    policy = TransitionPolicy()

    assert policy.after_verify(_state(tmp_path), _verification("passed", True)).next_phase is Phase.COMPLETE
    assert policy.after_execute(_state(tmp_path), has_diff=True).next_phase is Phase.VERIFY


@pytest.mark.parametrize("category", ["missing_executable", "infrastructure_error", "timeout"])
def test_verifier_environment_failures_are_terminal(tmp_path: Path, category: str) -> None:
    decision = TransitionPolicy().after_verify(_state(tmp_path), _verification(category))

    assert decision.next_phase is Phase.FAILED
    assert decision.terminal_reason == f"verification_{category}"


def test_repeated_failure_replans_and_empty_diff_replans(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.failure_signatures = ["same", "same"]

    assert TransitionPolicy().after_verify(
        state,
        VerificationResult(
            attempt=2,
            command=["pytest"],
            passed=False,
            exit_code=1,
            category="test_failure",
            failure_signature="same",
        ),
    ).next_phase is Phase.REPLAN
    assert TransitionPolicy().after_execute(state, has_diff=False).next_phase is Phase.REPLAN


@pytest.mark.parametrize(
    ("mutate", "next_phase", "reason"),
    [
        (lambda state: setattr(state.budgets, "phase_calls", state.task.budgets.max_phase_calls), Phase.ANALYZE, "phase_call_budget_exhausted"),
        (lambda state: setattr(state.budgets, "repair_attempts", state.task.budgets.max_repair_attempts), Phase.REPAIR, "repair_budget_exhausted"),
        (lambda state: setattr(state.budgets, "replan_attempts", state.task.budgets.max_replan_attempts), Phase.REPLAN, "replan_budget_exhausted"),
        (lambda state: setattr(state, "started_at", utc_now() - timedelta(hours=1)), Phase.VERIFY, "wall_clock_budget_exhausted"),
    ],
)
def test_budget_exhaustion_is_explicit(
    tmp_path: Path, mutate, next_phase: Phase, reason: str
) -> None:
    state = _state(tmp_path)
    if reason == "wall_clock_budget_exhausted":
        state.task.budgets.max_wall_seconds = 1
    mutate(state)

    assert BudgetController().check(state, next_phase=next_phase).terminal_reason == reason
