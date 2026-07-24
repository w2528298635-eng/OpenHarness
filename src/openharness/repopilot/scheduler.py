from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import (
    ActionRecord,
    AnalysisResult,
    ObservationRecord,
    Phase,
    RepairPlan,
    RepoRunState,
    RepoTaskSpec,
    TransitionDecision,
)
from .phase_runner import PhaseAgentRunner
from .policy import BudgetController, TransitionPolicy
from .report import render_report
from .store import RunStore


class RepoPilotScheduler:
    def __init__(
        self,
        *,
        store: RunStore,
        workspace,
        verifier,
        phase_runner: PhaseAgentRunner,
        transition_policy: TransitionPolicy | None = None,
        budget_controller: BudgetController | None = None,
    ):
        self.store = store
        self.workspace = workspace
        self.verifier = verifier
        self.phase_runner = phase_runner
        self.transitions = transition_policy or TransitionPolicy()
        self.budgets = budget_controller or BudgetController()

    async def start(self, task: RepoTaskSpec) -> RepoRunState:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        info = await self.workspace.create(task.repo_path, run_id)
        state = RepoRunState(
            run_id=run_id,
            task=task,
            original_repo=task.repo_path.resolve(),
            worktree_path=Path(info.path).resolve(),
            worktree_branch=info.branch,
        )
        self.store.create(state)
        self._event(state, "run_started", {"worktree": str(state.worktree_path)})
        return await self._run(state)

    async def resume(self, run_id: str) -> RepoRunState:
        state = self.store.load_state(run_id)
        if state.phase in {Phase.COMPLETE, Phase.FAILED}:
            return state
        self._event(state, "run_resumed", {"phase": state.phase.value})
        return await self._run(state)

    async def _run(self, state: RepoRunState) -> RepoRunState:
        while state.phase not in {Phase.COMPLETE, Phase.FAILED}:
            budget = self.budgets.check(
                state, next_phase=state.phase, changed_files=state.changed_files
            )
            if budget.next_phase is Phase.FAILED:
                self._transition(state, budget)
                break
            try:
                if state.phase is Phase.PRECHECK:
                    result = await self._verify(state)
                    decision = self.transitions.after_precheck(result)
                elif state.phase is Phase.ANALYZE:
                    output = await self._model_phase(state, Phase.ANALYZE)
                    state.analysis = AnalysisResult.model_validate(output.structured)
                    self.store.write_json(state.run_id, "analysis.json", state.analysis)
                    decision = TransitionDecision(next_phase=Phase.PLAN)
                elif state.phase is Phase.PLAN:
                    output = await self._model_phase(state, Phase.PLAN)
                    state.plan = RepairPlan.model_validate(output.structured)
                    self.store.write_json(state.run_id, "plan.json", state.plan)
                    decision = TransitionDecision(next_phase=Phase.EXECUTE)
                elif state.phase in {Phase.EXECUTE, Phase.REPAIR}:
                    active_phase = state.phase
                    if active_phase is Phase.REPAIR:
                        state.budgets.repair_attempts += 1
                    await self._model_phase(
                        state,
                        active_phase,
                        diff_summary=self._latest_verification_summary(state),
                    )
                    diff = await self.workspace.diff(state.worktree_path)
                    files = await self.workspace.changed_files(state.worktree_path)
                    state.changed_files = files
                    self.store.write_text(state.run_id, "diff.patch", diff)
                    signature = self.workspace.diff_signature(diff)
                    state.diff_signatures.append(signature)
                    violation = self.workspace.validate_changed_files(
                        files, state.task.allowed_paths
                    )
                    if active_phase is Phase.EXECUTE:
                        decision = self.transitions.after_execute(
                            state, has_diff=bool(diff.strip()), policy_violation=violation
                        )
                    elif violation:
                        decision = TransitionDecision(
                            next_phase=Phase.FAILED,
                            terminal_reason="policy_violation",
                            detail=violation,
                        )
                    else:
                        decision = self.transitions.after_repair(
                            state, diff_changed=bool(diff.strip())
                        )
                elif state.phase is Phase.VERIFY:
                    result = await self._verify(state)
                    decision = self.transitions.after_verify(state, result)
                    if result.failure_signature:
                        state.failure_signatures.append(result.failure_signature)
                elif state.phase is Phase.REPLAN:
                    state.budgets.replan_attempts += 1
                    output = await self._model_phase(
                        state,
                        Phase.REPLAN,
                        diff_summary=self._latest_verification_summary(state),
                    )
                    state.plan = RepairPlan.model_validate(output.structured)
                    self.store.write_json(state.run_id, "plan.json", state.plan)
                    decision = self.transitions.after_replan(valid=True)
                else:
                    raise RuntimeError(f"unsupported phase: {state.phase}")
            except Exception as exc:
                decision = TransitionDecision(
                    next_phase=Phase.FAILED,
                    terminal_reason=f"{state.phase.value.lower()}_failed",
                    detail=str(exc),
                )
            self._transition(state, decision)

        state.completed_at = datetime.now(timezone.utc)
        state.updated_at = state.completed_at
        self.store.save_state(state)
        report = render_report(state, self.store.run_dir(state.run_id))
        self.store.write_text(state.run_id, "report.md", report)
        self._event(
            state,
            "run_finished",
            {"phase": state.phase.value, "reason": state.terminal_reason},
        )
        return state

    async def _model_phase(
        self, state: RepoRunState, phase: Phase, *, diff_summary: str = ""
    ):
        action_id = f"{state.run_id}:{phase.value.lower()}:{state.budgets.phase_calls + 1}"
        self._record_action(state, action_id, phase, "phase_agent", {"cwd": str(state.worktree_path)})
        state.budgets.phase_calls += 1
        output = await self.phase_runner.run(
            phase, state, state.worktree_path, diff_summary=diff_summary
        )
        if output.tokens_used is not None:
            state.budgets.total_tokens = (state.budgets.total_tokens or 0) + output.tokens_used
        for action in output.actions:
            self.store.append_event({"run_id": state.run_id, "kind": "action", **action.model_dump(mode="json")})
        for observation in output.observations:
            self.store.append_event({"run_id": state.run_id, "kind": "observation", **observation.model_dump(mode="json")})
        self._record_observation(state, action_id, "success", output.final_text[:1000])
        return output

    async def _verify(self, state: RepoRunState):
        attempt = len(state.verification_history) + 1
        action_id = f"{state.run_id}:verify:{attempt}"
        self._record_action(
            state,
            action_id,
            Phase.VERIFY if state.phase is Phase.VERIFY else Phase.PRECHECK,
            "pytest",
            {"argv": state.task.verify_command},
        )
        result = await self.verifier.verify(
            state.task.verify_command, state.worktree_path, attempt=attempt
        )
        state.verification_history.append(result)
        self.store.write_json(
            state.run_id, f"verification-{attempt}.json", result
        )
        self.store.write_text(
            state.run_id,
            f"verification-{attempt}.log",
            result.stdout + ("\n" if result.stdout and result.stderr else "") + result.stderr,
        )
        self._record_observation(
            state,
            action_id,
            "success" if result.passed else "failure",
            result.category,
            result.failure_signature,
        )
        return result

    def _transition(self, state: RepoRunState, decision: TransitionDecision) -> None:
        previous = state.phase
        state.phase = decision.next_phase
        if decision.terminal_reason:
            state.terminal_reason = decision.terminal_reason
        state.updated_at = datetime.now(timezone.utc)
        self._event(
            state,
            "transition",
            {
                "from": previous.value,
                "to": state.phase.value,
                "reason": decision.terminal_reason,
                "detail": decision.detail,
            },
        )
        self.store.save_state(state)

    def _record_action(
        self, state: RepoRunState, action_id: str, phase: Phase, kind: str, parameters: dict
    ) -> None:
        if action_id in state.action_ids:
            return
        state.action_ids.append(action_id)
        action = ActionRecord(
            action_id=action_id,
            phase=phase,
            action_type=kind,
            parameters=parameters,
            source="scheduler",
        )
        self.store.append_event(
            {"run_id": state.run_id, "kind": "action", **action.model_dump(mode="json")}
        )

    def _record_observation(
        self,
        state: RepoRunState,
        action_id: str,
        status: str,
        summary: str,
        failure_signature: str | None = None,
    ) -> None:
        observation = ObservationRecord(
            action_id=action_id,
            status=status,
            summary=summary,
            failure_signature=failure_signature,
        )
        self.store.append_event(
            {
                "run_id": state.run_id,
                "kind": "observation",
                **observation.model_dump(mode="json"),
            }
        )

    def _event(self, state: RepoRunState, kind: str, data: dict) -> None:
        self.store.append_event(
            {
                "run_id": state.run_id,
                "kind": kind,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **data,
            }
        )

    @staticmethod
    def _latest_verification_summary(state: RepoRunState) -> str:
        if not state.verification_history:
            return ""
        result = state.verification_history[-1]
        return f"{result.category}: {(result.stdout + result.stderr)[-2000:]}"
