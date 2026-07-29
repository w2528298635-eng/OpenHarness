from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .handlers import RepairPhaseHandler
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
from .usage import RunSummary, TokenUsage
from .workflow import WorkflowDefinition, WorkflowRuntime


class InvalidAnalysisError(ValueError):
    pass


class InvalidPlanError(ValueError):
    pass


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
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        info = await self.workspace.create(task.repo_path, run_id)
        state = RepoRunState(
            run_id=run_id,
            task=task,
            original_repo=task.repo_path.resolve(),
            worktree_path=Path(info.path).resolve(),
            worktree_branch=info.branch,
            worktree_slug=getattr(info, "slug", None),
            worktree_root=getattr(info, "base_dir", None),
        )
        self.store.create(state)
        self.store.write_text(state.run_id, "diff.patch", "")
        runtime = self._workflow_runtime(state)
        completed = await runtime.start(task)
        return self._finalize(completed)

    async def resume(self, run_id: str) -> RepoRunState:
        state = self.store.load_state(run_id)
        if state.phase in {Phase.COMPLETE, Phase.FAILED}:
            return state
        self._event(state, "run_resumed", {"phase": state.phase.value})
        completed = await self._workflow_runtime(state).resume(state)
        return self._finalize(completed)

    def _workflow_runtime(self, state: RepoRunState) -> WorkflowRuntime:
        phases = (
            Phase.PRECHECK,
            Phase.ANALYZE,
            Phase.PLAN,
            Phase.EXECUTE,
            Phase.VERIFY,
            Phase.REPAIR,
            Phase.REPLAN,
        )
        definition = WorkflowDefinition(
            name="repopilot-repair",
            version="2",
            initial_phase=Phase.PRECHECK,
            terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
            handlers={
                phase: RepairPhaseHandler(
                    phase=phase,
                    execute=self._execute_phase,
                    classify_exception=self._classify_phase_exception,
                )
                for phase in phases
            },
        )
        return WorkflowRuntime(
            definition=definition,
            create_state=lambda task: state,
            checkpoint=self.store.save_state,
            emit=self.store.append_event,
            budget_check=self._budget_reason,
        )

    def _budget_reason(self, state: RepoRunState) -> str | None:
        decision = self.budgets.check(
            state,
            next_phase=state.phase,
            changed_files=state.changed_files,
        )
        return decision.terminal_reason

    async def _execute_phase(self, state: RepoRunState, phase: Phase) -> TransitionDecision:
        if phase is Phase.PRECHECK:
            result = await self._verify_with_timeout_retry(state)
            return self.transitions.after_precheck(result)
        if phase is Phase.ANALYZE:
            output = await self._model_phase(state, Phase.ANALYZE)
            state.analysis = AnalysisResult.model_validate(output.structured)
            self._validate_analysis(state)
            self.store.write_json(state.run_id, "analysis.json", state.analysis)
            return TransitionDecision(next_phase=Phase.PLAN)
        if phase is Phase.PLAN:
            output = await self._model_phase(state, Phase.PLAN)
            state.plan = RepairPlan.model_validate(output.structured)
            self._validate_plan(state)
            self.store.write_json(state.run_id, "plan.json", state.plan)
            return TransitionDecision(next_phase=Phase.EXECUTE)
        if phase in {Phase.EXECUTE, Phase.REPAIR}:
            before_diff = (
                await self.workspace.diff(state.worktree_path) if phase is Phase.REPAIR else ""
            )
            if phase is Phase.REPAIR:
                state.budgets.repair_attempts += 1
            await self._model_phase(
                state,
                phase,
                diff_summary=self._latest_verification_summary(state),
            )
            diff = await self.workspace.diff(state.worktree_path)
            files = await self.workspace.changed_files(state.worktree_path)
            state.changed_files = files
            self.store.write_text(state.run_id, "diff.patch", diff)
            signature = self.workspace.diff_signature(diff)
            if state.diff_signatures and state.diff_signatures[-1] == signature:
                state.budgets.repeated_diffs += 1
            else:
                state.budgets.repeated_diffs = 0
            state.diff_signatures.append(signature)
            violation = self.workspace.validate_changed_files(files, state.task.allowed_paths)
            if phase is Phase.EXECUTE:
                return self.transitions.after_execute(
                    state,
                    has_diff=bool(diff.strip()),
                    policy_violation=violation,
                )
            if violation:
                return TransitionDecision(
                    next_phase=Phase.FAILED,
                    terminal_reason="policy_violation",
                    detail=violation,
                )
            return self.transitions.after_repair(
                state,
                diff_changed=(
                    bool(diff.strip())
                    and self.workspace.diff_signature(diff)
                    != self.workspace.diff_signature(before_diff)
                ),
            )
        if phase is Phase.VERIFY:
            result = await self._verify_with_timeout_retry(state)
            decision = self.transitions.after_verify(state, result)
            if result.failure_signature:
                state.failure_signatures.append(result.failure_signature)
            return decision
        if phase is Phase.REPLAN:
            state.budgets.replan_attempts += 1
            analysis_output = await self._model_phase(
                state,
                Phase.ANALYZE,
                diff_summary=self._latest_verification_summary(state),
            )
            state.analysis = AnalysisResult.model_validate(analysis_output.structured)
            self._validate_analysis(state)
            self.store.write_json(state.run_id, "analysis.json", state.analysis)
            output = await self._model_phase(
                state,
                Phase.REPLAN,
                diff_summary=self._latest_verification_summary(state),
            )
            state.plan = RepairPlan.model_validate(output.structured)
            self._validate_plan(state)
            self.store.write_json(state.run_id, "plan.json", state.plan)
            return self.transitions.after_replan(valid=True)
        raise RuntimeError(f"unsupported phase: {phase}")

    @staticmethod
    def _classify_phase_exception(phase: Phase, exc: Exception) -> TransitionDecision:
        if isinstance(exc, InvalidAnalysisError):
            terminal_reason = "invalid_analysis"
        elif isinstance(exc, InvalidPlanError):
            terminal_reason = "invalid_plan"
        else:
            terminal_reason = f"{phase.value.lower()}_failed"
        return TransitionDecision(
            next_phase=Phase.FAILED,
            terminal_reason=terminal_reason,
            detail=str(exc),
        )

    def _finalize(self, state: RepoRunState) -> RepoRunState:
        if state.completed_at is None:
            state.completed_at = datetime.now(UTC)
        state.updated_at = state.completed_at
        self.store.save_state(state)
        report = render_report(state, self.store.run_dir(state.run_id))
        self.store.write_text(state.run_id, "report.md", report)
        summary = RunSummary(
            run_id=state.run_id,
            workflow="repopilot-repair",
            workflow_version="2",
            phase=state.phase,
            terminal_reason=state.terminal_reason,
            started_at=state.started_at,
            completed_at=state.completed_at,
            model_calls=state.budgets.phase_calls,
            phase_calls=state.budgets.phase_calls,
            verification_checks=len(state.verification_history),
            repair_attempts=state.budgets.repair_attempts,
            replan_attempts=state.budgets.replan_attempts,
            token_usage=TokenUsage(total_tokens=state.budgets.total_tokens or 0),
            changed_files=state.changed_files,
            artifacts={
                "state": str(self.store.run_dir(state.run_id) / "state.json"),
                "events": str(self.store.run_dir(state.run_id) / "events.jsonl"),
                "diff": str(self.store.run_dir(state.run_id) / "diff.patch"),
                "report": str(self.store.run_dir(state.run_id) / "report.md"),
            },
            warnings=self.store.event_warnings,
        )
        self.store.write_json(state.run_id, "summary.json", summary)
        return state

    async def _model_phase(self, state: RepoRunState, phase: Phase, *, diff_summary: str = ""):
        action_id = f"{state.run_id}:{phase.value.lower()}:{state.budgets.phase_calls + 1}"
        self._record_action(
            state, action_id, phase, "phase_agent", {"cwd": str(state.worktree_path)}
        )
        state.budgets.phase_calls += 1
        output = await self.phase_runner.run(
            phase, state, state.worktree_path, diff_summary=diff_summary
        )
        if output.tokens_used is not None:
            state.budgets.total_tokens = (state.budgets.total_tokens or 0) + output.tokens_used
        for action in output.actions:
            signature = json.dumps(
                [action.action_type, action.parameters],
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            if state.action_signatures and state.action_signatures[-1] == signature:
                state.budgets.repeated_actions += 1
            else:
                state.budgets.repeated_actions = 0
            state.action_signatures.append(signature)
            self.store.append_event(
                {"run_id": state.run_id, "kind": "action", **action.model_dump(mode="json")}
            )
        for observation in output.observations:
            self.store.append_event(
                {
                    "run_id": state.run_id,
                    "kind": "observation",
                    **observation.model_dump(mode="json"),
                }
            )
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
        self.store.write_json(state.run_id, f"verification-{attempt}.json", result)
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

    async def _verify_with_timeout_retry(self, state: RepoRunState):
        result = await self._verify(state)
        if result.category == "timeout":
            result = await self._verify(state)
        return result

    def _transition(self, state: RepoRunState, decision: TransitionDecision) -> None:
        previous = state.phase
        state.phase = decision.next_phase
        if decision.terminal_reason:
            state.terminal_reason = decision.terminal_reason
        state.updated_at = datetime.now(UTC)
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
        self.store.save_state(state)

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
                "timestamp": datetime.now(UTC).isoformat(),
                **data,
            }
        )

    @staticmethod
    def _resolve_inside(worktree: Path, raw_path: str, *, must_exist: bool) -> Path:
        candidate = (worktree / raw_path).resolve()
        try:
            candidate.relative_to(worktree.resolve())
        except ValueError as exc:
            raise ValueError(f"path escapes worktree: {raw_path}") from exc
        if must_exist and not candidate.is_file():
            raise ValueError(f"evidence file does not exist: {raw_path}")
        if not must_exist and not candidate.exists() and not candidate.parent.is_dir():
            raise ValueError(f"planned file parent does not exist: {raw_path}")
        return candidate

    def _validate_analysis(self, state: RepoRunState) -> None:
        if state.analysis is None or state.worktree_path is None:
            raise InvalidAnalysisError("analysis or worktree is missing")
        paths = [
            *state.analysis.suspected_files,
            *(evidence.file for evidence in state.analysis.evidence),
        ]
        try:
            for path in paths:
                self._resolve_inside(state.worktree_path, path, must_exist=True)
        except ValueError as exc:
            raise InvalidAnalysisError(str(exc)) from exc

    def _validate_plan(self, state: RepoRunState) -> None:
        if state.plan is None or state.worktree_path is None:
            raise InvalidPlanError("plan or worktree is missing")
        paths = list(state.plan.expected_files)
        for step in state.plan.steps:
            paths.extend(step.target_files)
        violation = self.workspace.validate_changed_files(
            sorted(set(paths)), state.task.allowed_paths
        )
        if violation:
            raise InvalidPlanError(violation)
        try:
            for path in paths:
                self._resolve_inside(state.worktree_path, path, must_exist=False)
        except ValueError as exc:
            raise InvalidPlanError(str(exc)) from exc

    @staticmethod
    def _latest_verification_summary(state: RepoRunState) -> str:
        if not state.verification_history:
            return ""
        result = state.verification_history[-1]
        return f"{result.category}: {(result.stdout + result.stderr)[-2000:]}"
