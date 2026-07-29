from __future__ import annotations

import json

from .models import AnalysisResult, Phase, RepairPlan, RepoRunState
from .prompt_registry import PromptRegistry, PromptTemplate

PROMPT_VERSION = "2"

_COMMON = """You are operating inside one bounded phase of RepoPilot.
Follow the phase boundary exactly. Only the verifier may declare a repair successful;
never claim that tests passed unless a verifier result is present.

Runtime context:
{context}

"""

_REGISTRY = PromptRegistry()
_REGISTRY.register(
    PromptTemplate(
        name="analyze",
        version=PROMPT_VERSION,
        template=_COMMON
        + """Goal: locate the root cause and cite evidence from real repository files.
This phase is read-only: you must not modify any file.
Return only JSON matching the AnalysisResult schema:
{schema}""",
        required_variables=frozenset({"context", "schema"}),
    )
)
for name in ("plan", "replan"):
    _REGISTRY.register(
        PromptTemplate(
            name=name,
            version=PROMPT_VERSION,
            template=_COMMON
            + """Goal: produce the smallest repair plan with an explicit file scope.
This phase is read-only: you must not modify any file.
Return only JSON matching the RepairPlan schema:
{schema}""",
            required_variables=frozenset({"context", "schema"}),
        )
    )
for name in ("execute", "repair"):
    _REGISTRY.register(
        PromptTemplate(
            name=name,
            version=PROMPT_VERSION,
            template=_COMMON
            + """Goal: edit code according to the validated plan and current evidence.
Do not change the verification command, do not use a shell, and do not leave the
allowed paths. After editing, briefly state the actual change without inventing test
results.""",
            required_variables=frozenset({"context"}),
        )
    )

_PHASE_NAMES = {
    Phase.ANALYZE: "analyze",
    Phase.PLAN: "plan",
    Phase.REPLAN: "replan",
    Phase.EXECUTE: "execute",
    Phase.REPAIR: "repair",
}


def _remaining(state: RepoRunState) -> dict[str, int | None]:
    limits = state.task.budgets
    usage = state.budgets
    return {
        "phase_calls": limits.max_phase_calls - usage.phase_calls,
        "repair_attempts": limits.max_repair_attempts - usage.repair_attempts,
        "replan_attempts": limits.max_replan_attempts - usage.replan_attempts,
        "tokens": (
            None
            if limits.max_total_tokens is None or usage.total_tokens is None
            else limits.max_total_tokens - usage.total_tokens
        ),
    }


def prompt_version_for_phase(phase: Phase) -> str:
    if phase not in _PHASE_NAMES:
        raise ValueError(f"phase does not use a model prompt: {phase}")
    return PROMPT_VERSION


def build_phase_prompt(
    phase: Phase,
    state: RepoRunState,
    *,
    diff_summary: str = "",
    retrieved_context: str = "",
) -> str:
    try:
        name = _PHASE_NAMES[phase]
    except KeyError as exc:
        raise ValueError(f"phase does not use a model prompt: {phase}") from exc
    context = {
        "issue": state.task.issue,
        "allowed_paths": state.task.allowed_paths,
        "analysis": state.analysis.model_dump(mode="json") if state.analysis else None,
        "plan": state.plan.model_dump(mode="json") if state.plan else None,
        "latest_verification": (
            state.verification_history[-1].model_dump(mode="json")
            if state.verification_history
            else None
        ),
        "diff_summary": diff_summary,
        "retrieved_repository_context": retrieved_context or None,
        "remaining_budget": _remaining(state),
    }
    variables = {
        "context": json.dumps(context, ensure_ascii=False, sort_keys=True),
    }
    if phase is Phase.ANALYZE:
        variables["schema"] = json.dumps(
            AnalysisResult.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
    elif phase in {Phase.PLAN, Phase.REPLAN}:
        variables["schema"] = json.dumps(
            RepairPlan.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
    return _REGISTRY.render(name, version=PROMPT_VERSION, **variables)
