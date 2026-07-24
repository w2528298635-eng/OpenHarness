from __future__ import annotations

import json

from .models import AnalysisResult, Phase, RepairPlan, RepoRunState


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


def build_phase_prompt(phase: Phase, state: RepoRunState, *, diff_summary: str = "") -> str:
    context = {
        "问题": state.task.issue,
        "允许路径": state.task.allowed_paths,
        "已有分析": state.analysis.model_dump(mode="json") if state.analysis else None,
        "已有计划": state.plan.model_dump(mode="json") if state.plan else None,
        "最近验证": (
            state.verification_history[-1].model_dump(mode="json")
            if state.verification_history
            else None
        ),
        "当前差异摘要": diff_summary,
        "剩余预算": _remaining(state),
    }
    common = (
        "你正在 RepoPilot 的一个受限阶段中。严格服从阶段边界。"
        "只有验证器可以宣布修复成功；你不得自行声称测试已经通过。\n"
        f"上下文：{json.dumps(context, ensure_ascii=False)}\n"
    )
    if phase is Phase.ANALYZE:
        return (
            common + "目标：定位根因并引用真实代码证据。不得修改任何文件。"
            "最终只输出符合 AnalysisResult JSON Schema 的 JSON："
            + json.dumps(AnalysisResult.model_json_schema(), ensure_ascii=False)
        )
    if phase in {Phase.PLAN, Phase.REPLAN}:
        return (
            common + "目标：制定最小、文件范围明确的修复计划。不得修改文件。"
            "最终只输出符合 RepairPlan JSON Schema 的 JSON："
            + json.dumps(RepairPlan.model_json_schema(), ensure_ascii=False)
        )
    if phase in {Phase.EXECUTE, Phase.REPAIR}:
        return (
            common + "目标：按计划编辑代码，仅处理验证器给出的证据。"
            "不得修改验证命令，不得使用 shell，不得越过允许路径。"
            "完成编辑后简短说明实际修改；不要输出虚构测试结果。"
        )
    raise ValueError(f"phase does not use a model prompt: {phase}")
