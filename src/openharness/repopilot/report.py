from __future__ import annotations

from pathlib import Path

from .models import RepoRunState


def render_report(state: RepoRunState, run_dir: Path) -> str:
    latest = state.verification_history[-1] if state.verification_history else None
    outcome = state.terminal_reason or state.phase.value
    changed = "\n".join(f"- `{path}`" for path in state.changed_files) or "- 无"
    verification = (
        f"{latest.category}（exit_code={latest.exit_code}，"
        f"耗时={latest.duration_seconds:.2f}s）"
        if latest
        else "尚未执行"
    )
    return f"""# RepoPilot 运行报告：{state.run_id}

## 结果

- 状态：`{state.phase.value}`
- 原因：`{outcome}`
- 问题：{state.task.issue}
- 工作树：`{state.worktree_path or "未创建"}`
- 最近验证：{verification}

## 修改文件

{changed}

## 预算使用

- 模型阶段调用：{state.budgets.phase_calls}/{state.task.budgets.max_phase_calls}
- Repair：{state.budgets.repair_attempts}/{state.task.budgets.max_repair_attempts}
- Replan：{state.budgets.replan_attempts}/{state.task.budgets.max_replan_attempts}
- Token：{state.budgets.total_tokens if state.budgets.total_tokens is not None else "unavailable"}

## 可检查产物

- `{run_dir / "state.json"}`
- `{run_dir / "events.jsonl"}`
- `{run_dir / "diff.patch"}`
- `{run_dir / "verification-<attempt>.json"}`
- `{run_dir / "verification-<attempt>.log"}`
"""
