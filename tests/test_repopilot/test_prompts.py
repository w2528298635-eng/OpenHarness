from pathlib import Path

from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.prompts import build_phase_prompt


def _state(tmp_path: Path) -> RepoRunState:
    return RepoRunState(
        run_id="run",
        task=RepoTaskSpec(repo_path=tmp_path, issue="off by one", verify_command=["pytest"]),
    )


def test_analyze_prompt_is_read_only_and_requests_schema(tmp_path: Path) -> None:
    prompt = build_phase_prompt(Phase.ANALYZE, _state(tmp_path))

    assert "off by one" in prompt
    assert "不得修改" in prompt
    assert "AnalysisResult" in prompt


def test_execute_prompt_cannot_claim_success(tmp_path: Path) -> None:
    prompt = build_phase_prompt(Phase.EXECUTE, _state(tmp_path), diff_summary="none")

    assert "只有验证器" in prompt
    assert "剩余预算" in prompt
