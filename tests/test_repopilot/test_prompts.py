from pathlib import Path

from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.prompts import build_phase_prompt, prompt_version_for_phase


def _state(tmp_path: Path) -> RepoRunState:
    return RepoRunState(
        run_id="run",
        task=RepoTaskSpec(repo_path=tmp_path, issue="off by one", verify_command=["pytest"]),
    )


def test_analyze_prompt_is_read_only_and_requests_schema(tmp_path: Path) -> None:
    prompt = build_phase_prompt(Phase.ANALYZE, _state(tmp_path))

    assert "off by one" in prompt
    assert "must not modify" in prompt
    assert "AnalysisResult" in prompt
    assert prompt_version_for_phase(Phase.ANALYZE) == "2"


def test_execute_prompt_cannot_claim_success(tmp_path: Path) -> None:
    prompt = build_phase_prompt(Phase.EXECUTE, _state(tmp_path), diff_summary="none")

    assert "Only the verifier" in prompt
    assert "remaining_budget" in prompt
