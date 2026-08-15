from pathlib import Path

from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.report import render_report


def test_report_contains_outcome_budget_and_artifact_locations(tmp_path: Path) -> None:
    state = RepoRunState(
        run_id="run-1",
        task=RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"]),
        phase=Phase.FAILED,
        terminal_reason="bug_not_reproduced",
    )

    report = render_report(state, tmp_path / "run-1")

    assert "bug_not_reproduced" in report
    assert "预算使用" in report
    assert "events.jsonl" in report
