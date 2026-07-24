from pathlib import Path

import pytest
from pydantic import ValidationError

from openharness.repopilot.models import AnalysisResult, Phase, RepoRunState, RepoTaskSpec


def test_task_requires_non_empty_issue_and_command(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RepoTaskSpec(repo_path=tmp_path, issue=" ", verify_command=["pytest"])
    with pytest.raises(ValidationError):
        RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=[])


def test_analysis_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult(suspected_files=[], root_cause="x", evidence=[], confidence=1.1)


def test_run_state_round_trips_json(tmp_path: Path) -> None:
    task = RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"])
    state = RepoRunState(run_id="run-1", task=task, phase=Phase.PRECHECK)

    restored = RepoRunState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.budgets.phase_calls == 0
