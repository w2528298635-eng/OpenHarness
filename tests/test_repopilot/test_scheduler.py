from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.repopilot.models import (
    Phase,
    PhaseRunResult,
    RepoTaskSpec,
    VerificationResult,
)
from openharness.repopilot.scheduler import RepoPilotScheduler
from openharness.repopilot.store import RunStore


ANALYSIS = {
    "suspected_files": ["app.py"],
    "root_cause": "off by one",
    "evidence": [{"file": "app.py", "observation": "bad boundary"}],
    "confidence": 0.9,
}
PLAN = {
    "hypothesis": "fix boundary",
    "steps": [
        {
            "id": "1",
            "description": "change comparison",
            "target_files": ["app.py"],
            "expected_behavior": "boundary works",
        }
    ],
    "expected_files": ["app.py"],
    "expected_behavior": "tests pass",
}


class FakeWorkspace:
    async def create(self, repo, run_id):
        worktree = repo.parent / f"worktree-{run_id}"
        worktree.mkdir()
        return SimpleNamespace(path=worktree, branch=f"repopilot/{run_id}")

    async def diff(self, worktree):
        return "diff --git a/app.py b/app.py\n+fixed"

    async def changed_files(self, worktree):
        return ["app.py"]

    def validate_changed_files(self, paths, allowed):
        return None

    def diff_signature(self, diff):
        return "diff-signature"


class FakeVerifier:
    def __init__(self, results):
        self.results = list(results)

    async def verify(self, argv, cwd, *, attempt):
        result = self.results.pop(0)
        result.attempt = attempt
        return result


class FakeRunner:
    def __init__(self):
        self.phases = []

    async def run(self, phase, state, cwd, *, diff_summary=""):
        self.phases.append(phase)
        structured = ANALYSIS if phase is Phase.ANALYZE else PLAN if phase in {Phase.PLAN, Phase.REPLAN} else None
        return PhaseRunResult(phase=phase, structured=structured, tokens_used=5)


def verification(passed: bool, signature: str | None = None):
    return VerificationResult(
        attempt=0,
        command=["pytest"],
        passed=passed,
        exit_code=0 if passed else 1,
        category="passed" if passed else "test_failure",
        failure_signature=signature,
    )


@pytest.mark.asyncio
async def test_successful_run_can_only_complete_after_verification(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo)
    runner = FakeRunner()
    scheduler = RepoPilotScheduler(
        store=store,
        workspace=FakeWorkspace(),
        verifier=FakeVerifier([verification(False, "baseline"), verification(True)]),
        phase_runner=runner,
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )

    assert state.phase is Phase.COMPLETE
    assert runner.phases == [Phase.ANALYZE, Phase.PLAN, Phase.EXECUTE]
    assert len(state.verification_history) == 2
    assert (store.run_dir(state.run_id) / "report.md").exists()
    assert store.load_state(state.run_id).phase is Phase.COMPLETE


@pytest.mark.asyncio
async def test_bug_not_reproduced_stops_before_model_call(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = FakeRunner()
    scheduler = RepoPilotScheduler(
        store=RunStore(repo),
        workspace=FakeWorkspace(),
        verifier=FakeVerifier([verification(True)]),
        phase_runner=runner,
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )

    assert state.phase is Phase.FAILED
    assert state.terminal_reason == "bug_not_reproduced"
    assert runner.phases == []


@pytest.mark.asyncio
async def test_failed_verification_enters_repair_then_verifies_again(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = FakeRunner()
    scheduler = RepoPilotScheduler(
        store=RunStore(repo),
        workspace=FakeWorkspace(),
        verifier=FakeVerifier(
            [verification(False, "baseline"), verification(False, "new"), verification(True)]
        ),
        phase_runner=runner,
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )

    assert state.phase is Phase.COMPLETE
    assert Phase.REPAIR in runner.phases
    assert state.budgets.repair_attempts == 1
