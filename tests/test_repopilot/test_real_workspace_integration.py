import subprocess
import sys
from pathlib import Path

import pytest

from openharness.repopilot.models import Phase, PhaseRunResult, RepoTaskSpec
from openharness.repopilot.scheduler import RepoPilotScheduler
from openharness.repopilot.store import RunStore
from openharness.repopilot.verifier import PythonPytestVerifier
from openharness.repopilot.workspace import WorkspaceManager


class EditingRunner:
    async def run(self, phase, state, cwd, *, diff_summary=""):
        if phase is Phase.ANALYZE:
            structured = {
                "suspected_files": ["discount.py"],
                "root_cause": "exclusive upper bound",
                "evidence": [
                    {
                        "file": "discount.py",
                        "observation": "rate 1 is rejected",
                    }
                ],
                "confidence": 1,
            }
        elif phase in {Phase.PLAN, Phase.REPLAN}:
            structured = {
                "hypothesis": "make upper bound inclusive",
                "steps": [
                    {
                        "id": "1",
                        "description": "fix boundary",
                        "target_files": ["discount.py"],
                        "expected_behavior": "rate 1 returns zero",
                    }
                ],
                "expected_files": ["discount.py"],
                "expected_behavior": "tests pass",
            }
        else:
            structured = None
            target = cwd / "discount.py"
            target.write_text(
                target.read_text(encoding="utf-8").replace("0 <= rate < 1", "0 <= rate <= 1"),
                encoding="utf-8",
            )
        return PhaseRunResult(phase=phase, structured=structured, tokens_used=1)


def _commit_baseline(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.mark.asyncio
async def test_real_worktree_edit_and_pytest_leave_original_untouched(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "discount.py").write_text(
        "def price(total, rate):\n"
        "    if not 0 <= rate < 1:\n"
        "        raise ValueError('rate')\n"
        "    return total * (1 - rate)\n",
        encoding="utf-8",
    )
    (repo / "test_discount.py").write_text(
        "from discount import price\n\ndef test_full_discount():\n    assert price(100, 1) == 0\n",
        encoding="utf-8",
    )
    _commit_baseline(repo)
    scheduler = RepoPilotScheduler(
        store=RunStore(repo),
        workspace=WorkspaceManager(base_path=Path(sys.executable).parents[2] / ".rp-test-wt"),
        verifier=PythonPytestVerifier(timeout_seconds=30),
        phase_runner=EditingRunner(),
    )

    state = await scheduler.start(
        RepoTaskSpec(
            repo_path=repo,
            issue="full discount is rejected",
            verify_command=[sys.executable, "-m", "pytest", "-q", "test_discount.py"],
            allowed_paths=["discount.py"],
        )
    )

    assert state.phase is Phase.COMPLETE
    assert "rate < 1" in (repo / "discount.py").read_text(encoding="utf-8")
    assert "rate <= 1" in (state.worktree_path / "discount.py").read_text(encoding="utf-8")
    assert (repo / ".openharness" / "repopilot" / "runs" / state.run_id / "report.md").exists()

    await scheduler.workspace.cleanup(
        repo,
        state.worktree_path,
        force=True,
    )
    assert not state.worktree_path.exists()
