import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.repopilot.models import (
    Phase,
    PhaseRunResult,
    RepoTaskSpec,
    TokenUsage,
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
        (worktree / "app.py").write_text("broken = True\n", encoding="utf-8")
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
        structured = (
            ANALYSIS
            if phase is Phase.ANALYZE
            else PLAN
            if phase in {Phase.PLAN, Phase.REPLAN}
            else None
        )
        return PhaseRunResult(phase=phase, structured=structured, tokens_used=5)


class ContextCapturingRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def run(self, phase, state, cwd, *, diff_summary="", retrieved_context=""):
        self.contexts.append((phase, retrieved_context))
        return await super().run(
            phase,
            state,
            cwd,
            diff_summary=diff_summary,
        )


class DetailedUsageRunner(FakeRunner):
    async def run(self, phase, state, cwd, *, diff_summary=""):
        result = await super().run(phase, state, cwd, diff_summary=diff_summary)
        result.token_usage = TokenUsage(input_tokens=7, output_tokens=3)
        result.tokens_used = 10
        return result


def verification(passed: bool, signature: str | None = None):
    return VerificationResult(
        attempt=0,
        command=["pytest"],
        passed=passed,
        exit_code=0 if passed else 1,
        category="passed" if passed else "test_failure",
        failure_signature=signature,
    )


def timeout_verification():
    return VerificationResult(
        attempt=0,
        command=["pytest"],
        passed=False,
        category="timeout",
        failure_signature="timeout",
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
        RepoTaskSpec(
            repo_path=repo,
            issue="broken",
            verify_command=["pytest"],
            budgets={"max_repeated_diffs": 10},
        )
    )

    assert state.phase is Phase.COMPLETE
    assert runner.phases == [Phase.ANALYZE, Phase.PLAN, Phase.EXECUTE]
    assert len(state.verification_history) == 2
    assert (store.run_dir(state.run_id) / "report.md").exists()
    assert store.load_state(state.run_id).phase is Phase.COMPLETE
    events = [
        json.loads(line)
        for line in (store.run_dir(state.run_id) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        event.get("schema_version") == 1 and event.get("kind") == "phase_started"
        for event in events
    )


@pytest.mark.asyncio
async def test_retrieval_writes_trace_and_supplies_analyze_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        "def calculate_discount(total):\n    return total * 0.9\n",
        encoding="utf-8",
    )
    store = RunStore(repo)
    runner = ContextCapturingRunner()
    scheduler = RepoPilotScheduler(
        store=store,
        workspace=FakeWorkspace(),
        verifier=FakeVerifier([verification(False, "baseline"), verification(True)]),
        phase_runner=runner,
    )

    state = await scheduler.start(
        RepoTaskSpec(
            repo_path=repo,
            issue="broken",
            verify_command=["pytest"],
            retrieval={"enabled": True, "context_char_budget": 2000},
        )
    )

    analyze_context = next(context for phase, context in runner.contexts if phase is Phase.ANALYZE)
    assert "broken = True" in analyze_context
    traces = list(store.run_dir(state.run_id).glob("context-analyze-*.json"))
    assert len(traces) == 1


@pytest.mark.asyncio
async def test_scheduler_persists_detailed_model_usage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo)
    scheduler = RepoPilotScheduler(
        store=store,
        workspace=FakeWorkspace(),
        verifier=FakeVerifier([verification(False, "baseline"), verification(True)]),
        phase_runner=DetailedUsageRunner(),
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )
    summary = json.loads((store.run_dir(state.run_id) / "summary.json").read_text(encoding="utf-8"))

    assert state.budgets.input_tokens == 21
    assert state.budgets.output_tokens == 9
    assert state.budgets.total_tokens == 30
    assert summary["token_usage"]["input_tokens"] == 21


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
    assert (scheduler.store.run_dir(state.run_id) / "diff.patch").exists()


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
        RepoTaskSpec(
            repo_path=repo,
            issue="broken",
            verify_command=["pytest"],
            budgets={"max_repeated_diffs": 10},
        )
    )

    assert state.phase is Phase.COMPLETE
    assert Phase.REPAIR in runner.phases
    assert state.budgets.repair_attempts == 1


@pytest.mark.asyncio
async def test_precheck_timeout_is_retried_once(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = FakeRunner()
    scheduler = RepoPilotScheduler(
        store=RunStore(repo),
        workspace=FakeWorkspace(),
        verifier=FakeVerifier(
            [timeout_verification(), verification(False, "baseline"), verification(True)]
        ),
        phase_runner=runner,
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )

    assert state.phase is Phase.COMPLETE
    assert len(state.verification_history) == 3


class UnchangedRepairWorkspace(FakeWorkspace):
    def __init__(self):
        self.diffs = iter(
            [
                "diff --git a/app.py b/app.py\n+first",
                "diff --git a/app.py b/app.py\n+first",
                "diff --git a/app.py b/app.py\n+first",
                "diff --git a/app.py b/app.py\n+second",
            ]
        )

    async def diff(self, worktree):
        return next(self.diffs)

    def diff_signature(self, diff):
        return diff


@pytest.mark.asyncio
async def test_unchanged_repair_diff_replans_instead_of_reverifying(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = FakeRunner()
    scheduler = RepoPilotScheduler(
        store=RunStore(repo),
        workspace=UnchangedRepairWorkspace(),
        verifier=FakeVerifier(
            [verification(False, "baseline"), verification(False, "new"), verification(True)]
        ),
        phase_runner=runner,
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )

    assert state.phase is Phase.COMPLETE
    assert runner.phases == [
        Phase.ANALYZE,
        Phase.PLAN,
        Phase.EXECUTE,
        Phase.REPAIR,
        Phase.ANALYZE,
        Phase.REPLAN,
        Phase.EXECUTE,
    ]


class InterruptOnceRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.interrupted = False

    async def run(self, phase, state, cwd, *, diff_summary=""):
        if phase is Phase.ANALYZE and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return await super().run(phase, state, cwd, diff_summary=diff_summary)


@pytest.mark.asyncio
async def test_resume_reruns_incomplete_phase_without_duplicate_action(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = InterruptOnceRunner()
    store = RunStore(repo)
    scheduler = RepoPilotScheduler(
        store=store,
        workspace=FakeWorkspace(),
        verifier=FakeVerifier([verification(False, "baseline"), verification(True)]),
        phase_runner=runner,
    )

    with pytest.raises(KeyboardInterrupt):
        await scheduler.start(
            RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
        )
    run_id = next(store.root.iterdir()).name
    state = await scheduler.resume(run_id)
    events = [
        json.loads(line)
        for line in (store.run_dir(run_id) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert state.phase is Phase.COMPLETE
    analyze_actions = [
        event
        for event in events
        if event.get("kind") == "action" and event.get("phase") == Phase.ANALYZE.value
    ]
    assert len(analyze_actions) == 1


class InvalidEvidenceRunner(FakeRunner):
    async def run(self, phase, state, cwd, *, diff_summary=""):
        if phase is Phase.ANALYZE:
            return PhaseRunResult(
                phase=phase,
                structured={
                    **ANALYSIS,
                    "suspected_files": ["../secret.py"],
                    "evidence": [
                        {
                            "file": "../secret.py",
                            "observation": "outside worktree",
                        }
                    ],
                },
            )
        return await super().run(phase, state, cwd, diff_summary=diff_summary)


@pytest.mark.asyncio
async def test_analysis_evidence_must_exist_inside_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    scheduler = RepoPilotScheduler(
        store=RunStore(repo),
        workspace=FakeWorkspace(),
        verifier=FakeVerifier([verification(False, "baseline")]),
        phase_runner=InvalidEvidenceRunner(),
    )

    state = await scheduler.start(
        RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"])
    )

    assert state.phase is Phase.FAILED
    assert state.terminal_reason == "invalid_analysis"
