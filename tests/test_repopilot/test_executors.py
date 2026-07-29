from pathlib import Path

import pytest

from openharness.repopilot.executors import (
    OpenHarnessPhaseExecutor,
    ScriptedPhaseExecutor,
)
from openharness.repopilot.models import Phase, PhaseRunResult, RepoRunState, RepoTaskSpec


def _state(tmp_path: Path) -> RepoRunState:
    return RepoRunState(
        run_id="run",
        task=RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"]),
    )


@pytest.mark.asyncio
async def test_scripted_executor_consumes_expected_phase_without_provider(
    tmp_path: Path,
) -> None:
    executor = ScriptedPhaseExecutor(
        {
            Phase.ANALYZE: [
                PhaseRunResult(
                    phase=Phase.ANALYZE,
                    structured={
                        "suspected_files": ["app.py"],
                        "root_cause": "boundary",
                        "evidence": [{"file": "app.py", "observation": "wrong"}],
                        "confidence": 0.9,
                    },
                )
            ]
        }
    )

    result = await executor.run(
        Phase.ANALYZE,
        _state(tmp_path),
        tmp_path,
        prompt_version="2",
        context_ids=["chunk-1"],
        tool_names=["read_file"],
    )

    assert result.phase is Phase.ANALYZE
    assert executor.calls[0].context_ids == ("chunk-1",)
    with pytest.raises(RuntimeError, match="no scripted result"):
        await executor.run(Phase.ANALYZE, _state(tmp_path), tmp_path)


@pytest.mark.asyncio
async def test_openharness_executor_delegates_to_runner(tmp_path: Path) -> None:
    class Runner:
        async def run(self, phase, state, cwd, **kwargs):
            self.arguments = (phase, state, cwd, kwargs)
            return PhaseRunResult(phase=phase, final_text="done")

    runner = Runner()
    executor = OpenHarnessPhaseExecutor(runner)
    result = await executor.run(
        Phase.EXECUTE,
        _state(tmp_path),
        tmp_path,
        retrieved_context="evidence",
    )

    assert result.final_text == "done"
    assert runner.arguments[3]["retrieved_context"] == "evidence"
