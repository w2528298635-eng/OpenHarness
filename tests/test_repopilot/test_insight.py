from pathlib import Path

import pytest

from openharness.repopilot.insight import InsightRequest, RepositoryInsightWorkflow
from openharness.repopilot.store import RunStore


@pytest.mark.asyncio
async def test_insight_workflow_is_read_only_and_citations_resolve(tmp_path: Path) -> None:
    (tmp_path / "engine.py").write_text(
        "class Scheduler:\n    def transition(self):\n        return 'next'\n",
        encoding="utf-8",
    )
    store = RunStore(tmp_path)
    workflow = RepositoryInsightWorkflow(store=store)

    report = await workflow.run(
        InsightRequest(
            repo_path=tmp_path,
            question="How does Scheduler transition?",
        )
    )

    assert report.findings
    assert report.prompt_version == "insight-1"
    assert report.context_ids
    for finding in report.findings:
        for citation in finding.citations:
            assert (tmp_path / citation.path).is_file()
    assert (tmp_path / "engine.py").read_text(encoding="utf-8").startswith("class Scheduler")
    state = store.load_state(report.run_id)
    assert state.phase.value == "COMPLETE"
    phases = [event.phase for event in store.load_events(report.run_id) if event.phase]
    assert phases[:2] == ["INSIGHT_SCAN", "INSIGHT_SCAN"]
