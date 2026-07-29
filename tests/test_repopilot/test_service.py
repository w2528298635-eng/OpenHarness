from pathlib import Path

import pytest

from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.service import RepoPilotService
from openharness.repopilot.store import RunStore


class Scheduler:
    def __init__(self, state):
        self.state = state
        self.cancelled = False

    async def start(self, spec):
        return self.state

    async def resume(self, run_id):
        return self.state

    def request_cancel(self):
        self.cancelled = True


@pytest.mark.asyncio
async def test_service_starts_reads_events_and_safe_artifacts(tmp_path: Path) -> None:
    spec = RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"])
    state = RepoRunState(run_id="r1", task=spec, phase=Phase.COMPLETE)
    store = RunStore(tmp_path)
    store.create(state)
    store.write_text("r1", "report.md", "report")
    scheduler = Scheduler(state)
    service = RepoPilotService(
        scheduler_factory=lambda *_: scheduler,
        store_factory=lambda _: store,
    )

    completed = await service.start(spec)

    assert completed.run_id == "r1"
    assert service.get("r1").phase is Phase.COMPLETE
    assert service.artifact("r1", "report.md").read_text() == "report"
    assert service.events("r1") == []
    with pytest.raises(ValueError, match="artifact name"):
        service.artifact("r1", "../state.json")


@pytest.mark.asyncio
async def test_service_requests_cooperative_cancellation(tmp_path: Path) -> None:
    spec = RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"])
    state = RepoRunState(run_id="r1", task=spec)
    scheduler = Scheduler(state)
    service = RepoPilotService(
        scheduler_factory=lambda *_: scheduler,
        store_factory=RunStore,
    )
    service._schedulers["r1"] = scheduler

    service.cancel("r1")

    assert scheduler.cancelled is True


@pytest.mark.asyncio
async def test_service_submit_tracks_background_operation(tmp_path: Path) -> None:
    spec = RepoTaskSpec(repo_path=tmp_path, issue="broken", verify_command=["pytest"])
    state = RepoRunState(run_id="r1", task=spec, phase=Phase.COMPLETE)
    service = RepoPilotService(
        scheduler_factory=lambda *_: Scheduler(state),
        store_factory=RunStore,
    )

    accepted = service.submit(spec)
    await service._operation_tasks[accepted.operation_id]
    completed = service.operation(accepted.operation_id)

    assert completed.status.value == "completed"
    assert completed.run_id == "r1"
