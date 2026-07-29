from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from openharness.repopilot.api import create_app
from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.service import UnknownRunError


class Service:
    def __init__(self, tmp_path: Path):
        self.state = RepoRunState(
            run_id="r1",
            task=RepoTaskSpec(
                repo_path=tmp_path,
                issue="broken",
                verify_command=["pytest"],
            ),
            phase=Phase.COMPLETE,
        )

    async def start(self, spec):
        return self.state

    def submit(self, spec):
        from openharness.repopilot.service import OperationState, RunOperation

        return RunOperation(operation_id="op1", status=OperationState.ACCEPTED)

    def operation(self, operation_id):
        from openharness.repopilot.service import OperationState, RunOperation

        if operation_id != "op1":
            raise UnknownRunError(operation_id)
        return RunOperation(operation_id="op1", status=OperationState.RUNNING)

    def get(self, run_id, repo=None):
        if run_id != "r1":
            raise UnknownRunError(run_id)
        return self.state

    def events(self, run_id, repo=None):
        self.get(run_id)
        return []

    def artifacts(self, run_id, repo=None):
        self.get(run_id)
        return []

    def cancel(self, run_id):
        self.get(run_id)


def test_api_health_status_and_unknown_run(tmp_path: Path) -> None:
    client = TestClient(create_app(Service(tmp_path)))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/runs/r1").status_code == 200
    assert client.get("/runs/missing").status_code == 404
    assert client.get("/operations/op1").json()["status"] == "running"
    assert client.post("/runs/r1/cancel").status_code == 202
