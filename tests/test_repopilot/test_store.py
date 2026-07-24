import json
from pathlib import Path

from openharness.repopilot.models import Phase, RepoRunState, RepoTaskSpec
from openharness.repopilot.store import RunStore


def _state(repo: Path) -> RepoRunState:
    return RepoRunState(
        run_id="run-1",
        task=RepoTaskSpec(repo_path=repo, issue="broken", verify_command=["pytest"]),
        phase=Phase.PRECHECK,
    )


def test_store_round_trips_state_and_appends_events(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    state = _state(tmp_path)
    store.create(state)
    state.phase = Phase.ANALYZE
    store.save_state(state)
    store.append_event({"kind": "phase", "value": 1})
    store.append_event({"kind": "phase", "value": 2})

    assert store.load_state("run-1").phase is Phase.ANALYZE
    lines = (store.run_dir("run-1") / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["value"] for line in lines] == [1, 2]
    assert not (store.run_dir("run-1") / "state.json.tmp").exists()


def test_orphaned_temporary_file_does_not_replace_durable_state(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    state = _state(tmp_path)
    store.create(state)
    (store.run_dir("run-1") / "state.json.tmp").write_text("broken", encoding="utf-8")

    assert store.load_state("run-1").phase is Phase.PRECHECK
