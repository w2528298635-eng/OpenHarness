import json
from pathlib import Path

import pytest

from openharness.repopilot.events import RunEvent, RunEventKind
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


def test_artifact_write_recreates_missing_run_directory(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.create(_state(tmp_path))
    run_dir = store.run_dir("run-1")
    for path in run_dir.iterdir():
        path.unlink()
    run_dir.rmdir()

    written = store.write_text("run-1", "context.json", "{}")

    assert written.read_text(encoding="utf-8") == "{}"


def test_artifact_write_retries_transient_missing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path)
    store.create(_state(tmp_path))
    original_write_text = Path.write_text
    attempts = 0

    def transient_write(path: Path, *args, **kwargs):
        nonlocal attempts
        if path.name == "context.json" and attempts == 0:
            attempts += 1
            raise FileNotFoundError(path)
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", transient_write)

    written = store.write_text("run-1", "context.json", "{}")

    assert attempts == 1
    assert written.read_text(encoding="utf-8") == "{}"


def test_configured_run_root_keeps_artifacts_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    central = tmp_path / "central"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OPENHARNESS_REPOPILOT_RUN_ROOT", str(central))

    store = RunStore(repo)
    store.create(_state(repo))

    assert store.root.is_relative_to(central)
    assert not (repo / ".openharness").exists()


def test_store_loads_legacy_events_as_typed_events(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.create(_state(tmp_path))
    store.append_event({"kind": "phase", "value": 1})

    events = store.load_events("run-1")

    assert len(events) == 1
    assert events[0].run_id == "run-1"
    assert events[0].kind is RunEventKind.LEGACY
    assert events[0].data == {"kind": "phase", "value": 1}


def test_event_write_failure_is_non_fatal_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    store.create(_state(tmp_path))
    original_open = Path.open

    def broken_events_open(path: Path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise OSError("disk unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken_events_open)

    written = store.append_event(RunEvent.create(run_id="run-1", kind=RunEventKind.RUN_STARTED))

    assert written is False
    assert store.event_warnings == ["event_write_failed: disk unavailable"]
