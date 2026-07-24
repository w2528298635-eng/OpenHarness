from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import RepoRunState


class RunStore:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.root = self.repo_root / ".openharness" / "repopilot" / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, state: RepoRunState) -> Path:
        directory = self.run_dir(state.run_id)
        directory.mkdir(parents=True, exist_ok=False)
        self.save_state(state)
        (directory / "events.jsonl").touch()
        return directory

    def save_state(self, state: RepoRunState) -> None:
        directory = self.run_dir(state.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "state.json.tmp"
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(directory / "state.json")

    def load_state(self, run_id: str) -> RepoRunState:
        return RepoRunState.model_validate_json(
            (self.run_dir(run_id) / "state.json").read_text(encoding="utf-8")
        )

    def append_event(self, event: BaseModel | dict[str, Any]) -> None:
        if isinstance(event, BaseModel):
            payload = event.model_dump(mode="json")
            run_id = getattr(event, "run_id", None)
        else:
            payload = event
            run_id = event.get("run_id")
        if run_id is None:
            candidates = [path for path in self.root.glob("*") if path.is_dir()]
            if len(candidates) != 1:
                raise ValueError("event must include run_id when store has multiple runs")
            target = candidates[0]
        else:
            target = self.run_dir(str(run_id))
        with (target / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def write_json(self, run_id: str, name: str, value: BaseModel | Any) -> Path:
        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        return self.write_text(
            run_id, name, json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )

    def write_text(self, run_id: str, name: str, text: str) -> Path:
        target = self.run_dir(run_id) / name
        target.write_text(text, encoding="utf-8")
        return target
