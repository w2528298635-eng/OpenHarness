from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class BenchmarkCase(BaseModel):
    id: str
    task: Path


class BenchmarkManifest(BaseModel):
    name: str
    cases: list[BenchmarkCase]

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, value: list[BenchmarkCase]) -> list[BenchmarkCase]:
        ids = [case.id for case in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate benchmark case id")
        if not value:
            raise ValueError("benchmark requires at least one case")
        return value


def load_benchmark(path: Path) -> BenchmarkManifest:
    manifest_path = path.expanduser().resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = BenchmarkManifest.model_validate(payload)
    for case in manifest.cases:
        if not case.task.is_absolute():
            case.task = (manifest_path.parent / case.task).resolve()
    return manifest
