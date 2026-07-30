from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .models import SampleManifest, SamplingConfig
from .sampler import sample_manifest

_PUBLIC_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "difficulty",
)


class ManifestConflictError(RuntimeError):
    pass


class DatasetProvider(Protocol):
    dataset_name: str
    revision: str

    def rows(self) -> Iterable[Mapping[str, Any]]: ...


class JsonDatasetProvider:
    def __init__(
        self,
        path: Path,
        *,
        dataset_name: str,
        revision: str,
    ):
        self.path = path
        self.dataset_name = dataset_name
        self.revision = revision

    def rows(self) -> Iterable[Mapping[str, Any]]:
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix.casefold() == ".jsonl":
            return [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise TypeError("offline SWE-bench JSON must contain a list of rows")
        return payload


class HuggingFaceDatasetProvider:
    """Lazy public dataset provider; `datasets` remains an optional dependency."""

    def __init__(
        self,
        *,
        dataset_name: str = "SWE-bench/SWE-bench_Verified",
        revision: str,
        split: str = "test",
    ):
        self.dataset_name = dataset_name
        self.revision = revision
        self.split = split

    def rows(self) -> Iterable[Mapping[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dataset support is not installed; "
                "install OpenHarness with the 'swebench' extra"
            ) from exc
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            revision=self.revision,
        )
        return dataset


def _public_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {field: row[field] for field in _PUBLIC_FIELDS}


def _atomic_write_manifest(target: Path, manifest: SampleManifest) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = manifest.model_dump_json(indent=2)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def prepare_manifest(
    provider: DatasetProvider,
    output_path: Path,
    config: SamplingConfig,
    *,
    force: bool = False,
) -> SampleManifest:
    effective_config = config.model_copy(
        update={"dataset_name": provider.dataset_name}
    )
    manifest = sample_manifest(
        _public_rows(provider.rows()),
        effective_config,
        dataset_revision=provider.revision,
    )
    if output_path.exists():
        existing = SampleManifest.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if existing.sha256 == manifest.sha256:
            return existing
        if not force:
            raise ManifestConflictError(
                f"{output_path} contains a different frozen manifest; "
                "use force=True to replace it explicitly"
            )
    _atomic_write_manifest(output_path, manifest)
    return manifest
