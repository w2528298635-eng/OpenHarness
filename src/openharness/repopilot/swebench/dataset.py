from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
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
        revision: str | None = None,
        split: str = "test",
        api_factory: Callable[[], Any] | None = None,
        loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
        file_downloader: Callable[..., str] | None = None,
        parquet_reader: Callable[[str], Any] | None = None,
        cache_dir: Path | None = None,
    ):
        self.dataset_name = dataset_name
        self._revision = revision
        self.split = split
        self._api_factory = api_factory
        self._loader = loader
        self._file_downloader = file_downloader
        self._parquet_reader = parquet_reader
        self._cache_dir = cache_dir or Path(tempfile.gettempdir()) / "repopilot-hf"

    @property
    def revision(self) -> str:
        if self._revision is None:
            if self._api_factory is None:
                try:
                    from huggingface_hub import HfApi
                except ImportError as exc:
                    raise RuntimeError(
                        "Hugging Face support is not installed; install "
                        "OpenHarness with the 'swebench' extra"
                    ) from exc
                self._api_factory = HfApi
            info = self._api_factory().dataset_info(self.dataset_name)
            revision = getattr(info, "sha", None)
            if not isinstance(revision, str) or not revision:
                raise RuntimeError(
                    f"Hugging Face returned no immutable revision for {self.dataset_name}"
                )
            self._revision = revision
        return self._revision

    def rows(self) -> Iterable[Mapping[str, Any]]:
        loader = self._loader
        if loader is not None:
            return loader(
                self.dataset_name,
                split=self.split,
                revision=self.revision,
                streaming=True,
            )
        downloader = self._file_downloader
        if downloader is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face dataset support is not installed; "
                    "install OpenHarness with the 'swebench' extra"
                ) from exc
            downloader = hf_hub_download
        parquet_reader = self._parquet_reader
        if parquet_reader is None:
            try:
                from pyarrow.parquet import read_table
            except ImportError as exc:
                raise RuntimeError(
                    "PyArrow is required to read the public SWE-bench metadata"
                ) from exc
            parquet_reader = read_table
        parquet_path = downloader(
            repo_id=self.dataset_name,
            repo_type="dataset",
            filename=f"data/{self.split}-00000-of-00001.parquet",
            revision=self.revision,
            cache_dir=str(self._cache_dir),
        )
        return parquet_reader(parquet_path).to_pylist()


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
