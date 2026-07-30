from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from openharness.repopilot.swebench.dataset import (
    JsonDatasetProvider,
    ManifestConflictError,
    prepare_manifest,
)
from openharness.repopilot.swebench.models import SampleManifest, SamplingConfig


class StaticProvider:
    dataset_name = "public/test"
    revision = "revision-123"

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def rows(self) -> Iterable[Mapping[str, Any]]:
        return list(self._rows)


def _rows(*, suffix: str = "") -> list[dict[str, Any]]:
    return [
        {
            "instance_id": f"alpha__repo-easy{suffix}",
            "repo": "alpha/repo",
            "base_commit": "abc",
            "problem_statement": "Fix easy issue.",
            "difficulty": "<15 min fix",
            "patch": "private gold patch",
            "test_patch": "private tests",
        },
        {
            "instance_id": f"beta__repo-medium{suffix}",
            "repo": "beta/repo",
            "base_commit": "def",
            "problem_statement": "Fix medium issue.",
            "difficulty": "15 min - 1 hour",
            "FAIL_TO_PASS": ["private_test"],
        },
        {
            "instance_id": f"gamma__repo-hard{suffix}",
            "repo": "gamma/repo",
            "base_commit": "ghi",
            "problem_statement": "Fix hard issue.",
            "difficulty": "1-4 hours",
            "PASS_TO_PASS": ["private_regression"],
        },
    ]


def test_prepare_manifest_persists_revision_and_strips_gold_fields(tmp_path: Path) -> None:
    target = tmp_path / "formal-manifest.json"

    manifest = prepare_manifest(
        StaticProvider(_rows()),
        target,
        SamplingConfig(easy=1, medium=1, hard=1, seed=20260730),
    )

    persisted = SampleManifest.model_validate_json(target.read_text(encoding="utf-8"))
    assert persisted == manifest
    assert persisted.dataset_revision == "revision-123"
    raw = target.read_text(encoding="utf-8")
    assert "private gold patch" not in raw
    assert "FAIL_TO_PASS" not in raw
    assert not list(tmp_path.glob("*.tmp"))


def test_prepare_manifest_refuses_to_replace_a_different_manifest(tmp_path: Path) -> None:
    target = tmp_path / "formal-manifest.json"
    config = SamplingConfig(easy=1, medium=1, hard=1, seed=20260730)
    prepare_manifest(StaticProvider(_rows()), target, config)

    with pytest.raises(ManifestConflictError, match="use force=True"):
        prepare_manifest(StaticProvider(_rows(suffix="-changed")), target, config)


def test_prepare_manifest_force_replaces_a_different_manifest(tmp_path: Path) -> None:
    target = tmp_path / "formal-manifest.json"
    config = SamplingConfig(easy=1, medium=1, hard=1, seed=20260730)
    original = prepare_manifest(StaticProvider(_rows()), target, config)

    replaced = prepare_manifest(
        StaticProvider(_rows(suffix="-changed")),
        target,
        config,
        force=True,
    )

    assert replaced.sha256 != original.sha256
    assert "changed" in target.read_text(encoding="utf-8")


def test_json_dataset_provider_reads_jsonl_and_records_explicit_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n",
        encoding="utf-8",
    )

    provider = JsonDatasetProvider(
        source,
        dataset_name="public/offline",
        revision="sha256:fixture",
    )

    assert provider.dataset_name == "public/offline"
    assert provider.revision == "sha256:fixture"
    assert list(provider.rows()) == _rows()

