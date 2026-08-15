from pathlib import Path

import pytest

from openharness.repopilot.benchmark import load_benchmark


def test_benchmark_manifest_resolves_task_paths(tmp_path: Path) -> None:
    task = tmp_path / "task.yaml"
    task.write_text("x", encoding="utf-8")
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text(
        "name: smoke\ncases:\n  - id: boundary\n    task: task.yaml\n",
        encoding="utf-8",
    )

    loaded = load_benchmark(manifest)

    assert loaded.cases[0].task == task.resolve()


def test_benchmark_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text(
        "name: smoke\ncases:\n  - {id: x, task: a.yaml}\n  - {id: x, task: b.yaml}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark(manifest)
