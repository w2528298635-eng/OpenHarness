import shutil
import subprocess
import sys
from pathlib import Path

from openharness.repopilot.benchmark import load_benchmark
from openharness.repopilot.task_loader import load_task


def test_discount_example_is_reproducible_and_manifests_load(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "examples" / "repopilot"
    target = tmp_path / "repopilot"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            ".openharness",
            ".openharness-repopilot-worktrees",
            ".pytest_cache",
            "__pycache__",
        ),
    )
    repo = target / "discount_bug"
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_discount.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert load_task(target / "task.example.yaml").repo_path == repo.resolve()
    assert (
        load_benchmark(target / "benchmark.example.yaml").cases[0].task
        == (target / "task.example.yaml").resolve()
    )
