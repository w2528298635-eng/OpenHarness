"""Public, leakage-safe pytest check used by the SWE-bench compatibility adapter."""

from __future__ import annotations

import subprocess


def test_repository_contains_an_uncommitted_patch() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "the agent has not produced a patch"
