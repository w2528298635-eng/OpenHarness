import sys
from pathlib import Path

import pytest

from openharness.repopilot.verifier import PythonPytestVerifier, normalize_failure_signature


@pytest.mark.asyncio
async def test_verifier_classifies_pass_and_failure(tmp_path: Path) -> None:
    passing = await PythonPytestVerifier().verify(
        [sys.executable, "-c", "print('ok')"], tmp_path, attempt=1
    )
    failing = await PythonPytestVerifier().verify(
        [sys.executable, "-c", "print('FAILED test_x.py::test_x'); raise SystemExit(1)"],
        tmp_path,
        attempt=2,
    )

    assert passing.passed and passing.category == "passed"
    assert not failing.passed and failing.category == "test_failure"
    assert failing.failure_signature


@pytest.mark.asyncio
async def test_verifier_classifies_missing_executable_and_timeout(tmp_path: Path) -> None:
    missing = await PythonPytestVerifier().verify(
        ["definitely-not-a-real-executable"], tmp_path, attempt=1
    )
    timed_out = await PythonPytestVerifier(timeout_seconds=0.05).verify(
        [sys.executable, "-c", "import time; time.sleep(1)"], tmp_path, attempt=2
    )

    assert missing.category == "missing_executable"
    assert timed_out.category == "timeout"


def test_failure_signature_ignores_volatile_values() -> None:
    left = normalize_failure_signature(
        r"C:\tmp\a\test_x.py:12 failed in 0.13s at 0x7FFABC", ""
    )
    right = normalize_failure_signature(
        r"D:\other\test_x.py:99 failed in 3.27s at 0x1AAEEE", ""
    )

    assert left == right
