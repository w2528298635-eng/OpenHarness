from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from pathlib import Path

from .models import VerificationResult

_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s:]+(?:\\[^\s:]+)*")
_POSIX_PATH = re.compile(r"/(?:[^/\s:]+/)*[^/\s:]+")
_LINE = re.compile(r":\d+")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?s\b")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def normalize_failure_signature(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    text = _WINDOWS_PATH.sub("<path>", text)
    text = _POSIX_PATH.sub("<path>", text)
    text = _LINE.sub(":<line>", text)
    text = _DURATION.sub("<time>", text)
    text = _ADDRESS.sub("<address>", text)
    text = " ".join(text.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _classify(exit_code: int, stdout: str, stderr: str) -> str:
    if exit_code == 0:
        return "passed"
    combined = f"{stdout}\n{stderr}".lower()
    collection_markers = (
        "error collecting",
        "interrupted: 1 error during collection",
        "modulenotfounderror",
        "importerror while importing test module",
    )
    if any(marker in combined for marker in collection_markers):
        return "collection_error"
    if "failed" in combined or "assertionerror" in combined:
        return "test_failure"
    return "infrastructure_error"


class PythonPytestVerifier:
    def __init__(self, timeout_seconds: float = 300):
        self.timeout_seconds = timeout_seconds

    async def verify(self, argv: list[str], cwd: Path, *, attempt: int) -> VerificationResult:
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return VerificationResult(
                attempt=attempt,
                command=argv,
                passed=False,
                category="missing_executable",
                stderr=str(exc),
                duration_seconds=time.monotonic() - started,
                failure_signature=normalize_failure_signature("", str(exc)),
            )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return VerificationResult(
                attempt=attempt,
                command=argv,
                passed=False,
                category="timeout",
                duration_seconds=time.monotonic() - started,
                failure_signature=normalize_failure_signature("timeout", ""),
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        exit_code = process.returncode or 0
        category = _classify(exit_code, stdout, stderr)
        return VerificationResult(
            attempt=attempt,
            command=argv,
            passed=exit_code == 0,
            exit_code=exit_code,
            category=category,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            failure_signature=(
                None if exit_code == 0 else normalize_failure_signature(stdout, stderr)
            ),
        )
