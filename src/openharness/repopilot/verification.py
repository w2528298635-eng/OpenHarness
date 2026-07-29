from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path, PurePath
from typing import Protocol

from pydantic import BaseModel, Field, computed_field

from .failures import (
    FailureCategory,
    FailureRecord,
    classify_failure,
    normalize_failure_signature,
)
from .models import Phase

_SUMMARY_LIMIT = 4000


class VerificationCheckResult(BaseModel):
    name: str
    passed: bool
    required: bool = True
    fatal: bool = False
    category: FailureCategory
    summary: str = ""
    stdout: str = ""
    stderr: str = ""
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    duration_seconds: float = 0
    artifact_path: str | None = None
    failure_signature: str | None = None


class VerificationSuiteResult(BaseModel):
    attempt: int
    checks: list[VerificationCheckResult]
    duration_seconds: float

    @computed_field
    @property
    def passed(self) -> bool:
        return all(check.passed or not check.required for check in self.checks)

    @computed_field
    @property
    def failed_required(self) -> list[str]:
        return [check.name for check in self.checks if check.required and not check.passed]

    @computed_field
    @property
    def primary_failure(self) -> FailureRecord | None:
        for check in self.checks:
            if check.required and not check.passed:
                return FailureRecord(
                    category=check.category,
                    source_phase=Phase.VERIFY,
                    signature=check.failure_signature
                    or normalize_failure_signature(check.summary, ""),
                    summary=check.summary,
                    artifact_path=check.artifact_path,
                )
        return None


class VerificationCheck(Protocol):
    name: str

    async def run(
        self,
        cwd: Path,
        *,
        attempt: int,
        artifact_dir: Path | None = None,
    ) -> VerificationCheckResult: ...


class CompositeVerifier:
    def __init__(self, checks: list[VerificationCheck]):
        if not checks:
            raise ValueError("composite verifier requires at least one check")
        self.checks = list(checks)

    async def verify(
        self,
        cwd: Path,
        *,
        attempt: int,
        artifact_dir: Path | None = None,
    ) -> VerificationSuiteResult:
        started = time.monotonic()
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        results: list[VerificationCheckResult] = []
        for check in self.checks:
            result = await check.run(cwd, attempt=attempt, artifact_dir=artifact_dir)
            results.append(result)
            if not result.passed and result.fatal:
                break
        return VerificationSuiteResult(
            attempt=attempt,
            checks=results,
            duration_seconds=time.monotonic() - started,
        )


class CommandVerificationCheck:
    def __init__(
        self,
        *,
        name: str,
        argv: list[str],
        timeout_seconds: float = 300,
        required: bool = True,
        fatal: bool = False,
    ):
        if not argv:
            raise ValueError("verification command must not be empty")
        self.name = name
        self.argv = list(argv)
        self.timeout_seconds = timeout_seconds
        self.required = required
        self.fatal = fatal

    async def run(
        self,
        cwd: Path,
        *,
        attempt: int,
        artifact_dir: Path | None = None,
    ) -> VerificationCheckResult:
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *self.argv,
                cwd=str(cwd),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return self._failure(
                category=FailureCategory.CONFIGURATION,
                summary=str(exc),
                duration=time.monotonic() - started,
            )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return self._failure(
                category=FailureCategory.TIMEOUT,
                summary=f"command timed out after {self.timeout_seconds}s",
                duration=time.monotonic() - started,
            )
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        exit_code = process.returncode or 0
        artifact_path = self._write_artifact(
            artifact_dir, attempt=attempt, stdout=stdout, stderr=stderr
        )
        failure = classify_failure(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            source_phase=Phase.VERIFY,
            artifact_path=artifact_path,
        )
        summary = (stderr.strip() or stdout.strip() or failure.category.value)[:_SUMMARY_LIMIT]
        return VerificationCheckResult(
            name=self.name,
            passed=exit_code == 0,
            required=self.required,
            fatal=self.fatal,
            category=failure.category,
            summary=summary,
            stdout=stdout[:_SUMMARY_LIMIT],
            stderr=stderr[:_SUMMARY_LIMIT],
            command=self.argv,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
            artifact_path=artifact_path,
            failure_signature=None if exit_code == 0 else failure.signature,
        )

    def _failure(
        self,
        *,
        category: FailureCategory,
        summary: str,
        duration: float,
    ) -> VerificationCheckResult:
        return VerificationCheckResult(
            name=self.name,
            passed=False,
            required=self.required,
            fatal=self.fatal,
            category=category,
            summary=summary[:_SUMMARY_LIMIT],
            command=self.argv,
            duration_seconds=duration,
            failure_signature=normalize_failure_signature(summary, ""),
        )

    def _write_artifact(
        self,
        artifact_dir: Path | None,
        *,
        attempt: int,
        stdout: str,
        stderr: str,
    ) -> str | None:
        if artifact_dir is None:
            return None
        target = artifact_dir / f"verification-{attempt}-{self.name}.log"
        target.write_text(
            f"$ {' '.join(self.argv)}\n\nSTDOUT\n{stdout}\n\nSTDERR\n{stderr}",
            encoding="utf-8",
        )
        return str(target)


class PythonCompileCheck(CommandVerificationCheck):
    def __init__(self, changed_files: list[str], *, required: bool = True):
        python_files = [path for path in changed_files if path.endswith(".py")]
        argv = [sys.executable, "-m", "py_compile", *python_files]
        super().__init__(
            name="python_compile",
            argv=argv,
            required=required,
            fatal=True,
        )
        self.python_files = python_files

    async def run(
        self,
        cwd: Path,
        *,
        attempt: int,
        artifact_dir: Path | None = None,
    ) -> VerificationCheckResult:
        if not self.python_files:
            return VerificationCheckResult(
                name=self.name,
                passed=True,
                required=self.required,
                fatal=self.fatal,
                category=FailureCategory.PASSED,
                summary="no changed Python files",
            )
        return await super().run(cwd, attempt=attempt, artifact_dir=artifact_dir)


class PytestCheck(CommandVerificationCheck):
    pass


class DiffScopeCheck:
    name = "diff_scope"

    def __init__(
        self,
        changed_files: list[str],
        allowed_paths: list[str] | None,
        *,
        max_changed_files: int,
    ):
        self.changed_files = list(changed_files)
        self.allowed_paths = allowed_paths
        self.max_changed_files = max_changed_files

    async def run(
        self,
        cwd: Path,
        *,
        attempt: int,
        artifact_dir: Path | None = None,
    ) -> VerificationCheckResult:
        del cwd, attempt, artifact_dir
        violations: list[str] = []
        if len(self.changed_files) > self.max_changed_files:
            violations.append(
                f"{len(self.changed_files)} changed files exceeds {self.max_changed_files}"
            )
        for path in self.changed_files:
            normalized = path.replace("\\", "/")
            sensitive = normalized == ".git" or normalized.startswith(
                (".git/", ".openharness/")
            )
            outside_allowlist = self.allowed_paths and not any(
                PurePath(normalized).match(pattern) for pattern in self.allowed_paths
            )
            if sensitive or outside_allowlist:
                violations.append(path)
        return VerificationCheckResult(
            name=self.name,
            passed=not violations,
            required=True,
            fatal=True,
            category=(
                FailureCategory.PASSED
                if not violations
                else FailureCategory.OUT_OF_SCOPE_DIFF
            ),
            summary="passed" if not violations else f"out of scope: {', '.join(violations)}",
            failure_signature=(
                None
                if not violations
                else normalize_failure_signature("\n".join(violations), "")
            ),
        )


class DiffSanityCheck:
    name = "diff_sanity"

    def __init__(self, diff: str, changed_files: list[str]):
        self.diff = diff
        self.changed_files = list(changed_files)

    async def run(
        self,
        cwd: Path,
        *,
        attempt: int,
        artifact_dir: Path | None = None,
    ) -> VerificationCheckResult:
        del cwd, attempt, artifact_dir
        if not self.diff.strip() or not self.changed_files:
            return VerificationCheckResult(
                name=self.name,
                passed=False,
                required=True,
                fatal=True,
                category=FailureCategory.NO_DIFF,
                summary="no source diff was produced",
                failure_signature=normalize_failure_signature("no diff", ""),
            )
        test_only = all(
            Path(path).name.startswith("test_")
            or "tests" in {part.casefold() for part in Path(path).parts}
            for path in self.changed_files
        )
        return VerificationCheckResult(
            name=self.name,
            passed=True,
            required=True,
            fatal=False,
            category=FailureCategory.PASSED,
            summary="diff changes only tests" if test_only else "passed",
        )
