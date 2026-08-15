from __future__ import annotations

from pathlib import Path

from .failures import FailureCategory, normalize_failure_signature
from .models import VerificationResult
from .verification import CommandVerificationCheck

__all__ = ["PythonPytestVerifier", "normalize_failure_signature"]


class PythonPytestVerifier:
    def __init__(self, timeout_seconds: float = 300):
        self.timeout_seconds = timeout_seconds

    async def verify(self, argv: list[str], cwd: Path, *, attempt: int) -> VerificationResult:
        result = await CommandVerificationCheck(
            name="pytest",
            argv=argv,
            timeout_seconds=self.timeout_seconds,
        ).run(cwd, attempt=attempt)
        category = {
            FailureCategory.PASSED: "passed",
            FailureCategory.ASSERTION: "test_failure",
            FailureCategory.COLLECTION: "collection_error",
            FailureCategory.DEPENDENCY: "collection_error",
            FailureCategory.CONFIGURATION: "missing_executable",
            FailureCategory.TIMEOUT: "timeout",
        }.get(result.category, "infrastructure_error")
        return VerificationResult(
            attempt=attempt,
            command=argv,
            passed=result.passed,
            exit_code=result.exit_code,
            category=category,
            stdout=result.stdout,
            stderr=result.stderr or (result.summary if not result.passed else ""),
            duration_seconds=result.duration_seconds,
            failure_signature=result.failure_signature,
        )
