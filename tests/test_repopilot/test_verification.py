from pathlib import Path

import pytest

from openharness.repopilot.failures import FailureCategory
from openharness.repopilot.verification import (
    CompositeVerifier,
    DiffSanityCheck,
    DiffScopeCheck,
    PythonCompileCheck,
    VerificationCheckResult,
)


class FakeCheck:
    def __init__(self, result: VerificationCheckResult, calls: list[str]):
        self.result = result
        self.calls = calls
        self.name = result.name

    async def run(
        self, cwd: Path, *, attempt: int, artifact_dir: Path | None = None
    ) -> VerificationCheckResult:
        del cwd, attempt, artifact_dir
        self.calls.append(self.name)
        return self.result


def result(
    name: str,
    *,
    passed: bool,
    required: bool = True,
    fatal: bool = False,
) -> VerificationCheckResult:
    return VerificationCheckResult(
        name=name,
        passed=passed,
        required=required,
        fatal=fatal,
        category=FailureCategory.PASSED if passed else FailureCategory.ASSERTION,
        summary="passed" if passed else "failed",
    )


@pytest.mark.asyncio
async def test_composite_verifier_aggregates_required_and_optional_checks(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    verifier = CompositeVerifier(
        [
            FakeCheck(result("compile", passed=True), calls),
            FakeCheck(result("lint", passed=False, required=False), calls),
            FakeCheck(result("target", passed=True), calls),
        ]
    )

    suite = await verifier.verify(tmp_path, attempt=1)

    assert suite.passed is True
    assert calls == ["compile", "lint", "target"]
    assert [check.name for check in suite.checks] == calls
    assert suite.failed_required == []


@pytest.mark.asyncio
async def test_fatal_failure_skips_remaining_checks(tmp_path: Path) -> None:
    calls: list[str] = []
    verifier = CompositeVerifier(
        [
            FakeCheck(result("scope", passed=False, fatal=True), calls),
            FakeCheck(result("target", passed=True), calls),
        ]
    )

    suite = await verifier.verify(tmp_path, attempt=1)

    assert suite.passed is False
    assert calls == ["scope"]
    assert suite.failed_required == ["scope"]
    assert suite.primary_failure is not None
    assert suite.primary_failure.category is FailureCategory.ASSERTION


@pytest.mark.asyncio
async def test_composite_verifier_passes_artifact_directory(tmp_path: Path) -> None:
    seen: list[Path | None] = []

    class ArtifactCheck:
        name = "artifact"

        async def run(self, cwd: Path, *, attempt: int, artifact_dir: Path | None = None):
            del cwd, attempt
            seen.append(artifact_dir)
            return result("artifact", passed=True)

    artifacts = tmp_path / "artifacts"
    suite = await CompositeVerifier([ArtifactCheck()]).verify(
        tmp_path, attempt=2, artifact_dir=artifacts
    )

    assert suite.passed
    assert seen == [artifacts]
    assert artifacts.is_dir()


@pytest.mark.asyncio
async def test_python_compile_check_reports_syntax_failure(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    suite = await CompositeVerifier([PythonCompileCheck(["broken.py"])]).verify(
        tmp_path, attempt=1, artifact_dir=tmp_path / "artifacts"
    )

    assert suite.passed is False
    assert suite.checks[0].category is FailureCategory.SYNTAX
    assert suite.checks[0].artifact_path


@pytest.mark.asyncio
async def test_diff_checks_reject_empty_and_out_of_scope_changes(tmp_path: Path) -> None:
    scope = await DiffScopeCheck(
        ["src/allowed.py", "secrets.txt"],
        ["src/**"],
        max_changed_files=3,
    ).run(tmp_path, attempt=1)
    empty = await DiffSanityCheck("", []).run(tmp_path, attempt=1)

    assert scope.category is FailureCategory.OUT_OF_SCOPE_DIFF
    assert scope.fatal is True
    assert empty.category is FailureCategory.NO_DIFF
