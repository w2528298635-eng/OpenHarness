from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field

GIB = 1024**3


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandRunner(Protocol):
    def run(self, argv: list[str], *, timeout_seconds: float) -> CommandResult: ...


class SubprocessCommandRunner:
    def run(self, argv: list[str], *, timeout_seconds: float) -> CommandResult:
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=None,
                stdout=_decode_timeout_stream(exc.stdout),
                stderr=_decode_timeout_stream(exc.stderr) or "command timed out",
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return CommandResult(exit_code=None, stdout="", stderr=str(exc))
        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


class SystemFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    machine: str
    logical_cpus: int = Field(ge=0)
    memory_bytes: int = Field(ge=0)
    free_disk_bytes: int = Field(ge=0)

    @classmethod
    def detect(cls, cache_path: Path) -> SystemFacts:
        existing = cache_path.resolve()
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        return cls(
            system=platform.system(),
            machine=platform.machine(),
            logical_cpus=os.cpu_count() or 0,
            memory_bytes=_total_memory(),
            free_disk_bytes=shutil.disk_usage(existing).free,
        )


def _total_memory() -> int:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return 0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return 0
    return int(pages * page_size)


class DoctorCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: Literal["pass", "warn", "fail"]
    summary: str


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: SystemFacts
    checks: tuple[DoctorCheck, ...]

    @computed_field
    @property
    def formal_ready(self) -> bool:
        return all(check.status != "fail" for check in self.checks)


def _command_check(
    name: str,
    result: CommandResult,
    *,
    success_summary: str,
) -> DoctorCheck:
    if result.exit_code == 0 and not result.timed_out:
        return DoctorCheck(name=name, status="pass", summary=success_summary)
    detail = (result.stderr.strip() or result.stdout.strip() or "command failed")[:1000]
    return DoctorCheck(name=name, status="fail", summary=detail)


def run_doctor(
    cache_path: Path,
    *,
    command_runner: CommandRunner | None = None,
    system_facts: SystemFacts | None = None,
) -> DoctorReport:
    runner = command_runner or SubprocessCommandRunner()
    facts = system_facts or SystemFacts.detect(cache_path)
    checks: list[DoctorCheck] = []
    checks.append(
        _command_check(
            "docker_client",
            runner.run(["docker", "--version"], timeout_seconds=15),
            success_summary="Docker client is available",
        )
    )
    checks.append(
        _command_check(
            "docker_daemon",
            runner.run(
                ["docker", "info", "--format", "{{json .}}"],
                timeout_seconds=30,
            ),
            success_summary="Docker Engine is reachable",
        )
    )
    if facts.system.casefold() == "windows":
        checks.append(
            _command_check(
                "wsl2",
                runner.run(["wsl", "--status"], timeout_seconds=15),
                success_summary="WSL status is available",
            )
        )

    machine = facts.machine.casefold()
    supported_arch = machine in {"amd64", "x86_64"}
    checks.append(
        DoctorCheck(
            name="architecture",
            status="pass" if supported_arch else "fail",
            summary=f"{facts.machine}; x86_64/AMD64 is required for formal local runs",
        )
    )
    checks.append(
        DoctorCheck(
            name="cpu",
            status="pass" if facts.logical_cpus >= 8 else "warn",
            summary=f"{facts.logical_cpus} logical CPUs; 8 or more recommended",
        )
    )
    checks.append(
        DoctorCheck(
            name="memory",
            status="pass" if facts.memory_bytes >= 16 * GIB else "warn",
            summary=f"{facts.memory_bytes / GIB:.1f} GiB RAM; 16 GiB or more recommended",
        )
    )
    free_gib = facts.free_disk_bytes / GIB
    if facts.free_disk_bytes < 20 * GIB:
        disk_status: Literal["pass", "warn", "fail"] = "fail"
        disk_summary = (
            f"{free_gib:.1f} GiB free near cache; 20 GiB minimum is required "
            "before a three-instance subset calibration"
        )
    elif facts.free_disk_bytes < 120 * GIB:
        disk_status = "warn"
        disk_summary = (
            f"{free_gib:.1f} GiB free near cache; below the 120 GiB full-suite "
            "recommendation, so run the three-instance disk projection first"
        )
    else:
        disk_status = "pass"
        disk_summary = (
            f"{free_gib:.1f} GiB free near cache; meets the 120 GiB "
            "full-suite recommendation"
        )
    checks.append(
        DoctorCheck(name="disk", status=disk_status, summary=disk_summary)
    )
    return DoctorReport(facts=facts, checks=tuple(checks))


class HarnessPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    model_name_or_path: str
    model_patch: str


def write_predictions_jsonl(
    target: Path,
    predictions: Sequence[HarnessPrediction],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            prediction.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for prediction in predictions
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class HarnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "completed",
        "failed",
        "timeout",
        "missing_report",
    ]
    total: int = 0
    submitted: int = 0
    completed: int = 0
    resolved: int = 0
    resolution_rate: float = 0.0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    report_path: str | None = None


class OfficialHarnessRunner:
    def __init__(
        self,
        *,
        python_executable: str,
        command_runner: CommandRunner | None = None,
    ):
        self.python_executable = python_executable
        self.command_runner = command_runner or SubprocessCommandRunner()

    def evaluate(
        self,
        *,
        dataset_name: str,
        predictions_path: Path,
        run_id: str,
        result_path: Path,
        max_workers: int,
        cache_level: Literal["none", "base", "env", "instance"],
        timeout_seconds: float = 7200,
        instance_ids: Sequence[str] = (),
        clean: bool = True,
    ) -> HarnessResult:
        argv = [
            self.python_executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
            "--cache_level",
            cache_level,
            "--clean",
            str(clean).lower(),
        ]
        if instance_ids:
            argv.extend(["--instance_ids", *instance_ids])
        execution = self.command_runner.run(argv, timeout_seconds=timeout_seconds)
        common = {
            "exit_code": execution.exit_code,
            "stdout": execution.stdout[-4000:],
            "stderr": execution.stderr[-4000:],
        }
        if execution.timed_out:
            return HarnessResult(status="timeout", **common)
        if execution.exit_code != 0:
            return HarnessResult(status="failed", **common)
        if not result_path.exists():
            return HarnessResult(status="missing_report", **common)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        total = int(payload.get("total_instances", payload.get("total", 0)))
        submitted = int(
            payload.get("submitted_instances", payload.get("submitted", total))
        )
        completed = int(
            payload.get("completed_instances", payload.get("completed", submitted))
        )
        resolved_value = payload.get("resolved_instances", payload.get("resolved", 0))
        resolved = (
            len(resolved_value)
            if isinstance(resolved_value, list)
            else int(resolved_value)
        )
        return HarnessResult(
            status="completed",
            total=total,
            submitted=submitted,
            completed=completed,
            resolved=resolved,
            resolution_rate=resolved / submitted if submitted else 0.0,
            report_path=str(result_path),
            **common,
        )
