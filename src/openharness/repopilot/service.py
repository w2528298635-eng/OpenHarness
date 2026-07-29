from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .events import RunEvent
from .models import RepoRunState, RepoTaskSpec
from .store import RunStore

SchedulerFactory = Callable[[Path, float], Any]
StoreFactory = Callable[[Path], RunStore]


class UnknownRunError(LookupError):
    pass


class OperationState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunOperation(BaseModel):
    operation_id: str
    status: OperationState
    run_id: str | None = None
    error: str | None = None


class RepoPilotService:
    """Application boundary shared by terminal and HTTP adapters."""

    def __init__(
        self,
        scheduler_factory: SchedulerFactory,
        *,
        store_factory: StoreFactory = RunStore,
    ):
        self.scheduler_factory = scheduler_factory
        self.store_factory = store_factory
        self._states: dict[str, RepoRunState] = {}
        self._schedulers: dict[str, Any] = {}
        self._repositories: dict[str, Path] = {}
        self._operations: dict[str, RunOperation] = {}
        self._operation_tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self, spec: RepoTaskSpec) -> RepoRunState:
        scheduler = self.scheduler_factory(
            spec.repo_path,
            spec.budgets.verify_timeout_seconds,
        )
        state = await scheduler.start(spec)
        self._remember(state, scheduler)
        return state

    def submit(self, spec: RepoTaskSpec) -> RunOperation:
        operation_id = uuid4().hex
        operation = RunOperation(
            operation_id=operation_id,
            status=OperationState.ACCEPTED,
        )
        self._operations[operation_id] = operation
        self._operation_tasks[operation_id] = asyncio.create_task(
            self._run_operation(operation_id, spec)
        )
        return operation

    def operation(self, operation_id: str) -> RunOperation:
        try:
            return self._operations[operation_id]
        except KeyError as exc:
            raise UnknownRunError(operation_id) from exc

    async def resume(self, run_id: str, repo: Path) -> RepoRunState:
        store = self.store_factory(repo)
        persisted = store.load_state(run_id)
        scheduler = self.scheduler_factory(
            repo,
            persisted.task.budgets.verify_timeout_seconds,
        )
        self._schedulers[run_id] = scheduler
        state = await scheduler.resume(run_id)
        self._remember(state, scheduler)
        return state

    def get(self, run_id: str, repo: Path | None = None) -> RepoRunState:
        if run_id in self._states:
            return self._states[run_id]
        store = self._store_for(run_id, repo)
        try:
            state = store.load_state(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise UnknownRunError(run_id) from exc
        self._states[run_id] = state
        self._repositories[run_id] = store.repo_root
        return state

    def cancel(self, run_id: str) -> None:
        scheduler = self._schedulers.get(run_id)
        if scheduler is None:
            raise UnknownRunError(run_id)
        scheduler.request_cancel()

    async def _run_operation(self, operation_id: str, spec: RepoTaskSpec) -> None:
        scheduler = self.scheduler_factory(
            spec.repo_path,
            spec.budgets.verify_timeout_seconds,
        )
        self._schedulers[operation_id] = scheduler
        self._operations[operation_id] = RunOperation(
            operation_id=operation_id,
            status=OperationState.RUNNING,
        )
        try:
            state = await scheduler.start(spec)
        except Exception as exc:  # noqa: BLE001 - operation boundary records provider failures
            self._operations[operation_id] = RunOperation(
                operation_id=operation_id,
                status=OperationState.FAILED,
                error=str(exc)[:1000],
            )
            return
        self._remember(state, scheduler)
        self._operations[operation_id] = RunOperation(
            operation_id=operation_id,
            status=OperationState.COMPLETED,
            run_id=state.run_id,
        )

    def events(self, run_id: str, repo: Path | None = None) -> list[RunEvent]:
        self.get(run_id, repo)
        return self._store_for(run_id, repo).load_events(run_id)

    def artifact(self, run_id: str, name: str, repo: Path | None = None) -> Path:
        if (
            not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError("artifact name must be one safe file name")
        self.get(run_id, repo)
        run_dir = self._store_for(run_id, repo).run_dir(run_id).resolve()
        target = (run_dir / name).resolve()
        if target.parent != run_dir or not target.is_file():
            raise FileNotFoundError(name)
        return target

    def artifacts(self, run_id: str, repo: Path | None = None) -> list[str]:
        self.get(run_id, repo)
        run_dir = self._store_for(run_id, repo).run_dir(run_id)
        return sorted(path.name for path in run_dir.iterdir() if path.is_file())

    def _remember(self, state: RepoRunState, scheduler: Any) -> None:
        self._states[state.run_id] = state
        self._schedulers[state.run_id] = scheduler
        self._repositories[state.run_id] = state.task.repo_path.resolve()

    def _store_for(self, run_id: str, repo: Path | None) -> RunStore:
        root = repo or self._repositories.get(run_id)
        if root is None:
            raise UnknownRunError(run_id)
        return self.store_factory(Path(root))
