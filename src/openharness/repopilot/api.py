from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised by core-only installations
    raise ImportError(
        "RepoPilot API dependencies are not installed; install OpenHarness with the 'api' extra"
    ) from exc

from .service import RepoPilotService, UnknownRunError
from .task_loader import load_task


class RunSubmission(BaseModel):
    task_file: Path


def create_app(service: RepoPilotService | None = None) -> FastAPI:
    if service is None:
        from .cli import _scheduler

        service = RepoPilotService(_scheduler)
    app = FastAPI(title="RepoPilot API", version="1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=202)
    async def start_run(request: RunSubmission) -> dict:
        try:
            spec = load_task(request.task_file)
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return service.submit(spec).model_dump(mode="json")

    @app.get("/operations/{operation_id}")
    async def get_operation(operation_id: str) -> dict:
        try:
            return service.operation(operation_id).model_dump(mode="json")
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown operation") from exc

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, repo: Path | None = None) -> dict:
        try:
            return service.get(run_id, repo).model_dump(mode="json")
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc

    @app.get("/runs/{run_id}/events")
    async def get_events(run_id: str, repo: Path | None = None) -> list[dict]:
        try:
            return [event.model_dump(mode="json") for event in service.events(run_id, repo)]
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc

    @app.get("/runs/{run_id}/artifacts")
    async def list_artifacts(run_id: str, repo: Path | None = None) -> list[str]:
        try:
            return service.artifacts(run_id, repo)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc

    @app.get("/runs/{run_id}/artifacts/{name}")
    async def get_artifact(
        run_id: str,
        name: str,
        repo: Path | None = None,
    ) -> FileResponse:
        try:
            return FileResponse(service.artifact(run_id, name, repo))
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown artifact") from exc

    @app.post("/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: str) -> dict[str, str]:
        try:
            service.cancel(run_id)
        except UnknownRunError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        return {"run_id": run_id, "status": "cancellation_requested"}

    return app
