from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .context import ContextBuilder, ContextSelection
from .models import Phase, RepoRunState, RepoTaskSpec
from .retrieval import RepositoryIndex
from .store import RunStore
from .workflow import PhaseResult, RunContext, WorkflowDefinition, WorkflowRuntime

INSIGHT_PROMPT_VERSION = "insight-1"


class InsightRequest(BaseModel):
    repo_path: Path
    question: str
    max_findings: int = Field(default=5, ge=1, le=20)
    context_char_budget: int = Field(default=12_000, ge=500)

    @field_validator("question")
    @classmethod
    def question_is_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class InsightCitation(BaseModel):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    chunk_id: str


class InsightFinding(BaseModel):
    title: str
    summary: str
    citations: list[InsightCitation]


class InsightReport(BaseModel):
    run_id: str
    question: str
    generated_at: datetime
    prompt_version: str
    context_ids: list[str]
    findings: list[InsightFinding]


HandlerFunction = Callable[[RunContext], Awaitable[PhaseResult]]


class _Handler:
    def __init__(self, function: HandlerFunction):
        self.function = function

    async def handle(self, context: RunContext) -> PhaseResult:
        return await self.function(context)


class RepositoryInsightWorkflow:
    """Read-only repository Q&A built on the shared workflow runtime."""

    def __init__(self, *, store: RunStore):
        self.store = store
        self._index: RepositoryIndex | None = None
        self._selection: ContextSelection | None = None
        self._report: InsightReport | None = None
        self._request: InsightRequest | None = None

    async def run(self, request: InsightRequest) -> InsightReport:
        root = request.repo_path.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository directory does not exist: {root}")
        request = request.model_copy(update={"repo_path": root})
        self._request = request
        run_id = "insight-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        task = RepoTaskSpec(
            repo_path=root,
            issue=request.question,
            verify_command=["pytest"],
        )
        state = RepoRunState(
            run_id=run_id,
            task=task,
            phase=Phase.INSIGHT_SCAN,
            original_repo=root,
            worktree_path=root,
        )
        self.store.create(state)
        definition = WorkflowDefinition(
            name="repository-insight",
            version="1",
            initial_phase=Phase.INSIGHT_SCAN,
            terminal_phases=frozenset({Phase.COMPLETE, Phase.FAILED}),
            handlers={
                Phase.INSIGHT_SCAN: _Handler(self._scan),
                Phase.INSIGHT_RETRIEVE: _Handler(self._retrieve),
                Phase.INSIGHT_ANALYZE: _Handler(self._analyze),
                Phase.INSIGHT_REPORT: _Handler(self._render),
            },
        )
        runtime = WorkflowRuntime(
            definition=definition,
            create_state=lambda _: state,
            checkpoint=self.store.save_state,
            emit=self.store.append_event,
        )
        completed = await runtime.start(task)
        if completed.phase is not Phase.COMPLETE or self._report is None:
            raise RuntimeError(completed.terminal_reason or "insight workflow failed")
        self.store.write_json(run_id, "insight.json", self._report)
        self.store.write_text(run_id, "insight.md", self._markdown(self._report))
        return self._report

    async def _scan(self, context: RunContext) -> PhaseResult:
        self._index = RepositoryIndex.build(context.state.task.repo_path)
        return PhaseResult(
            next_phase=Phase.INSIGHT_RETRIEVE,
            detail=f"indexed {len(self._index.chunks)} chunks",
        )

    async def _retrieve(self, context: RunContext) -> PhaseResult:
        assert self._index is not None and self._request is not None
        self._selection = ContextBuilder(
            char_budget=self._request.context_char_budget,
            top_k=max(self._request.max_findings * 2, 5),
        ).build(index=self._index, query=self._request.question)
        self.store.write_json(
            context.state.run_id,
            "insight-context.json",
            self._selection,
        )
        return PhaseResult(
            next_phase=Phase.INSIGHT_ANALYZE,
            detail=f"selected {len(self._selection.selected_chunks)} chunks",
        )

    async def _analyze(self, context: RunContext) -> PhaseResult:
        assert self._selection is not None and self._request is not None
        findings = []
        for selected in self._selection.selected_chunks[: self._request.max_findings]:
            chunk = selected.chunk
            first_line = next(
                (line.strip() for line in chunk.text.splitlines() if line.strip()),
                "Repository evidence",
            )
            findings.append(
                InsightFinding(
                    title=chunk.symbol or chunk.path,
                    summary=(
                        f"Relevant source evidence: {first_line[:240]} "
                        f"(retrieval reason: {selected.reason})."
                    ),
                    citations=[
                        InsightCitation(
                            path=chunk.path,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            chunk_id=chunk.chunk_id,
                        )
                    ],
                )
            )
        self._report = InsightReport(
            run_id=context.state.run_id,
            question=self._request.question,
            generated_at=datetime.now(UTC),
            prompt_version=INSIGHT_PROMPT_VERSION,
            context_ids=[item.chunk.chunk_id for item in self._selection.selected_chunks],
            findings=findings,
        )
        return PhaseResult(
            next_phase=Phase.INSIGHT_REPORT,
            detail=f"created {len(findings)} cited findings",
        )

    async def _render(self, context: RunContext) -> PhaseResult:
        assert self._report is not None
        return PhaseResult(
            next_phase=Phase.COMPLETE,
            detail="insight report persisted",
        )

    @staticmethod
    def _markdown(report: InsightReport) -> str:
        lines = [
            "# Repository Insight",
            "",
            f"Question: {report.question}",
            "",
        ]
        for finding in report.findings:
            lines.extend([f"## {finding.title}", "", finding.summary, ""])
            for citation in finding.citations:
                lines.append(f"- `{citation.path}:{citation.start_line}-{citation.end_line}`")
            lines.append("")
        return "\n".join(lines)
