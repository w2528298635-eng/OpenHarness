from __future__ import annotations

from pydantic import BaseModel, Field

from .embedding import LocalEmbeddingEncoder
from .retrieval import CodeChunk, RepositoryIndex, RetrievalQuery, ScoredChunk


class SelectedContextChunk(BaseModel):
    chunk: CodeChunk
    score: float
    reason: str


class ContextSelection(BaseModel):
    query: str
    rendered: str
    total_chars: int
    char_budget: int
    selected_chunks: list[SelectedContextChunk] = Field(default_factory=list)


class ContextBuilder:
    def __init__(self, *, char_budget: int = 12_000, top_k: int = 12):
        if char_budget < 100:
            raise ValueError("context char budget must be at least 100")
        self.char_budget = char_budget
        self.top_k = top_k

    def build(
        self,
        *,
        index: RepositoryIndex,
        query: str,
        failure_text: str = "",
        priority_paths: list[str] | None = None,
    ) -> ContextSelection:
        parts: list[str] = []
        remaining = self.char_budget
        if failure_text.strip():
            failure = f"### Verification failure\n{failure_text.strip()}\n\n"
            failure = failure[:remaining]
            parts.append(failure)
            remaining -= len(failure)

        candidates: list[tuple[ScoredChunk, str]] = []
        priority = list(dict.fromkeys(priority_paths or []))
        for priority_path in priority:
            for chunk in index.chunks:
                if chunk.path == priority_path:
                    candidates.append(
                        (
                            ScoredChunk(
                                chunk=chunk,
                                score=1000.0,
                                reasons=["priority_path"],
                            ),
                            "priority_path",
                        )
                    )
        if __import__("os").environ.get("REPOPILOT_HYBRID_RETRIEVAL") == "1":
            result = index.hybrid_search(RetrievalQuery(text=query, top_k=self.top_k), encoder=LocalEmbeddingEncoder())
        else:
            result = index.search(RetrievalQuery(text=query, top_k=self.top_k))
        candidates.extend((match, "+".join(match.reasons) or "lexical") for match in result.matches)

        selected: list[SelectedContextChunk] = []
        seen: set[str] = set()
        for match, reason in candidates:
            if match.chunk.chunk_id in seen or remaining <= 0:
                continue
            header = f"### {match.chunk.path}:{match.chunk.start_line}-{match.chunk.end_line}"
            if match.chunk.symbol:
                header += f" ({match.chunk.symbol})"
            block = f"{header}\n{match.chunk.text.rstrip()}\n\n"
            if len(block) > remaining:
                if remaining < len(header) + 20:
                    continue
                block = block[: remaining - len("\n...[TRUNCATED]")] + "\n...[TRUNCATED]"
            parts.append(block)
            remaining -= len(block)
            seen.add(match.chunk.chunk_id)
            selected.append(
                SelectedContextChunk(
                    chunk=match.chunk,
                    score=match.score,
                    reason=reason,
                )
            )
        rendered = "".join(parts)
        return ContextSelection(
            query=query,
            rendered=rendered,
            total_chars=len(rendered),
            char_budget=self.char_budget,
            selected_chunks=selected,
        )
