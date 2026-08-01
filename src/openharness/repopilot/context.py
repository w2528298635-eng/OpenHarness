from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .embedding import LocalEmbeddingEncoder
from .reranker import LocalCrossEncoderReranker
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
    def __init__(
        self,
        *,
        char_budget: int = 12_000,
        top_k: int = 12,
        retrieval_strategy: str = "lexical",
        query_planning: bool = True,
        structural_expansion: bool = False,
        reranker: str = "none",
        reranker_candidate_k: int = 40,
        reranker_strict: bool = False,
    ):
        if char_budget < 100:
            raise ValueError("context char budget must be at least 100")
        self.char_budget = char_budget
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy
        self.query_planning = query_planning
        self.structural_expansion = structural_expansion
        self.reranker = reranker
        self.reranker_candidate_k = reranker_candidate_k
        self.reranker_strict = reranker_strict

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
        hybrid_enabled = (
            self.retrieval_strategy == "hybrid"
            or os.environ.get("REPOPILOT_HYBRID_RETRIEVAL") == "1"
        )
        query_planning_enabled = (
            self.query_planning
            and os.environ.get("REPOPILOT_QUERY_PLANNING", "1") != "0"
        )
        if hybrid_enabled:
            retrieval_top_k = (
                max(self.top_k, self.reranker_candidate_k)
                if self.reranker == "cross_encoder"
                else self.top_k
            )
            result = index.hybrid_search(
                RetrievalQuery(text=query, top_k=retrieval_top_k),
                encoder=LocalEmbeddingEncoder(),
                query_planning=query_planning_enabled,
            )
        else:
            result = index.search(
                RetrievalQuery(text=query, top_k=self.top_k),
                query_planning=query_planning_enabled,
            )
        if self.reranker == "cross_encoder" and result.matches:
            try:
                reranked = LocalCrossEncoderReranker().rank(
                    query,
                    [match.chunk for match in result.matches],
                    top_k=self.top_k,
                )
            except RuntimeError:
                if self.reranker_strict:
                    raise
            else:
                original = result.matches
                result.matches = [
                    ScoredChunk(
                        chunk=original[index].chunk,
                        score=score,
                        reasons=[*original[index].reasons, "reranker"],
                    )
                    for index, score in reranked
                ]
        retrieval_candidates = [
            (match, "+".join(match.reasons) or "lexical")
            for match in result.matches
        ]
        matched_ids = {match.chunk.chunk_id for match in result.matches}
        structural = (
            index.expand_structure(
                [match.chunk for match in result.matches],
                limit=max(self.top_k * 2, self.top_k + 4),
            )
            if self.structural_expansion
            and os.environ.get("REPOPILOT_STRUCTURAL_EXPANSION", "1") != "0"
            else []
        )
        seed_score = min((match.score for match in result.matches), default=0.0)
        structural_candidates = [
            (
                ScoredChunk(chunk=chunk, score=seed_score * 0.5, reasons=["structure"]),
                "structure",
            )
            for chunk in structural
            if chunk.chunk_id not in matched_ids
        ]
        structural_iterator = iter(structural_candidates)
        for position, candidate in enumerate(retrieval_candidates):
            candidates.append(candidate)
            if position == 0 or (position + 1) % 3 == 0:
                neighbor = next(structural_iterator, None)
                if neighbor is not None:
                    candidates.append(neighbor)
        candidates.extend(structural_iterator)

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
