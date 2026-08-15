from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .embedding import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_MAX_SEQ_LENGTH,
    LocalEmbeddingEncoder,
)
from .reranker import (
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    LocalCrossEncoderReranker,
    blend_rankings,
)
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
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_revision: str = DEFAULT_EMBEDDING_REVISION,
        embedding_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        reranker: str = "none",
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_revision: str = DEFAULT_RERANKER_REVISION,
        reranker_max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        reranker_candidate_k: int = 40,
        reranker_weight: float = 0.5,
        reranker_strict: bool = False,
    ):
        if char_budget < 100:
            raise ValueError("context char budget must be at least 100")
        self.char_budget = char_budget
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy
        self.query_planning = query_planning
        self.structural_expansion = structural_expansion
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.embedding_max_seq_length = embedding_max_seq_length
        self.reranker = reranker
        self.reranker_model = reranker_model
        self.reranker_revision = reranker_revision
        self.reranker_max_length = reranker_max_length
        self.reranker_candidate_k = reranker_candidate_k
        self.reranker_weight = reranker_weight
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
                encoder=LocalEmbeddingEncoder(
                    model=self.embedding_model,
                    revision=self.embedding_revision,
                    max_seq_length=self.embedding_max_seq_length,
                ),
                query_planning=query_planning_enabled,
            )
        else:
            result = index.search(
                RetrievalQuery(text=query, top_k=self.top_k),
                query_planning=query_planning_enabled,
            )
        if self.reranker == "cross_encoder" and result.matches:
            try:
                reranked = LocalCrossEncoderReranker(
                    model=self.reranker_model,
                    revision=self.reranker_revision,
                    max_length=self.reranker_max_length,
                ).rank(
                    query,
                    [match.chunk for match in result.matches],
                    top_k=len(result.matches),
                )
            except RuntimeError:
                if self.reranker_strict:
                    raise
            else:
                original = result.matches
                reranked = blend_rankings(
                    base_scores=[match.score for match in original],
                    reranked=reranked,
                    reranker_weight=self.reranker_weight,
                    top_k=self.top_k,
                )
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
