from __future__ import annotations

import ast
import fnmatch
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]+")
_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".openharness",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
    }
)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN.finditer(text)]


class CodeChunk(BaseModel):
    chunk_id: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol: str = ""
    kind: str
    text: str


class RetrievalQuery(BaseModel):
    text: str
    top_k: int = Field(default=8, ge=1, le=100)


class ScoredChunk(BaseModel):
    chunk: CodeChunk
    score: float
    reasons: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: RetrievalQuery
    matches: list[ScoredChunk]
    indexed_chunks: int


class RepositoryIndex(BaseModel):
    root: Path
    chunks: list[CodeChunk]

    @classmethod
    def build(
        cls,
        root: Path,
        *,
        allowed_paths: list[str] | None = None,
        max_file_bytes: int = 200_000,
        max_chunk_chars: int = 4000,
    ) -> RepositoryIndex:
        resolved = root.resolve()
        chunks: list[CodeChunk] = []
        for path in sorted(resolved.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(resolved).as_posix()
            if any(part in _IGNORED_DIRECTORIES for part in path.relative_to(resolved).parts):
                continue
            if allowed_paths is not None and not any(
                fnmatch.fnmatchcase(relative, pattern) for pattern in allowed_paths
            ):
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "\x00" in text:
                continue
            chunks.extend(
                _chunk_file(
                    relative,
                    text,
                    max_chunk_chars=max_chunk_chars,
                )
            )
        return cls(root=resolved, chunks=chunks)

    def search(self, query: RetrievalQuery) -> RetrievalResult:
        query_terms = _tokens(query.text)
        if not query_terms or not self.chunks:
            return RetrievalResult(
                query=query,
                matches=[],
                indexed_chunks=len(self.chunks),
            )
        documents = [_tokens(f"{chunk.path} {chunk.symbol} {chunk.text}") for chunk in self.chunks]
        document_frequency = Counter(term for terms in documents for term in set(terms))
        total_documents = len(documents)
        scored: list[ScoredChunk] = []
        query_folded = query.text.casefold()
        for chunk, terms in zip(self.chunks, documents, strict=True):
            counts = Counter(terms)
            score = 0.0
            reasons: list[str] = []
            for term in query_terms:
                if counts[term]:
                    inverse_frequency = (
                        math.log((total_documents + 1) / (document_frequency[term] + 1)) + 1
                    )
                    score += (1 + math.log(counts[term])) * inverse_frequency
            if chunk.symbol and chunk.symbol.casefold() in query_folded:
                score += 4.0
                reasons.append("symbol")
            path_terms = set(_tokens(chunk.path))
            if any(term in path_terms for term in query_terms):
                score += 2.0
                reasons.append("path")
            if score > 0:
                if not reasons:
                    reasons.append("lexical")
                scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=round(score, 6),
                        reasons=reasons,
                    )
                )
        scored.sort(
            key=lambda item: (
                -item.score,
                item.chunk.path,
                item.chunk.start_line,
                item.chunk.chunk_id,
            )
        )
        return RetrievalResult(
            query=query,
            matches=scored[: query.top_k],
            indexed_chunks=len(self.chunks),
        )

    def hybrid_search(self, query: RetrievalQuery, *, encoder) -> RetrievalResult:
        """Fuse lexical relevance with locally computed cosine similarity."""
        lexical = self.search(RetrievalQuery(text=query.text, top_k=len(self.chunks)))
        query_vector, *chunk_vectors = encoder(
            [query.text, *(f"{chunk.path} {chunk.symbol} {chunk.text}" for chunk in self.chunks)]
        )
        lexical_scores = {item.chunk.chunk_id: item.score for item in lexical.matches}
        scored: list[ScoredChunk] = []
        for chunk, vector in zip(self.chunks, chunk_vectors, strict=True):
            numerator = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            semantic = numerator / (query_norm * vector_norm) if query_norm and vector_norm else 0.0
            lexical_score = lexical_scores.get(chunk.chunk_id, 0.0)
            score = lexical_score + semantic * 10
            if score > 0:
                reasons = ["semantic"] if semantic > 0 else []
                if lexical_score:
                    reasons.append("lexical")
                scored.append(ScoredChunk(chunk=chunk, score=round(score, 6), reasons=reasons))
        scored.sort(key=lambda item: (-item.score, item.chunk.path, item.chunk.start_line))
        return RetrievalResult(query=query, matches=scored[: query.top_k], indexed_chunks=len(self.chunks))


def _chunk_file(path: str, text: str, *, max_chunk_chars: int) -> list[CodeChunk]:
    lines = text.splitlines(keepends=True)
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return _text_chunks(path, lines, max_chunk_chars=max_chunk_chars)
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if nodes:
            chunks: list[CodeChunk] = []
            first_line = min(node.lineno for node in nodes)
            if first_line > 1:
                chunks.extend(
                    _make_bounded_chunks(
                        path,
                        lines[: first_line - 1],
                        start_line=1,
                        symbol="",
                        kind="module",
                        max_chunk_chars=max_chunk_chars,
                    )
                )
            for node in nodes:
                end_line = getattr(node, "end_lineno", node.lineno)
                chunks.extend(
                    _make_bounded_chunks(
                        path,
                        lines[node.lineno - 1 : end_line],
                        start_line=node.lineno,
                        symbol=node.name,
                        kind=type(node).__name__.casefold(),
                        max_chunk_chars=max_chunk_chars,
                    )
                )
            return chunks
    return _text_chunks(path, lines, max_chunk_chars=max_chunk_chars)


def _text_chunks(
    path: str,
    lines: list[str],
    *,
    max_chunk_chars: int,
) -> list[CodeChunk]:
    return _make_bounded_chunks(
        path,
        lines,
        start_line=1,
        symbol="",
        kind="text",
        max_chunk_chars=max_chunk_chars,
    )


def _make_bounded_chunks(
    path: str,
    lines: list[str],
    *,
    start_line: int,
    symbol: str,
    kind: str,
    max_chunk_chars: int,
) -> list[CodeChunk]:
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    current: list[str] = []
    current_chars = 0
    current_start = start_line
    for offset, line in enumerate(lines):
        if current and current_chars + len(line) > max_chunk_chars:
            chunks.append(
                _make_chunk(
                    path,
                    current,
                    start_line=current_start,
                    symbol=symbol,
                    kind=kind,
                )
            )
            current = []
            current_chars = 0
            current_start = start_line + offset
        if len(line) > max_chunk_chars:
            line = line[:max_chunk_chars]
        current.append(line)
        current_chars += len(line)
    if current:
        chunks.append(
            _make_chunk(
                path,
                current,
                start_line=current_start,
                symbol=symbol,
                kind=kind,
            )
        )
    return chunks


def _make_chunk(
    path: str,
    lines: list[str],
    *,
    start_line: int,
    symbol: str,
    kind: str,
) -> CodeChunk:
    text = "".join(lines)
    end_line = start_line + max(0, len(lines) - 1)
    identity = f"{path}:{start_line}:{end_line}:{symbol}:{text}".encode()
    return CodeChunk(
        chunk_id=hashlib.sha256(identity).hexdigest()[:20],
        path=path,
        start_line=start_line,
        end_line=end_line,
        symbol=symbol,
        kind=kind,
        text=text,
    )
