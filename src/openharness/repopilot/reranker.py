from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .embedding import CODE_REPRESENTATION_VERSION, render_code_chunk

if TYPE_CHECKING:
    from .retrieval import CodeChunk


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_RERANKER_MAX_LENGTH = 512


class LocalCrossEncoderReranker:
    """Reorder an already-retrieved candidate set with a local cross-encoder."""

    def __init__(
        self,
        *,
        cache_directory: Path | None = None,
        model: str = DEFAULT_RERANKER_MODEL,
        revision: str = DEFAULT_RERANKER_REVISION,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        batch_size: int = 4,
    ) -> None:
        self.python = os.environ.get(
            "REPOPILOT_RERANKER_PYTHON",
            os.environ.get(
                "REPOPILOT_EMBEDDING_PYTHON",
                r"E:\RepoPilot\embedding-runtime\Scripts\python.exe",
            ),
        )
        self.model_cache = os.environ.get(
            "REPOPILOT_RERANKER_CACHE",
            os.environ.get(
                "REPOPILOT_EMBEDDING_CACHE",
                r"E:\RepoPilot\models\sentence-transformers",
            ),
        )
        self.index_cache = cache_directory or Path(
            os.environ.get(
                "REPOPILOT_RERANKER_INDEX_CACHE",
                r"E:\RepoPilot\models\repopilot-index",
            )
        )
        self.model = os.environ.get("REPOPILOT_RERANKER_MODEL", model)
        self.revision = os.environ.get("REPOPILOT_RERANKER_REVISION", revision)
        self.max_length = max_length
        self.batch_size = batch_size
        self.last_stats: dict[str, int] = {}

    def rank(
        self,
        query: str,
        candidates: list[CodeChunk],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        selected_query = query.strip()
        if not selected_query:
            raise ValueError("a reranking query is required")
        if not candidates:
            self.last_stats = {"cache_hits": 0, "cache_misses": 0}
            return []

        texts = [render_code_chunk(chunk) for chunk in candidates]
        pair_ids = [
            hashlib.sha256(f"{selected_query}\n{text}".encode()).hexdigest()
            for text in texts
        ]
        model_key = (
            f"{self.model}@{self.revision}-{self.max_length}-"
            f"{CODE_REPRESENTATION_VERSION}"
        )
        worker = Path(__file__).with_name("reranker_worker.py")
        payload = {
            "query": selected_query,
            "texts": texts,
            "candidate_ids": [chunk.chunk_id for chunk in candidates],
            "pair_ids": pair_ids,
            "top_k": top_k,
            "model": self.model,
            "revision": self.revision,
            "model_key": model_key,
            "model_cache": self.model_cache,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "score_store": str(self.index_cache / "reranker-v1.sqlite3"),
        }
        completed = subprocess.run(
            [self.python, str(worker)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=1800,
        )
        if getattr(completed, "returncode", 0):
            detail = completed.stderr.strip() or "worker returned no stderr"
            raise RuntimeError(f"local reranker worker failed: {detail}")
        response = json.loads(completed.stdout)
        self.last_stats = {
            "cache_hits": int(response["cache_hits"]),
            "cache_misses": int(response["cache_misses"]),
        }
        return [(int(index), float(score)) for index, score in response["ranked"]]
