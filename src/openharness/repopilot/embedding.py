from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import CodeChunk


class LocalEmbeddingEncoder:
    """Optional local BGE encoder kept outside the project runtime."""

    def __init__(self, *, cache_directory: Path | None = None) -> None:
        self.python = os.environ.get("REPOPILOT_EMBEDDING_PYTHON", r"E:\RepoPilot\embedding-runtime\Scripts\python.exe")
        self.cache = os.environ.get("REPOPILOT_EMBEDDING_CACHE", r"E:\RepoPilot\models\sentence-transformers")
        self.index_cache = cache_directory or Path(
            os.environ.get("REPOPILOT_EMBEDDING_INDEX_CACHE", r"E:\RepoPilot\models\repopilot-index")
        )
        self.last_stats: dict[str, int] = {}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        program = """import json,sys\nfrom sentence_transformers import SentenceTransformer\nm=SentenceTransformer('BAAI/bge-small-en-v1.5',cache_folder=sys.argv[1])\nprint(json.dumps(m.encode(json.load(sys.stdin),normalize_embeddings=True).tolist()))\n"""
        completed = subprocess.run([self.python, "-c", program, self.cache], input=json.dumps(texts), text=True, capture_output=True, check=True, timeout=180)
        return json.loads(completed.stdout)

    def rank(
        self,
        query: str,
        chunks: list[CodeChunk],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        return self.rank_many([query], chunks, top_k=top_k)

    def rank_many(
        self,
        queries: list[str] | tuple[str, ...],
        chunks: list[CodeChunk],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        selected_queries = [query.strip() for query in queries if query.strip()]
        if not selected_queries:
            raise ValueError("at least one dense retrieval query is required")
        if not chunks:
            self.last_stats = {"cache_hits": 0, "cache_misses": 0}
            return []
        max_seq_length = 128
        texts = [f"{chunk.path} {chunk.symbol}\n{chunk.text[:400]}" for chunk in chunks]
        embedding_ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        identity = hashlib.sha256(
            ("dense-content-v3-128\n" + "\n".join(embedding_ids)).encode()
        ).hexdigest()
        worker = Path(__file__).with_name("embedding_worker.py")
        payload = {
            "query": selected_queries[0],
            "queries": selected_queries,
            "texts": texts,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "embedding_ids": embedding_ids,
            "top_k": top_k,
            "max_seq_length": max_seq_length,
            "cache_file": str(self.index_cache / f"{identity}.npy"),
            "vector_store": str(self.index_cache / "embeddings-v2.sqlite3"),
            "model_key": "bge-small-en-v1.5-128-content-v3",
            "legacy_model_key": "bge-small-en-v1.5-128-v2",
            "model": "BAAI/bge-small-en-v1.5",
            "model_cache": self.cache,
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
            raise RuntimeError(f"local embedding worker failed: {detail}")
        response = json.loads(completed.stdout)
        self.last_stats = {
            "cache_hits": int(response["cache_hits"]),
            "cache_misses": int(response["cache_misses"]),
        }
        return [(int(index), float(score)) for index, score in response["ranked"]]
