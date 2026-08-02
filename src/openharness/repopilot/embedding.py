from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import CodeChunk


DEFAULT_EMBEDDING_MODEL = "nomic-ai/CodeRankEmbed"
DEFAULT_EMBEDDING_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
DEFAULT_QUERY_PREFIX = "Represent this query for searching relevant code: "
DEFAULT_MAX_SEQ_LENGTH = 512
CODE_REPRESENTATION_VERSION = "code-context-v1"


def render_code_chunk(chunk: CodeChunk) -> str:
    """Render code together with the structural metadata used during retrieval."""
    lines = [f"File: {chunk.path}"]
    if chunk.symbol:
        lines.append(f"Symbol: {chunk.symbol}")
    lines.extend((f"Kind: {chunk.kind}", "Code:", chunk.text))
    return "\n".join(lines)


class LocalEmbeddingEncoder:
    """Optional local code-search encoder kept outside the project runtime."""

    def __init__(
        self,
        *,
        cache_directory: Path | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        revision: str = DEFAULT_EMBEDDING_REVISION,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        self.python = os.environ.get("REPOPILOT_EMBEDDING_PYTHON", r"E:\RepoPilot\embedding-runtime\Scripts\python.exe")
        self.cache = os.environ.get("REPOPILOT_EMBEDDING_CACHE", r"E:\RepoPilot\models\sentence-transformers")
        self.index_cache = cache_directory or Path(
            os.environ.get("REPOPILOT_EMBEDDING_INDEX_CACHE", r"E:\RepoPilot\models\repopilot-index")
        )
        self.model = model
        self.revision = revision
        self.query_prefix = os.environ.get("REPOPILOT_EMBEDDING_QUERY_PREFIX", query_prefix)
        self.max_seq_length = max_seq_length
        self.last_stats: dict[str, int] = {}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        program = """import json,sys\nfrom sentence_transformers import SentenceTransformer\nm=SentenceTransformer(sys.argv[2],revision=sys.argv[3],cache_folder=sys.argv[1],trust_remote_code=True)\nm.max_seq_length=int(sys.argv[4])\nprint(json.dumps(m.encode(json.load(sys.stdin),normalize_embeddings=True).tolist()))\n"""
        completed = subprocess.run(
            [
                self.python,
                "-c",
                program,
                self.cache,
                self.model,
                self.revision,
                str(self.max_seq_length),
            ],
            input=json.dumps(texts),
            text=True,
            capture_output=True,
            check=True,
            timeout=7200,
            env={**os.environ, "HF_HUB_CACHE": self.cache},
        )
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
        texts = [render_code_chunk(chunk) for chunk in chunks]
        embedding_ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        prefix_digest = hashlib.sha256(self.query_prefix.encode()).hexdigest()[:12]
        model_key = (
            f"{self.model}@{self.revision}-{self.max_seq_length}-"
            f"{CODE_REPRESENTATION_VERSION}-{prefix_digest}"
        )
        identity = hashlib.sha256(
            (model_key + "\n" + "\n".join(embedding_ids)).encode()
        ).hexdigest()
        worker = Path(__file__).with_name("embedding_worker.py")
        payload = {
            "query": selected_queries[0],
            "queries": selected_queries,
            "texts": texts,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "embedding_ids": embedding_ids,
            "top_k": top_k,
            "max_seq_length": self.max_seq_length,
            "cache_file": str(self.index_cache / f"{identity}.npy"),
            "vector_store": str(self.index_cache / "embeddings-v2.sqlite3"),
            "model_key": model_key,
            "model": self.model,
            "revision": self.revision,
            "query_prefix": self.query_prefix,
            "trust_remote_code": True,
            "model_cache": self.cache,
        }
        completed = subprocess.run(
            [self.python, str(worker)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=1800,
            env={**os.environ, "HF_HUB_CACHE": self.cache},
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
