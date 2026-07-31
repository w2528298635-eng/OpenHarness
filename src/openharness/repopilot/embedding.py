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
        texts = [f"{chunk.path} {chunk.symbol}\n{chunk.text[:1200]}" for chunk in chunks]
        identity = hashlib.sha256(
            "\n".join(chunk.chunk_id for chunk in chunks).encode()
        ).hexdigest()
        worker = Path(__file__).with_name("embedding_worker.py")
        payload = {
            "query": query,
            "texts": texts,
            "top_k": top_k,
            "cache_file": str(self.index_cache / f"{identity}.npy"),
            "model": "BAAI/bge-small-en-v1.5",
            "model_cache": self.cache,
        }
        completed = subprocess.run(
            [self.python, str(worker)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
            timeout=1800,
        )
        return [(int(index), float(score)) for index, score in json.loads(completed.stdout)]
