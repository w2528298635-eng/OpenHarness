from __future__ import annotations

import json
import os
import subprocess


class LocalEmbeddingEncoder:
    """Optional local BGE encoder kept outside the project runtime."""

    def __init__(self) -> None:
        self.python = os.environ.get("REPOPILOT_EMBEDDING_PYTHON", r"E:\RepoPilot\embedding-runtime\Scripts\python.exe")
        self.cache = os.environ.get("REPOPILOT_EMBEDDING_CACHE", r"E:\RepoPilot\models\sentence-transformers")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        program = """import json,sys\nfrom sentence_transformers import SentenceTransformer\nm=SentenceTransformer('BAAI/bge-small-en-v1.5',cache_folder=sys.argv[1])\nprint(json.dumps(m.encode(json.load(sys.stdin),normalize_embeddings=True).tolist()))\n"""
        completed = subprocess.run([self.python, "-c", program, self.cache], input=json.dumps(texts), text=True, capture_output=True, check=True, timeout=180)
        return json.loads(completed.stdout)
