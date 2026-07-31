from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    payload = json.load(sys.stdin)
    cache_file = Path(payload["cache_file"])
    model = SentenceTransformer(payload["model"], cache_folder=payload["model_cache"])
    model.max_seq_length = int(payload["max_seq_length"])
    texts = payload["texts"]
    chunk_ids = payload["chunk_ids"]
    vector_store = Path(payload["vector_store"])
    vector_store.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(vector_store)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "model_key TEXT NOT NULL, chunk_id TEXT NOT NULL, vector BLOB NOT NULL, "
        "PRIMARY KEY (model_key, chunk_id))"
    )

    def persist(values: np.ndarray) -> None:
        connection.executemany(
            "INSERT OR IGNORE INTO embeddings(model_key, chunk_id, vector) VALUES (?, ?, ?)",
            [
                (payload["model_key"], chunk_id, value.astype(np.float32).tobytes())
                for chunk_id, value in zip(chunk_ids, values, strict=True)
            ],
        )
        connection.commit()

    if cache_file.exists():
        embeddings = np.load(cache_file)
        persist(embeddings)
        cache_hits = len(chunk_ids)
        cache_misses = 0
    else:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached: dict[str, np.ndarray] = {}
        for start in range(0, len(chunk_ids), 800):
            batch = chunk_ids[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"SELECT chunk_id, vector FROM embeddings "
                f"WHERE model_key = ? AND chunk_id IN ({placeholders})",
                [payload["model_key"], *batch],
            )
            cached.update(
                (chunk_id, np.frombuffer(vector, dtype=np.float32))
                for chunk_id, vector in rows
            )
        missing = [index for index, chunk_id in enumerate(chunk_ids) if chunk_id not in cached]
        cache_hits = len(cached)
        cache_misses = len(missing)
        generated = model.encode(
            [texts[index] for index in missing],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        ) if missing else np.empty((0, 384), dtype=np.float32)
        generated_by_index = dict(zip(missing, generated, strict=True))
        embeddings = np.stack(
            [
                cached[chunk_id] if chunk_id in cached else generated_by_index[index]
                for index, chunk_id in enumerate(chunk_ids)
            ]
        )
        persist(embeddings)
        temporary = cache_file.with_suffix(".tmp.npy")
        np.save(temporary, embeddings)
        temporary.replace(cache_file)
    query = model.encode(
        ["Represent this sentence for searching relevant passages: " + payload["query"]],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = embeddings @ query
    count = min(int(payload["top_k"]), len(scores))
    indices = np.argsort(-scores)[:count]
    print(
        json.dumps(
            {
                "ranked": [[int(index), float(scores[index])] for index in indices],
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }
        )
    )
    connection.close()


if __name__ == "__main__":
    main()
