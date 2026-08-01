from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def _encode_missing_in_batches(
    *,
    missing,
    texts,
    embedding_ids,
    model,
    batch_size: int,
    persist,
):
    """Encode and durably persist each batch so an interrupted run can resume."""
    generated = {}
    for start in range(0, len(missing), batch_size):
        batch_indices = missing[start : start + batch_size]
        values = model.encode(
            [texts[index] for index in batch_indices],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        persist([embedding_ids[index] for index in batch_indices], values)
        generated.update(zip(batch_indices, values, strict=True))
    return generated


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    payload = json.load(sys.stdin)
    cache_file = Path(payload["cache_file"])
    model = SentenceTransformer(
        payload["model"],
        revision=payload.get("revision"),
        cache_folder=payload["model_cache"],
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
    )
    model.max_seq_length = int(payload["max_seq_length"])
    texts = payload["texts"]
    chunk_ids = payload["chunk_ids"]
    embedding_ids = payload.get("embedding_ids", chunk_ids)
    vector_store = Path(payload["vector_store"])
    vector_store.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(vector_store)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "model_key TEXT NOT NULL, chunk_id TEXT NOT NULL, vector BLOB NOT NULL, "
        "PRIMARY KEY (model_key, chunk_id))"
    )

    def persist(ids: list[str], values: np.ndarray) -> None:
        connection.executemany(
            "INSERT OR IGNORE INTO embeddings(model_key, chunk_id, vector) VALUES (?, ?, ?)",
            [
                (payload["model_key"], embedding_id, value.astype(np.float32).tobytes())
                for embedding_id, value in zip(ids, values, strict=True)
            ],
        )
        connection.commit()

    def load_cached(model_key: str, ids: list[str]) -> dict[str, np.ndarray]:
        cached: dict[str, np.ndarray] = {}
        for start in range(0, len(ids), 800):
            batch = ids[start : start + 800]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"SELECT chunk_id, vector FROM embeddings "
                f"WHERE model_key = ? AND chunk_id IN ({placeholders})",
                [model_key, *batch],
            )
            cached.update(
                (cache_id, np.frombuffer(vector, dtype=np.float32))
                for cache_id, vector in rows
            )
        return cached

    if cache_file.exists():
        embeddings = np.load(cache_file)
        # This exact repository vector matrix was persisted when it was built.
        # Re-inserting every row here turns a cache hit into thousands of no-op
        # SQLite writes and is unnecessary for ranking or cross-revision reuse.
        cache_hits = len(embedding_ids)
        cache_misses = 0
    else:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached = load_cached(payload["model_key"], embedding_ids)
        unresolved = [
            index
            for index, embedding_id in enumerate(embedding_ids)
            if embedding_id not in cached
        ]
        legacy_model_key = payload.get("legacy_model_key")
        if unresolved and legacy_model_key:
            legacy_ids = [chunk_ids[index] for index in unresolved]
            legacy = load_cached(legacy_model_key, legacy_ids)
            for index in unresolved:
                legacy_value = legacy.get(chunk_ids[index])
                if legacy_value is not None:
                    cached[embedding_ids[index]] = legacy_value
        missing = [
            index
            for index, embedding_id in enumerate(embedding_ids)
            if embedding_id not in cached
        ]
        cache_hits = len(embedding_ids) - len(missing)
        cache_misses = len(missing)
        generated_by_index = _encode_missing_in_batches(
            missing=missing,
            texts=texts,
            embedding_ids=embedding_ids,
            model=model,
            batch_size=int(payload.get("batch_size", 32)),
            persist=persist,
        )
        embeddings = np.stack(
            [
                cached[embedding_id]
                if embedding_id in cached
                else generated_by_index[index]
                for index, embedding_id in enumerate(embedding_ids)
            ]
        )
        temporary = cache_file.with_suffix(".tmp.npy")
        np.save(temporary, embeddings)
        temporary.replace(cache_file)
    queries = payload.get("queries", [payload["query"]])
    query_vectors = model.encode(
        [
            payload.get("query_prefix", "") + query
            for query in queries
        ],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = (embeddings @ query_vectors.T).max(axis=1)
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
