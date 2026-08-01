from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    payload = json.load(sys.stdin)
    pair_ids = payload["pair_ids"]
    score_store = Path(payload["score_store"])
    score_store.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(score_store)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS scores ("
        "model_key TEXT NOT NULL, pair_id TEXT NOT NULL, score REAL NOT NULL, "
        "PRIMARY KEY (model_key, pair_id))"
    )

    cached: dict[str, float] = {}
    for start in range(0, len(pair_ids), 800):
        batch = pair_ids[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT pair_id, score FROM scores "
            f"WHERE model_key = ? AND pair_id IN ({placeholders})",
            [payload["model_key"], *batch],
        )
        cached.update((pair_id, float(score)) for pair_id, score in rows)

    missing = [index for index, pair_id in enumerate(pair_ids) if pair_id not in cached]
    generated: dict[int, float] = {}
    if missing:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            payload["model"],
            revision=payload.get("revision"),
            cache_dir=payload["model_cache"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            payload["model"],
            revision=payload.get("revision"),
            cache_dir=payload["model_cache"],
        )
        model.eval()
        batch_size = int(payload.get("batch_size", 4))
        for start in range(0, len(missing), batch_size):
            batch_indices = missing[start : start + batch_size]
            inputs = tokenizer(
                [payload["query"] for _ in batch_indices],
                [payload["texts"][index] for index in batch_indices],
                padding=True,
                truncation=True,
                max_length=int(payload["max_length"]),
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**inputs).logits.reshape(-1).float().cpu().tolist()
            generated.update(
                (index, float(score))
                for index, score in zip(batch_indices, logits, strict=True)
            )
        connection.executemany(
            "INSERT OR REPLACE INTO scores(model_key, pair_id, score) VALUES (?, ?, ?)",
            [
                (payload["model_key"], pair_ids[index], generated[index])
                for index in missing
            ],
        )
        connection.commit()

    scores = [
        cached[pair_id] if pair_id in cached else generated[index]
        for index, pair_id in enumerate(pair_ids)
    ]
    count = min(int(payload["top_k"]), len(scores))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )[:count]
    print(
        json.dumps(
            {
                "ranked": [[index, scores[index]] for index in ranked_indices],
                "cache_hits": len(pair_ids) - len(missing),
                "cache_misses": len(missing),
            }
        )
    )
    connection.close()


if __name__ == "__main__":
    main()
