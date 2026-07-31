from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    payload = json.load(sys.stdin)
    cache_file = Path(payload["cache_file"])
    model = SentenceTransformer(payload["model"], cache_folder=payload["model_cache"])
    texts = payload["texts"]
    if cache_file.exists():
        embeddings = np.load(cache_file)
    else:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        embeddings = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
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
    print(json.dumps([[int(index), float(scores[index])] for index in indices]))


if __name__ == "__main__":
    main()
