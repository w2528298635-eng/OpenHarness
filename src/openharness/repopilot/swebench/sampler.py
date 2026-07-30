from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

from .models import (
    DifficultyStratum,
    PublicInstance,
    SampleManifest,
    SamplingConfig,
)


class InsufficientStratumError(ValueError):
    pass


def _stable_seed(seed: int, stratum: DifficultyStratum) -> int:
    digest = hashlib.sha256(f"{seed}:{stratum.value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _public_instance(row: Mapping[str, Any]) -> PublicInstance:
    source_difficulty = str(row.get("difficulty", "")).strip()
    payload = dict(row)
    payload["source_difficulty"] = source_difficulty
    payload["difficulty"] = DifficultyStratum.from_source(source_difficulty)
    return PublicInstance.model_validate(payload)


def _sample_stratum(
    instances: list[PublicInstance],
    *,
    requested: int,
    seed: int,
    stratum: DifficultyStratum,
) -> list[PublicInstance]:
    if len(instances) < requested:
        raise InsufficientStratumError(
            f"{stratum.value} requires {requested} instances "
            f"but only {len(instances)} are available"
        )
    if requested == 0:
        return []

    by_repo: dict[str, list[PublicInstance]] = defaultdict(list)
    for instance in sorted(instances, key=lambda item: item.instance_id):
        by_repo[instance.repo].append(instance)

    rng = random.Random(_stable_seed(seed, stratum))
    repo_names = sorted(by_repo)
    rng.shuffle(repo_names)
    queues: dict[str, deque[PublicInstance]] = {}
    for repo in repo_names:
        values = by_repo[repo]
        rng.shuffle(values)
        queues[repo] = deque(values)

    selected: list[PublicInstance] = []
    while len(selected) < requested:
        made_progress = False
        for repo in repo_names:
            queue = queues[repo]
            if not queue:
                continue
            selected.append(queue.popleft())
            made_progress = True
            if len(selected) == requested:
                break
        if not made_progress:
            raise AssertionError("sampling queues exhausted before requested count")
    return selected


def manifest_sha256(manifest: SampleManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"sha256"})
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_manifest(
    rows: Iterable[Mapping[str, Any]],
    config: SamplingConfig,
    *,
    dataset_revision: str,
) -> SampleManifest:
    grouped: dict[DifficultyStratum, list[PublicInstance]] = {
        stratum: [] for stratum in DifficultyStratum
    }
    for row in rows:
        instance = _public_instance(row)
        grouped[instance.difficulty].append(instance)

    selected: list[PublicInstance] = []
    for stratum in DifficultyStratum:
        selected.extend(
            _sample_stratum(
                grouped[stratum],
                requested=config.requested(stratum),
                seed=config.seed,
                stratum=stratum,
            )
        )

    provisional = SampleManifest.model_construct(
        schema_version=1,
        dataset_name=config.dataset_name,
        dataset_revision=dataset_revision,
        sampling=config,
        instances=tuple(selected),
        sha256="0" * 64,
    )
    return SampleManifest.model_validate(
        {
            **provisional.model_dump(mode="json", exclude={"sha256"}),
            "sha256": manifest_sha256(provisional),
        }
    )

