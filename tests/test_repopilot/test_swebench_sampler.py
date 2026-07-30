from __future__ import annotations

import pytest

from openharness.repopilot.swebench.models import SamplingConfig
from openharness.repopilot.swebench.sampler import (
    InsufficientStratumError,
    derive_pilot_manifest,
    manifest_sha256,
    sample_manifest,
)


def _row(
    number: int,
    *,
    repo: str,
    difficulty: str,
) -> dict[str, str]:
    return {
        "instance_id": f"{repo.replace('/', '__')}-{number}",
        "repo": repo,
        "base_commit": f"commit-{number}",
        "problem_statement": f"Repair public issue {number}.",
        "difficulty": difficulty,
    }


def _rows() -> list[dict[str, str]]:
    return [
        _row(1, repo="alpha/project", difficulty="<15 min fix"),
        _row(2, repo="alpha/project", difficulty="<15 min fix"),
        _row(3, repo="beta/project", difficulty="<15 min fix"),
        _row(4, repo="gamma/project", difficulty="<15 min fix"),
        _row(5, repo="alpha/project", difficulty="15 min - 1 hour"),
        _row(6, repo="beta/project", difficulty="15 min - 1 hour"),
        _row(7, repo="gamma/project", difficulty="15 min - 1 hour"),
        _row(8, repo="gamma/project", difficulty="15 min - 1 hour"),
        _row(9, repo="alpha/project", difficulty="1-4 hours"),
        _row(10, repo="beta/project", difficulty="1-4 hours"),
        _row(11, repo="gamma/project", difficulty=">4 hours"),
        _row(12, repo="gamma/project", difficulty=">4 hours"),
    ]


def test_sample_manifest_is_reproducible_when_input_order_changes() -> None:
    config = SamplingConfig(easy=2, medium=2, hard=2, seed=20260730)

    first = sample_manifest(
        _rows(),
        config,
        dataset_revision="revision-abc",
    )
    second = sample_manifest(
        list(reversed(_rows())),
        config,
        dataset_revision="revision-abc",
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.sha256 == manifest_sha256(first)
    assert len(first.instances) == 6


def test_sample_manifest_round_robins_repositories_before_reusing_one() -> None:
    config = SamplingConfig(easy=3, medium=1, hard=1, seed=20260730)

    manifest = sample_manifest(_rows(), config, dataset_revision="revision-abc")

    easy_repositories = {
        instance.repo for instance in manifest.instances if instance.difficulty.value == "easy"
    }
    assert easy_repositories == {"alpha/project", "beta/project", "gamma/project"}


def test_sample_manifest_digest_changes_with_dataset_revision() -> None:
    config = SamplingConfig(easy=1, medium=1, hard=1, seed=20260730)

    first = sample_manifest(_rows(), config, dataset_revision="revision-a")
    second = sample_manifest(_rows(), config, dataset_revision="revision-b")

    assert first.sha256 != second.sha256


def test_sample_manifest_reports_an_insufficient_stratum() -> None:
    config = SamplingConfig(easy=5, medium=1, hard=1, seed=20260730)

    with pytest.raises(
        InsufficientStratumError,
        match=r"easy requires 5 instances but only 4 are available",
    ):
        sample_manifest(_rows(), config, dataset_revision="revision-abc")


def test_sample_manifest_never_persists_gold_fields() -> None:
    rows: list[dict[str, str]] = _rows()
    rows[0]["patch"] = "diff --git a/private.py b/private.py"
    config = SamplingConfig(easy=1, medium=1, hard=1, seed=20260730)

    with pytest.raises(ValueError, match="gold-only"):
        sample_manifest(rows, config, dataset_revision="revision-abc")


def test_pilot_manifest_contains_one_frozen_instance_per_stratum() -> None:
    formal = sample_manifest(
        _rows(),
        SamplingConfig(easy=2, medium=2, hard=2, seed=20260730),
        dataset_revision="revision-abc",
    )

    pilot = derive_pilot_manifest(formal)

    assert len(pilot.instances) == 3
    assert {instance.difficulty.value for instance in pilot.instances} == {
        "easy",
        "medium",
        "hard",
    }
    assert pilot.sampling.easy == 1
    assert pilot.sampling.medium == 1
    assert pilot.sampling.hard == 1
    assert pilot.sha256 == manifest_sha256(pilot)
    assert {item.instance_id for item in pilot.instances}.issubset(
        {item.instance_id for item in formal.instances}
    )
