from __future__ import annotations

import pytest
from pydantic import ValidationError

from openharness.repopilot.swebench.models import (
    DifficultyStratum,
    PublicInstance,
    SamplingConfig,
)


def test_public_instance_rejects_gold_only_fields() -> None:
    payload = {
        "instance_id": "django__django-1",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix the field conversion.",
        "source_difficulty": "<15 min fix",
        "difficulty": DifficultyStratum.EASY,
        "patch": "diff --git a/answer.py b/answer.py",
    }

    with pytest.raises(ValidationError, match="gold-only"):
        PublicInstance.model_validate(payload)


def test_public_instance_rejects_unknown_fields() -> None:
    payload = {
        "instance_id": "django__django-1",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix the field conversion.",
        "source_difficulty": "<15 min fix",
        "difficulty": DifficultyStratum.EASY,
        "unexpected": "value",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PublicInstance.model_validate(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<15 min fix", DifficultyStratum.EASY),
        ("15 min - 1 hour", DifficultyStratum.MEDIUM),
        ("1-4 hours", DifficultyStratum.HARD),
        (">4 hours", DifficultyStratum.HARD),
        ("4 hours", DifficultyStratum.HARD),
    ],
)
def test_difficulty_stratum_uses_official_human_labels(
    value: str,
    expected: DifficultyStratum,
) -> None:
    assert DifficultyStratum.from_source(value) is expected


def test_difficulty_stratum_rejects_unrecognized_labels() -> None:
    with pytest.raises(ValueError, match="unsupported SWE-bench difficulty"):
        DifficultyStratum.from_source("unknown")


def test_sampling_config_requires_a_nonempty_sample() -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(easy=0, medium=0, hard=0)

