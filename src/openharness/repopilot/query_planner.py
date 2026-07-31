from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

_ERROR = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b")
_BACKTICK = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}\b")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original: str
    errors: tuple[str, ...]
    identifiers: tuple[str, ...]
    paths: tuple[str, ...]
    queries: tuple[str, ...]


class QueryPlanner:
    """Turn a verbose issue into a small set of high-signal retrieval queries."""

    def plan(self, text: str) -> QueryPlan:
        original = " ".join(text.split())
        errors = tuple(dict.fromkeys(_ERROR.findall(original)))
        paths = tuple(dict.fromkeys(_DOTTED.findall(original)))
        quoted = _BACKTICK.findall(original)
        identifiers = tuple(
            dict.fromkeys(
                value
                for value in [*quoted, *_IDENTIFIER.findall(original)]
                if "_" in value and value not in paths
            )
        )
        signals = " ".join((*errors, *identifiers, *paths))
        queries = tuple(dict.fromkeys(value for value in (original, signals) if value))
        return QueryPlan(
            original=original,
            errors=errors,
            identifiers=identifiers,
            paths=paths,
            queries=queries,
        )
