from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str
    required_variables: frozenset[str]

    def render(self, variables: dict[str, str]) -> str:
        provided = frozenset(variables)
        missing = self.required_variables - provided
        unexpected = provided - self.required_variables
        if missing:
            raise ValueError(f"missing prompt variables: {', '.join(sorted(missing))}")
        if unexpected:
            raise ValueError(f"unexpected prompt variables: {', '.join(sorted(unexpected))}")
        return self.template.format_map(variables)


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        key = (template.name, template.version)
        if key in self._templates:
            raise ValueError(f"prompt already registered: {template.name}@{template.version}")
        self._templates[key] = template

    def get(self, name: str, version: str) -> PromptTemplate:
        try:
            return self._templates[(name, version)]
        except KeyError as exc:
            raise KeyError(f"unknown prompt: {name}@{version}") from exc

    def render(self, name: str, *, version: str, **variables: str) -> str:
        return self.get(name, version).render(variables)
