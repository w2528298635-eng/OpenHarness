from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, computed_field

_HUNK_RE = re.compile(
    r"^@@ -(?P<old>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new>\d+)(?:,(?P<new_count>\d+))? @@"
)


def _normalize_path(value: str) -> str | None:
    normalized = value.strip().strip('"').replace("\\", "/")
    if normalized in {"/dev/null", "dev/null"}:
        return None
    if normalized.startswith(("a/", "b/")):
        normalized = normalized[2:]
    return PurePosixPath(normalized).as_posix()


def _is_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    parts = {part.casefold() for part in pure.parts}
    name = pure.name.casefold()
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


@dataclass
class _PatchSection:
    old_path: str | None
    new_path: str | None
    old_changed_lines: list[int] = field(default_factory=list)
    new_changed_lines: list[int] = field(default_factory=list)

    @property
    def target_path(self) -> str | None:
        return self.new_path or self.old_path


def _diff_paths(line: str) -> tuple[str | None, str | None]:
    values = shlex.split(line.removeprefix("diff --git "), posix=False)
    if len(values) != 2:
        raise ValueError(f"invalid unified diff header: {line}")
    return _normalize_path(values[0]), _normalize_path(values[1])


def _parse_sections(patch: str) -> list[_PatchSection]:
    sections: list[_PatchSection] = []
    current: _PatchSection | None = None
    old_line: int | None = None
    new_line: int | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            old_path, new_path = _diff_paths(line)
            current = _PatchSection(old_path=old_path, new_path=new_path)
            sections.append(current)
            old_line = None
            new_line = None
            continue
        if current is None:
            continue
        if line.startswith("rename from "):
            current.old_path = _normalize_path(line.removeprefix("rename from "))
            continue
        if line.startswith("rename to "):
            current.new_path = _normalize_path(line.removeprefix("rename to "))
            continue
        if line.startswith("--- "):
            current.old_path = _normalize_path(line[4:])
            continue
        if line.startswith("+++ "):
            current.new_path = _normalize_path(line[4:])
            continue
        match = _HUNK_RE.match(line)
        if match:
            old_line = int(match.group("old"))
            new_line = int(match.group("new"))
            continue
        if old_line is None or new_line is None:
            continue
        if line.startswith("-") and not line.startswith("---"):
            current.old_changed_lines.append(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            current.new_changed_lines.append(new_line)
            new_line += 1
        elif not line.startswith("\\"):
            old_line += 1
            new_line += 1
    return sections


class GoldLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    files: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]] = {}

    @computed_field
    @property
    def symbol_denominator(self) -> int:
        return sum(len(values) for values in self.symbols.values())


def extract_gold_files(
    patch: str,
    *,
    exclude_tests: bool = True,
) -> tuple[str, ...]:
    files: list[str] = []
    for section in _parse_sections(patch):
        path = section.target_path
        if path is None or (exclude_tests and _is_test_path(path)):
            continue
        if path not in files:
            files.append(path)
    return tuple(files)


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.symbols: list[tuple[int, int, str]] = []

    def _visit_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> None:
        qualified = ".".join((*self.stack, node.name))
        end_line = getattr(node, "end_lineno", node.lineno)
        self.symbols.append((node.lineno, end_line, qualified))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node)


def _symbols_for_lines(source: str | None, lines: list[int]) -> list[str]:
    if source is None or not lines:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    collector = _SymbolCollector()
    collector.visit(tree)
    result: list[str] = []
    for line in lines:
        candidates = [
            item
            for item in collector.symbols
            if item[0] <= line <= item[1]
        ]
        if not candidates:
            continue
        _, _, name = min(candidates, key=lambda item: (item[1] - item[0], -item[0]))
        if name not in result:
            result.append(name)
    return result


def extract_gold_labels(
    patch: str,
    *,
    base_sources: Mapping[str, str] | None = None,
    patched_sources: Mapping[str, str] | None = None,
    exclude_tests: bool = True,
) -> GoldLabels:
    base_sources = base_sources or {}
    patched_sources = patched_sources or {}
    files: list[str] = []
    symbols: dict[str, tuple[str, ...]] = {}
    for section in _parse_sections(patch):
        path = section.target_path
        if path is None or (exclude_tests and _is_test_path(path)):
            continue
        if path not in files:
            files.append(path)
        names: list[str] = []
        old_path = section.old_path or path
        new_path = section.new_path or path
        for name in _symbols_for_lines(
            base_sources.get(old_path),
            section.old_changed_lines,
        ):
            if name not in names:
                names.append(name)
        for name in _symbols_for_lines(
            patched_sources.get(new_path),
            section.new_changed_lines,
        ):
            if name not in names:
                names.append(name)
        if names:
            symbols[path] = tuple(names)
    return GoldLabels(files=tuple(files), symbols=symbols)
