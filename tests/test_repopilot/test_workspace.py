import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.repopilot.workspace import WorkspaceManager


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)


class FakeWorktrees:
    async def create_worktree(self, repo_path, slug, branch=None, agent_id=None):
        path = repo_path.parent / "isolated"
        path.mkdir()
        return SimpleNamespace(path=path, branch=branch, original_path=repo_path)


@pytest.mark.asyncio
async def test_workspace_creation_uses_isolated_location(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = WorkspaceManager(FakeWorktrees())

    info = await workspace.create(repo, "run-1")

    assert info.path == tmp_path / "isolated"
    assert info.path != repo
    assert info.branch == "repopilot/run-1"


def test_changed_path_policy_rejects_sensitive_and_out_of_scope(tmp_path: Path) -> None:
    workspace = WorkspaceManager(FakeWorktrees())

    assert workspace.validate_changed_files(["src/app.py"], ["src/**"]) is None
    assert workspace.validate_changed_files(["tests/test_app.py"], ["src/**"])
    assert workspace.validate_changed_files([".git/config"], None)
    assert workspace.validate_changed_files([".openharness/repopilot/x"], None)


def test_diff_signature_is_stable() -> None:
    workspace = WorkspaceManager(FakeWorktrees())
    assert workspace.diff_signature("same diff") == workspace.diff_signature("same diff")


@pytest.mark.asyncio
async def test_default_worktree_storage_is_next_to_repository(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    captured = {}

    class RecordingManager(FakeWorktrees):
        def __init__(self, base_dir):
            captured["base_dir"] = base_dir

    monkeypatch.setattr("openharness.repopilot.workspace.WorktreeManager", RecordingManager)
    workspace = WorkspaceManager()
    await workspace.create(repo, "run-1")

    assert captured["base_dir"] == tmp_path / ".openharness-repopilot-worktrees" / "repo"


@pytest.mark.asyncio
async def test_diff_includes_untracked_text_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "new_module.py").write_text("VALUE = 1\n", encoding="utf-8")

    diff = await WorkspaceManager(FakeWorktrees()).diff(repo)

    assert "new_module.py" in diff
    assert "+VALUE = 1" in diff
