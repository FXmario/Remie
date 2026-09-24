"""Per-tab directory isolation, including real Git worktree creation."""

import asyncio
import subprocess
from pathlib import Path

import pytest

from remie.tui import AgentApp, PromptSubmitted
from remie.tui.workspaces import WorkspaceError, separate_workspace


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def test_git_worktree_starts_at_head_and_preserves_original(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    git("init", "-q", cwd=root)
    (root / "sub").mkdir()
    (root / "sub" / "file.txt").write_text("committed")
    git("add", ".", cwd=root)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "initial", cwd=root)
    (root / "sub" / "untracked.txt").write_text("not in HEAD")
    monkeypatch.chdir(root / "sub")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            original = app._active_tab_id
            app.action_new_chat()
            await pilot.pause()
            second = app._active_tab_id
            target = Path(app._runtime().working_directory)
            assert target != root / "sub"
            assert target.parent.parent == tmp_path
            assert (target / "file.txt").read_text() == "committed"
            assert not (target / "untracked.txt").exists()
            assert git("branch", "--show-current", cwd=target).startswith(f"remie/tab-{second}-")
            assert git("rev-parse", "HEAD", cwd=target) == git("rev-parse", "HEAD", cwd=root)
            assert app._runtimes[original].working_directory is None
            app.on_prompt_submitted(PromptSubmitted(f"/change dir {root / 'sub'}"))
            # The original tab still owns that directory: remain isolated.
            assert app._runtime().working_directory != str(root / "sub")
            assert Path(app._runtime().working_directory).is_dir()
            assert app.switch_tab(original)
            await pilot.pause()
            assert app._tab_prompt_context()["working_directory"] == str(root / "sub")
        restored = AgentApp()
        async with restored.run_test():
            assert restored._runtimes[second].working_directory == str(app._runtimes[second].working_directory)

    asyncio.run(exercise())


def test_non_git_tabs_and_change_dir_get_empty_unique_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "original.txt").write_text("original")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            original = app._active_tab_id
            app.action_new_chat()
            await pilot.pause()
            second = app._active_tab_id
            workspace = Path(app._runtime().working_directory)
            assert workspace.parent == tmp_path
            assert list(workspace.iterdir()) == []
            app.on_prompt_submitted(PromptSubmitted(f"/change dir {tmp_path}"))
            assert Path(app._runtime().working_directory).parent == tmp_path
            assert Path(app._runtime().working_directory) == workspace
            assert app.switch_tab(original)
            await pilot.pause()
            assert app._tab_prompt_context()["working_directory"] == str(tmp_path)
            assert app.switch_tab(second)
            await pilot.pause()
            assert app._tab_prompt_context()["working_directory"] != str(tmp_path)

    asyncio.run(exercise())


def test_restoring_legacy_duplicate_tabs_isolates_second(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            original = app._active_tab_id
            app.action_new_chat()
            await pilot.pause()
            second = app._active_tab_id
            for tab in app._tab_layout["tabs"]:
                tab["working_directory"] = str(tmp_path)
                tab.pop("workspace_source", None)
            app._persist_tab_layout()
        restored = AgentApp()
        async with restored.run_test():
            assert restored._runtimes[original].working_directory == str(tmp_path)
            assert Path(restored._runtimes[second].working_directory).parent == tmp_path
            assert restored._runtimes[second].working_directory != str(tmp_path)

    asyncio.run(exercise())


def test_workspace_creation_error_does_not_switch_tabs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail(*args):
        raise WorkspaceError("cannot create workspace")

    monkeypatch.setattr("remie.tui.app.separate_workspace", fail)

    async def exercise():
        app = AgentApp()
        async with app.run_test():
            original = app._active_tab_id
            app.action_new_chat()
            assert app._active_tab_id == original
            assert len(app._tab_layout["tabs"]) == 1
            assert app._tab_prompt_context()["working_directory"] == str(tmp_path)

    asyncio.run(exercise())


def test_workspace_failure_preserves_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("remie.tui.workspaces.uuid.uuid4", lambda: type("Id", (), {"hex": "12345678"})())
    (tmp_path / "remie-tab-duplicate-12345678").mkdir()
    with pytest.raises(WorkspaceError):
        separate_workspace(tmp_path, "duplicate")
