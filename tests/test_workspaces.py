"""Per-tab directory isolation, including real Git worktree creation."""

import asyncio
import subprocess
from pathlib import Path

import pytest

from remie.tui import AgentApp, PromptSubmitted
from remie.tui.workspaces import WorkspaceError, separate_workspace
from remie.tools.common import tool_working_directory
from remie.tools.executor import ToolExecutor, _outside_project_paths


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
            assert app.sub_title.endswith(" · " + git("branch", "--show-current", cwd=root))
            app.action_new_chat()
            await pilot.pause()
            second = app._active_tab_id
            target = Path(app._runtime().working_directory)
            assert app.sub_title.endswith(" · " + git("branch", "--show-current", cwd=target))
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


def test_same_repository_worktrees_do_not_need_outside_permission(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    git("init", "-q", cwd=root)
    (root / "file.txt").write_text("original")
    git("add", ".", cwd=root)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "initial", cwd=root)
    tab = separate_workspace(root, "second")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    git("init", "-q", cwd=unrelated)
    (unrelated / "file.txt").write_text("other")
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("outside")
    (root / "link").symlink_to(sibling)
    prompts = []

    async def deny(question, _options):
        prompts.append(question)
        return "Deny"

    executor = ToolExecutor(deny, project_root=tab)
    assert _outside_project_paths(
        "edit_file", {"path": str(root / "new" / "file.txt")}, tab
    ) == []
    (root / "new").mkdir()

    async def check(name, args, *, allowed):
        result = await executor.execute(name, args)
        if allowed:
            assert "error" not in result, result
        else:
            assert result["error"].startswith("Permission denied"), result

    async def exercise():
        token = tool_working_directory.set(tab)
        try:
            await check("read_file", {"filename": str(root / "file.txt")}, allowed=True)
            await check("edit_file", {"path": str(root / "file.txt"),
                                      "old_str": "original", "new_str": "changed"}, allowed=True)
            await check("edit_file", {"path": str(root / "new" / "file.txt"),
                                      "old_str": "", "new_str": "new"}, allowed=True)
            await check("run_command", {"command": "pwd", "cwd": str(root)}, allowed=True)
            await check("run_command", {"command": f"cat {root / 'file.txt'}",
                                         "cwd": str(tab)}, allowed=True)
            await check("read_file", {"filename": str(unrelated / "file.txt")}, allowed=False)
            await check("read_file", {"filename": str(sibling)}, allowed=False)
            await check("read_file", {"filename": str(root / "link")}, allowed=False)
            await check("edit_file", {"path": str(tmp_path / "missing" / "file.txt"),
                                      "old_str": "", "new_str": "no"}, allowed=False)
        finally:
            tool_working_directory.reset(token)

    asyncio.run(exercise())
    assert len(prompts) == 4
    assert _outside_project_paths("read_file", {"filename": str(tab / "file.txt")}, root) == []
    assert (root / "file.txt").read_text() == "changed"
    assert (root / "new" / "file.txt").read_text() == "new"


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
            assert app.sub_title.endswith(" · " + workspace.name)
            assert app.switch_tab(original)
            await pilot.pause()
            assert app._tab_prompt_context()["working_directory"] == str(tmp_path)
            assert app.sub_title.endswith(" · " + tmp_path.name)
            other = tmp_path / "other"
            other.mkdir()
            app.on_prompt_submitted(PromptSubmitted(f"/change dir {other}"))
            assert app.sub_title.endswith(" · other")
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


def make_repository(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    git("init", "-q", cwd=root)
    (root / "file.txt").write_text("initial")
    git("add", ".", cwd=root)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "initial", cwd=root)
    return root


def test_worktree_listing_naming_and_safe_removal(tmp_path):
    from remie.tui.workspaces import list_worktrees, name_worktree, remove_worktree

    root = make_repository(tmp_path)
    first = separate_workspace(root, "first")
    branch = git("branch", "--show-current", cwd=first)
    items = list_worktrees(first)
    assert items[0]["main"] and items[0]["path"] == root
    assert items[1]["path"] == first
    renamed = name_worktree(first, "Fix / MODEL picker!")
    assert renamed.name == "project with spaces-fix-model-picker"
    assert not first.exists()
    assert git("branch", "--show-current", cwd=renamed) == branch
    second = name_worktree(separate_workspace(root, "second"), "Fix / MODEL picker!")
    assert second.name.endswith("-2")
    (renamed / "untracked").write_text("save me")
    with pytest.raises(WorkspaceError):
        remove_worktree(renamed)
    with pytest.raises(WorkspaceError):
        remove_worktree(root)
    (renamed / "untracked").unlink()
    remove_worktree(renamed)
    assert not renamed.exists()
    assert branch in git("branch", "--list", cwd=root)


def test_worktree_commands_resolve_and_complete():
    from remie.tui.slash_commands import resolve_slash_command, slash_command_matches

    for name in ("change worktree", "list worktree"):
        assert resolve_slash_command("/" + name).name == name
        assert resolve_slash_command("/" + name + "/").name == name
        assert name in {item.name for item in slash_command_matches("/" + name[:5])}


def test_picker_opens_new_tab_and_names_once(tmp_path, monkeypatch):
    from remie.storage.chats import rename_chat
    from remie.tui.workspaces import list_worktrees

    root = make_repository(tmp_path)
    target = separate_workspace(root, "existing")
    monkeypatch.chdir(root)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            original = app._active_tab_id
            async def choose(screen):
                return ("open", target)
            monkeypatch.setattr(app, "push_screen_wait", choose)
            await app._pick_worktree()
            await pilot.pause()
            second = app._active_tab_id
            assert second != original
            assert len(app._tab_layout["tabs"]) == 2
            assert Path(app._runtime().working_directory) == target
            rename_chat(app._runtime().chat_id, "Readable title", title_source="manual")
            app._name_tab_worktree(second)
            renamed = Path(app._runtime().working_directory)
            assert renamed.name.endswith("readable-title")
            rename_chat(app._runtime().chat_id, "Different title", title_source="manual")
            app._name_tab_worktree(second)
            assert Path(app._runtime().working_directory) == renamed
            assert len(list_worktrees(root)) == 2
            async def keep(screen):
                return "Keep worktree and close tab"
            monkeypatch.setattr(app, "push_screen_wait", keep)
            await app._close_worktree_tab(second)
            await pilot.pause()
            assert renamed.exists()
            assert second not in app._runtimes
    asyncio.run(exercise())


def test_close_worktree_cancel_dirty_and_delete(tmp_path, monkeypatch):
    root = make_repository(tmp_path)
    monkeypatch.chdir(root)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app.action_new_chat()
            await pilot.pause()
            tab_id = app._active_tab_id
            path = Path(app._runtime().working_directory)
            async def cancel(screen):
                return "Cancel"
            monkeypatch.setattr(app, "push_screen_wait", cancel)
            await app._close_worktree_tab(tab_id)
            assert tab_id in app._runtimes
            async def delete(screen):
                return "Delete worktree and close tab"
            monkeypatch.setattr(app, "push_screen_wait", delete)
            (path / "untracked").write_text("keep")
            await app._close_worktree_tab(tab_id)
            assert tab_id in app._runtimes and path.exists()
            (path / "untracked").unlink()
            await app._close_worktree_tab(tab_id)
            await pilot.pause()
            assert tab_id not in app._runtimes and not path.exists()
    asyncio.run(exercise())


def test_worktree_picker_and_close_prompt_ui(tmp_path, monkeypatch):
    from remie.tui.screens.worktrees import WorktreeScreen
    from remie.tui.screens.ask_user import AskUserScreen
    from textual.widgets import OptionList

    root = make_repository(tmp_path)
    separate_workspace(root, "existing")
    monkeypatch.chdir(root)

    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(110, 40)) as pilot:
            original = app._active_tab_id
            app._dispatch_slash_command("list worktree")
            await pilot.pause()
            assert isinstance(app.screen, WorktreeScreen)
            app.screen.query_one(OptionList).highlighted = 1
            await pilot.press("enter")
            await pilot.pause()
            second = app._active_tab_id
            assert second != original
            assert len(app._tab_layout["tabs"]) == 2
            assert not app.close_tab(second)
            await pilot.pause()
            assert isinstance(app.screen, AskUserScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert second in app._runtimes
            app._dispatch_slash_command("change worktree")
            await pilot.pause()
            assert isinstance(app.screen, WorktreeScreen)
            app.screen.query_one(OptionList).highlighted = 0
            await pilot.press("enter")
            await pilot.pause()
            assert app._active_tab_id == original
            assert len(app._tab_layout["tabs"]) == 2
    asyncio.run(exercise())


def test_delete_final_linked_tab_reopens_main(tmp_path, monkeypatch):
    root = make_repository(tmp_path)
    target = separate_workspace(root, "sole")
    monkeypatch.chdir(root)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app.on_prompt_submitted(PromptSubmitted(f"/change dir {target}"))
            tab_id = app._active_tab_id
            async def delete(screen):
                return "Delete worktree and close tab"
            monkeypatch.setattr(app, "push_screen_wait", delete)
            await app._close_worktree_tab(tab_id)
            await pilot.pause()
            assert not target.exists()
            assert len(app._tab_layout["tabs"]) == 1
            assert Path(app._runtime().working_directory) == root
    asyncio.run(exercise())


def test_delete_from_subdirectory_removes_entire_linked_folder(tmp_path):
    from remie.tui.workspaces import list_worktrees, remove_worktree

    root = make_repository(tmp_path)
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("committed")
    git("add", ".", cwd=root)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "nested file", cwd=root)
    linked = separate_workspace(root, "delete-nested")
    other = separate_workspace(root, "keep-other")
    remove_worktree(linked / "sub")
    assert not linked.exists()
    assert {item["path"] for item in list_worktrees(root)} == {root, other}
    assert (root / "sub" / "nested.txt").read_text() == "committed"
    with pytest.raises(WorkspaceError, match="main worktree is protected"):
        remove_worktree(root / "sub")
    assert root.is_dir() and other.is_dir()


def test_delete_reports_folder_left_by_git(tmp_path, monkeypatch):
    import remie.tui.workspaces as workspaces

    root = make_repository(tmp_path)
    linked = separate_workspace(root, "leftover")
    real_git = workspaces._git

    def leave_folder(directory, *args):
        if args[:2] == ("worktree", "remove"):
            return ""
        return real_git(directory, *args)

    monkeypatch.setattr(workspaces, "_git", leave_folder)
    with pytest.raises(WorkspaceError, match="left its folder behind"):
        workspaces.remove_worktree(linked)
    assert linked.is_dir() and root.is_dir()
