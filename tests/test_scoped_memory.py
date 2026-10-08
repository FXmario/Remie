import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from remie.storage.memory import (
    history_search_tool, home, memory_context, memory_path, memory_snapshot,
    memory_tool, project_id, scope_context,
)
from remie.tools.executor import ToolExecutor


@pytest.fixture
def context(tmp_path):
    scope = scope_context(tmp_path, "tab-a", "chat-a")
    token = memory_context.set(scope)
    yield scope
    memory_context.reset(token)


def test_memory_scope_isolation(context):
    assert memory_tool("add", "global", "Prefer concise answers")["success"]
    assert memory_tool("add", "project", "Run pytest")["success"]
    assert memory_tool("add", "tab", "Private investigation")["success"]
    other = {**context, "tab_id": "tab-b"}
    snapshot = memory_snapshot(other)
    assert "Run pytest" in snapshot
    assert "Private investigation" not in snapshot
    assert "Prefer concise" in memory_snapshot({**other, "project_id": "other"})
    assert "Run pytest" not in memory_snapshot({**other, "project_id": "other"})
    assert memory_path("tab", context).read_text() == "Private investigation"


def test_replace_remove_limits(context, monkeypatch):
    memory_tool("add", content="alpha original")
    memory_tool("add", content="beta original")
    assert "error" in memory_tool("replace", content="replacement", old_text="original")
    assert memory_tool("replace", content="replacement", old_text="alpha")["success"]
    monkeypatch.setenv("REMIE_MEMORY_TAB_LIMIT", "10")
    assert "error" in memory_tool("add", content="too much text")
    monkeypatch.setenv("REMIE_MEMORY_TAB_LIMIT", "2200")
    assert memory_tool("remove", old_text="beta")["success"]
    assert memory_tool()["content"] == "replacement"


def test_concurrent_project_writes(context):
    def write(i):
        token = memory_context.set({**context, "tab_id": f"tab-{i}"})
        try:
            return memory_tool("add", "project", f"note {i}")
        finally:
            memory_context.reset(token)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result["success"] for result in pool.map(write, range(12)))
    assert memory_tool(scope="project")["content"].count("note ") == 12


def write_chat(project, chat, tab, text, *, message_scope=None):
    folder = home() / "projects" / project / "chats"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "index.json"
    data = json.loads(index.read_text()) if index.exists() else {"chats": {}}
    data["chats"][chat] = {"name": f"Title {chat}", "updated_at": "2026-01-01"}
    index.write_text(json.dumps(data))
    message = {"role": "user", "content": text}
    if message_scope:
        message["memory_scope"] = message_scope
    path = folder / f"{chat}.json"
    path.write_text(json.dumps({"memory_scope": {"project_id": project, "tab_id": tab}, "transcript": [message]}))
    return path


def test_history_private_default_and_rebuild(context):
    project = context["project_id"]
    path = write_chat(project, "chat-a", "tab-a", "Podman network settings")
    write_chat(project, "chat-b", "tab-b", "Podman other tab")
    write_chat("other-project", "chat-c", "tab-a", "Podman unrelated project")
    result = history_search_tool("Podman")
    assert [row["chat_id"] for row in result["results"]] == ["chat-a"]
    assert result["results"][0]["title"] == "Title chat-a"
    assert len(history_search_tool("Podman", "project")["results"]) == 2
    path.unlink()
    assert history_search_tool("Podman")["results"] == []
    (home() / "history.sqlite").unlink()
    assert len(history_search_tool("Podman", "project")["results"]) == 1
    assert "error" in history_search_tool("Podman", "global")


def test_history_uses_message_scope(context):
    write_chat("other-project", "moved-chat", "tab-a", "original project detail", message_scope=context)
    assert len(history_search_tool("original", "project")["results"]) == 1
    assert history_search_tool('" OR * --')["results"] == []


def test_memory_approval_cancelled(context):
    async def deny(question, choices):
        assert "project memory" in question
        return "Cancel"
    async def exercise():
        result = await ToolExecutor(ask_user=deny).execute("memory", {"action": "add", "scope": "project", "content": "No save"})
        assert "error" in result
    asyncio.run(exercise())
    assert not memory_path("project", context).exists()


def test_tab_prompt_snapshot_and_reload(tmp_path, monkeypatch):
    from remie.tui.app import AgentApp
    monkeypatch.chdir(tmp_path)
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            original = app._active_tab_id
            ctx = app._tab_prompt_context()["memory_scope"]
            token = memory_context.set(ctx)
            try:
                memory_tool("add", "project", "Shared project fact")
                memory_tool("add", "tab", "Private tab fact")
            finally:
                memory_context.reset(token)
            assert "Shared project fact" not in app._tab_prompt_context()["memory_snapshot"]
            app._dispatch_slash_command("memory reload")
            assert "Private tab fact" in app.conversation[0]["content"]
            app.action_new_chat()
            await pilot.pause()
            assert "Shared project fact" in app.conversation[0]["content"]
            assert "Private tab fact" not in app.conversation[0]["content"]
            app.switch_tab(original)
            app._dispatch_slash_command("memory reload")
            assert "Private tab fact" in app.conversation[0]["content"]
        restored = AgentApp()
        async with restored.run_test():
            assert original in restored._runtimes
            assert "Private tab fact" in restored._tab_prompt_context()["memory_snapshot"]
    asyncio.run(exercise())


def test_linked_worktrees_share_identity(tmp_path):
    import subprocess
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", str(main)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(main), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "init"], check=True, capture_output=True)
    linked = tmp_path / "linked"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-b", "test", str(linked)], check=True, capture_output=True)
    assert project_id(main) == project_id(linked)


def test_legacy_json_backfill(context):
    from remie.storage.chats import create_chat, save_chat
    # Legacy local storage has no scope metadata. Its ownership is the launch
    # project, not whichever project's tab happens to issue a search.
    from pathlib import Path
    token = memory_context.set({**context, "project_id": project_id(Path.cwd())})
    try:
        chat = create_chat("Legacy")
        messages = [{"role": "user", "content": "legacy keyword"}]
        save_chat(chat["id"], messages, messages)
        assert len(history_search_tool("legacy", "project")["results"]) == 1
        assert history_search_tool("legacy")["results"] == []
        current = memory_context.set({**memory_context.get(), "chat_id": chat["id"]})
        try:
            assert len(history_search_tool("legacy")["results"]) == 1
        finally:
            memory_context.reset(current)
    finally:
        memory_context.reset(token)


def test_project_change_preserves_legacy_message_scope(tmp_path, monkeypatch):
    from remie.tui.app import AgentApp
    from remie.storage.chats import chat_file_path
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "other-project"
    other.mkdir()
    async def exercise():
        app = AgentApp()
        async with app.run_test():
            old = app._tab_prompt_context()["memory_scope"]
            # Simulate a restored older chat with an unscoped transcript.
            app._transcript.append({"role": "user", "content": "old project keyword"})
            app._dispatch_slash_command("change dir", str(other))
            new = app._tab_prompt_context()["memory_scope"]
            app._push_message("user", "new project keyword")
            app._save_current_chat()
            payload = json.loads(chat_file_path(app._chat_id).read_text())
            assert payload["transcript"][0]["memory_scope"] == old
            assert payload["transcript"][1]["memory_scope"] == new
            token = memory_context.set(new)
            try:
                assert history_search_tool("old project")["results"] == []
                assert len(history_search_tool("new project")["results"]) == 1
            finally:
                memory_context.reset(token)
    asyncio.run(exercise())


def test_context_propagates_to_background_tool_threads(context):
    async def exercise():
        async def worker(tab):
            token = memory_context.set({**context, "tab_id": tab})
            try:
                await asyncio.to_thread(memory_tool, "add", "tab", tab + " private")
                return await asyncio.to_thread(memory_tool)
            finally:
                memory_context.reset(token)
        results = await asyncio.gather(worker("tab-a"), worker("tab-b"))
        assert results[0]["content"] == "tab-a private"
        assert results[1]["content"] == "tab-b private"
    asyncio.run(exercise())


def test_approval_accepts_and_refuses_concurrent_changes(context):
    async def approve(question, choices):
        return "Save memory"
    async def exercise():
        executor = ToolExecutor(ask_user=approve)
        assert (await executor.execute("memory", {"action": "add", "content": "original entry"}))["success"]
        async def changed(question, choices):
            assert "original entry" in question
            memory_tool("replace", content="changed entry", old_text="original")
            return "Save memory"
        executor.ask_user = changed
        result = await executor.execute("memory", {"action": "replace", "content": "approved replacement", "old_text": "entry"})
        assert "changed during approval" in result["error"]
    asyncio.run(exercise())
    assert memory_tool()["content"] == "changed entry"
