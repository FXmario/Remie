"""Read-only TODO sidebar and checkpoint display integration."""
import asyncio

from textual.widgets import Button, Label

from remie.tui.app import AgentApp
from remie.tui.widgets import TodoSidebar, TabSidebar


def test_sidebar_updates_and_toggles(tmp_path, monkeypatch):
    monkeypatch.setenv("REMIE_HOME", str(tmp_path))
    from remie.storage.memory import memory_context
    from remie.storage.todo import todo_tool, begin_output, finish_output, todo_view

    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(140, 40)) as pilot:
            sidebar = app.query_one(TodoSidebar)
            assert not sidebar.display
            assert not app.query_one("#todo-show", Button).display
            assert "No TODO" in str(sidebar.query_one("#todo-content", Label).render())
            scope = app._tab_prompt_context()["memory_scope"]
            token = memory_context.set(scope)
            try:
                item = todo_tool("add", title="Build [literal] sidebar")
                todo_tool("update", item_id=item["id"], status="in_progress")
                ticket = begin_output("run_command", {"command": "pytest"})
                finish_output(ticket, {"stdout": "passed", "exit_code": 0})
                app._refresh_todo()
                assert sidebar.display
                await pilot.pause()
                assert sidebar.region.x > app.query_one(TabSidebar).region.right
                text = str(sidebar.query_one("#todo-content", Label).render())
                assert "Build [literal] sidebar" in text
                assert "in progress" in text
                assert not todo_view(scope)["items"][0].get("reviewed_outputs")
                todo_tool("read")
                todo_tool("update", item_id=item["id"], status="completed", summary="Tests passed")
                app._refresh_todo()
                assert "Tests passed" in str(sidebar.query_one("#todo-content", Label).render())
            finally:
                memory_context.reset(token)
            await pilot.click("#todo-hide")
            assert not sidebar.display
            assert app.query_one("#todo-show", Button).display
            await pilot.click("#todo-show")
            assert sidebar.display
            monkeypatch.setattr("remie.storage.todo.todo_view", lambda context: {"items": []})
            app._refresh_todo()
            assert not sidebar.display
            assert not app.query_one("#todo-show", Button).display
    asyncio.run(exercise())
