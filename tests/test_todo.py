import asyncio

from remie.tools.todo import todo_tool
from remie.tools.executor import ToolExecutor
from remie.prompts import build_system_prompt


def test_todo_validation():
    assert todo_tool([])["items"] == []
    assert "error" in todo_tool(None)
    assert "error" in todo_tool([{"title": "x", "status": "wrong"}])
    assert todo_tool([{"title": " x ", "status": "pending"}])["items"][0]["title"] == "x"


def test_todo_approval_and_update():
    asyncio.run(_check_approval())


async def _check_approval():
    updates = []
    async def ask(question, options):
        assert "Write tests" in question
        return "Approve"
    executor = ToolExecutor(ask_user=ask, update_todo=updates.append)
    result = await executor.execute("todo", {"items": [{"title": "Write tests", "status": "pending"}], "confirm": True})
    assert result["approved"] is True
    assert updates == [result["items"], result["items"]]
    assert "first use todo" in build_system_prompt(native_tools=True)


def test_todo_sidebar_visibility_and_controls():
    from remie.tui.app import AgentApp
    from textual.widgets import Label
    from pathlib import Path

    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(140, 35)) as pilot:
            sidebar = app.query_one("#todo-sidebar")
            show = app.query_one("#todo-show")
            assert not sidebar.display
            assert not show.display

            first_tab = app._active_tab_id
            app._update_todo([{"title": "First step", "status": "pending"}])
            await pilot.pause()
            assert sidebar.display
            assert not show.display
            assert "First step" in str(app.query_one(".todo-item", Label).render())

            await pilot.click("#todo-hide")
            assert not sidebar.display
            assert show.display
            await pilot.resize_terminal(100, 35)
            await pilot.pause()
            assert show.display
            await pilot.click("#todo-show")
            assert sidebar.display
            assert not show.display

            app.action_new_chat(directory=Path.cwd())
            await pilot.pause()
            assert app._active_tab_id != first_tab
            assert not sidebar.display
            assert not show.display
            assert app.switch_tab(first_tab)
            await pilot.pause()
            assert sidebar.display

            app._update_todo([])
            await pilot.pause()
            assert not sidebar.display
            assert not show.display

    asyncio.run(exercise())


async def _approve(_question, _options):
    return "Approve"


def _items(title="Implement change", status="pending", **extra):
    return [{"title": title, "status": status, **extra}]


def test_gate_blocks_work_without_plan_and_allows_inspection():
    calls = []
    executor = ToolExecutor(_approve, run=lambda name, args: calls.append(name) or {"ok": True})

    async def exercise():
        for name, args in [
            ("edit_file", {"path": "x"}), ("run_command", {"command": "ls"}),
            ("run_test_shards", {}), ("web_fetch", {"url": "https://example.org"}),
            ("memory", {"action": "add"}), ("future_tool", {})
        ]:
            assert (await executor.execute(name, args))["reason"] == "todo_approval_required"
        assert not calls
        await executor.execute("read_file", {"filename": "x"})
        await executor.execute("memory", {"action": "read"})
        assert calls == ["read_file", "memory"]
    asyncio.run(exercise())


def test_gate_rejection_revision_progress_and_clear():
    answer = "Suggest changes"
    questions = []
    async def ask(question, options):
        questions.append(question)
        return answer
    calls = []
    executor = ToolExecutor(ask, run=lambda name, args: calls.append(name) or {"ok": True})

    async def exercise():
        nonlocal answer
        result = await executor.execute("todo", {"items": _items(), "confirm": True})
        assert not result["approved"]
        assert (await executor.execute("edit_file", {"path": "x"}))["blocked"]
        answer = "Approve"
        assert (await executor.execute("todo", {"items": _items(), "confirm": True}))["approved"]
        await executor.execute("todo", {"items": _items(status="in_progress")})
        await executor.execute("edit_file", {"path": "x"})
        assert calls == ["edit_file"]
        assert len(questions) == 2
        assert not (await executor.execute("todo", {"items": _items("Different step")}))["approved"]
        assert (await executor.execute("run_command", {"command": "ls"}))["blocked"]
        await executor.execute("todo", {"items": _items("Different step"), "confirm": True})
        await executor.execute("todo", {"items": []})
        assert (await executor.execute("edit_file", {"path": "x"}))["blocked"]
    asyncio.run(exercise())


def test_gate_and_check_ids_are_tab_local():
    from contextvars import ContextVar
    scope = ContextVar("test_scope", default="a")
    executor = ToolExecutor(_approve, permission_scope=scope.get,
                            run=lambda n, a: {"status": "passed", "mode": "local", "result": {"exit_code": 0}})
    async def exercise():
        await executor.execute("todo", {"items": _items(), "confirm": True})
        check = await executor.execute("run_test_shards", {})
        scope.set("b")
        assert not executor.plan().items
        assert (await executor.execute("edit_file", {"path": "x"}))["blocked"]
        await executor.execute("todo", {"items": _items(), "confirm": True})
        assert "error" in await executor.execute("todo", {"items": _items(status="completed", evidence="Ran tests", check_id=check["check_id"])})
        scope.set("a")
        assert executor.plan().approved
        result = await executor.execute("todo", {"items": _items(status="completed", evidence="Ran tests", check_id=check["check_id"])})
        assert result["items"][0]["verification"] == "check_passed"
    asyncio.run(exercise())


def test_completion_evidence_and_failed_or_forged_checks():
    passed = False
    executor = ToolExecutor(_approve, run=lambda n, a: {"status": "passed" if passed else "failed",
                            "mode": "local", "result": {"exit_code": 0 if passed else 1}})
    async def exercise():
        nonlocal passed
        await executor.execute("todo", {"items": _items(), "confirm": True})
        assert "error" in await executor.execute("todo", {"items": _items(status="completed")})
        for check_id in ["invented", (await executor.execute("run_test_shards", {}))["check_id"]]:
            assert "error" in await executor.execute("todo", {"items": _items(status="completed", evidence="Claim", check_id=check_id)})
        result = await executor.execute("todo", {"items": _items(status="completed", evidence="Reviewed manually", verification="check_passed")})
        assert result["items"][0]["verification"] == "agent_reported"
        passed = True
        check_id = (await executor.execute("run_test_shards", {"command": "pytest -q"}))["check_id"]
        result = await executor.execute("todo", {"items": _items(status="completed", evidence="Test run passed", check_id=check_id)})
        assert result["items"][0]["check"]["command"] == "pytest -q"
        await executor.execute("todo", {"items": _items("Changed plan")})
        assert "error" in await executor.execute("todo", {"items": _items("Changed plan", status="completed", evidence="Old result", check_id=check_id)})
    asyncio.run(exercise())


def test_approval_is_rechecked_after_permission_wait(tmp_path):
    executor = None
    async def ask(question, options):
        if "Is this todo" in question:
            return "Approve"
        executor.plan().revoke()
        return "Allow once"
    calls = []
    executor = ToolExecutor(ask, project_root=tmp_path / "project",
                            run=lambda n, a: calls.append(n) or {})
    async def exercise():
        await executor.execute("todo", {"items": _items(), "confirm": True})
        result = await executor.execute("edit_file", {"path": str(tmp_path / "outside.txt")})
        assert result["reason"] == "todo_approval_required"
        assert not calls
    asyncio.run(exercise())


def test_cancelled_and_stale_approval_never_authorize_plan():
    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()
        async def ask(question, options):
            entered.set()
            await release.wait()
            return "Approve"
        executor = ToolExecutor(ask)
        pending = asyncio.create_task(executor.execute("todo", {"items": _items(), "confirm": True}))
        await entered.wait()
        await executor.execute("todo", {"items": _items("New plan")})
        release.set()
        assert not (await pending)["approved"]
        entered.clear()
        release.clear()
        pending = asyncio.create_task(executor.execute("todo", {"items": _items("New plan"), "confirm": True}))
        await entered.wait()
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        assert not executor.plan().approved
    asyncio.run(exercise())


def test_saved_todo_restores_evidence_but_not_permission():
    from remie.storage.chats import create_chat, load_chat, save_chat, save_chat_todo
    from remie.tools.todo import TodoState
    state = TodoState()
    state.replace(_items())
    state.approved = True
    check_id = state.record_check({"command": "pytest"}, {"status": "passed", "result": {"exit_code": 0}})
    state.replace(_items(status="completed", evidence="Tests passed", check_id=check_id))
    state.expanded = False
    state.user_override = True
    chat = create_chat()
    save_chat_todo(chat["id"], state.snapshot())
    # Normal transcript saves must not erase separately persisted UI state.
    save_chat(chat["id"], [{"role": "system", "content": "test"}], [], keep_empty=True)
    restored = TodoState.restore(load_chat(chat["id"])["todo"])
    assert restored.items == state.items
    assert not restored.approved
    assert not restored.checks
    assert not restored.expanded and restored.user_override
    assert "approved" not in state.snapshot()
    assert TodoState.restore({"items": "corrupt", "approved": True}).items == []


def test_todo_visibility_is_per_tab_and_background_updates_are_isolated():
    from pathlib import Path
    from remie.tui.app import AgentApp
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(140, 35)) as pilot:
            first = app._active_tab_id
            app._update_todo(_items("Tab A"))
            await pilot.pause()
            await pilot.click("#todo-hide")
            app.action_new_chat(directory=Path.cwd())
            await pilot.pause()
            second = app._active_tab_id
            app._update_todo(_items("Tab B"))
            await pilot.pause()
            assert app.query_one("#todo-sidebar").display
            token = app._task_tab.set(first)
            try:
                app._update_todo(_items("Background A"))
                await pilot.pause()
                assert app.query_one("#todo-sidebar").display
            finally:
                app._task_tab.reset(token)
            assert app._runtime(second).todo.items == _items("Tab B")
            app.switch_tab(first)
            await pilot.pause()
            assert not app.query_one("#todo-sidebar").display
            assert app.query_one("#todo-show").display
    asyncio.run(exercise())


def test_app_restart_restores_tab_todos_without_approval(monkeypatch, tmp_path):
    from remie.tui.app import AgentApp
    monkeypatch.chdir(tmp_path)
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(140, 35)) as pilot:
            first = app._active_tab_id
            app._update_todo(_items("Saved step"))
            app._runtime().todo.approved = True
            await pilot.pause()
            await pilot.click("#todo-hide")
        resumed = AgentApp()
        async with resumed.run_test(size=(140, 35)) as pilot:
            assert resumed._active_tab_id == first
            assert resumed._runtime().todo.items == _items("Saved step")
            assert not resumed._runtime().todo.approved
            assert not resumed.query_one("#todo-sidebar").display
            assert resumed.query_one("#todo-show").display
            result = await resumed._tool_executor.execute("edit_file", {"path": "not-created.txt", "new_str": "x"})
            assert result["blocked"]
            assert not (tmp_path / "not-created.txt").exists()
    asyncio.run(exercise())


def test_new_user_request_cannot_reuse_prior_approval(tmp_path):
    import json
    from remie.core.runner import AgentRunner
    from remie.providers.events import FinishEvent, TextDelta, ToolCallEvent
    from remie.core.events import ToolCompleted

    async def exercise(native):
        target = tmp_path / ("native.txt" if native else "text.txt")
        args = {"path": str(target), "new_str": "must not be written"}
        class Provider:
            rounds = 0
            async def stream(self, _conversation):
                self.rounds += 1
                if self.rounds == 1:
                    if native:
                        yield ToolCallEvent("call-1", "edit_file", json.dumps(args))
                    else:
                        yield TextDelta("tool: edit_file(" + json.dumps(args) + ")")
                else:
                    yield TextDelta("Done")
                yield FinishEvent("stop")
        executor = ToolExecutor(_approve, project_root=tmp_path)
        await executor.execute("todo", {"items": _items(), "confirm": True})
        assert executor.plan().approved
        runner = AgentRunner(executor, Provider())
        events = [event async for event in runner.run_turn([], "New task", native_tool_calling=native)]
        completed = next(event for event in events if isinstance(event, ToolCompleted))
        assert completed.result["reason"] == "todo_approval_required"
        assert not target.exists()
    asyncio.run(exercise(False))
    asyncio.run(exercise(True))


def test_revised_plan_keeps_historical_completion_but_revokes_approval():
    from remie.tools.todo import TodoState
    state = TodoState()
    state.replace(_items())
    check_id = state.record_check({}, {"status": "passed", "result": {"exit_code": 0}})
    completed = _items(status="completed", evidence="Check passed", check_id=check_id)
    state.replace(completed)
    state.approved = True
    result = state.replace(completed + _items("Extra step"))
    assert "error" not in result
    assert not state.approved
    assert state.items[0]["verification"] == "check_passed"
    assert not state.checks


def test_known_destructive_command_rejected_without_dispatch():
    calls = []
    executor = ToolExecutor(_approve, run=lambda n, a: calls.append(n) or {})
    result = asyncio.run(executor.execute("run_command", {"command": "rm -rf /"}))
    assert result["blocked"]
    assert "recursive forced delete" in result["reason"]
    assert not calls
