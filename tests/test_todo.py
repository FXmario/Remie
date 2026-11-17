"""Persistent TODO checkpoints without an ordinary-tool approval gate."""
import asyncio

from remie.prompts import build_system_prompt
from remie.tools.registry import TOOL_REGISTRY, get_tool_schemas
from remie.tools.executor import ToolExecutor


def test_todo_tool_and_prompt():
    assert "todo" in TOOL_REGISTRY
    assert "todo" in {schema["name"] for schema in get_tool_schemas()}
    for native in (False, True):
        prompt = build_system_prompt(native_tools=native)
        assert "todo" in prompt.lower()
        assert "Before implementation, persist the full initial checklist" in prompt
        assert "Do not create steps one at a time" in prompt
        assert "follow the TODO in order" in prompt
        assert "Keep future steps pending and update the checklist" in prompt


def test_tools_execute_without_plan(tmp_path):
    async def exercise():
        async def ask(question, options):
            raise AssertionError("Unexpected approval prompt: " + question)
        calls = []
        def run(name, args):
            calls.append(name)
            return {"ok": True}
        executor = ToolExecutor(ask_user=ask, run=run, project_root=tmp_path)
        for name, args in (
            ("edit_file", {"path": str(tmp_path / "doc.txt"), "old_str": "a", "new_str": "b"}),
            ("run_command", {"command": "git status", "cwd": str(tmp_path)}),
            ("run_test_shards", {"cwd": str(tmp_path)}),
        ):
            assert (await executor.execute(name, args))["ok"]
        assert calls == ["edit_file", "run_command", "run_test_shards"]
    asyncio.run(exercise())


def test_destructive_commands_still_blocked():
    async def exercise():
        async def ask(question, options):
            raise AssertionError("Unexpected prompt")
        executor = ToolExecutor(ask_user=ask)
        assert (await executor.execute("run_command", {"command": "rm -rf /"}))["blocked"]
    asyncio.run(exercise())


def test_persistent_evidence_and_isolation(tmp_path, monkeypatch):
    from remie.storage.memory import memory_context
    from remie.storage.todo import todo_tool, todo_snapshot, begin_output, finish_output
    monkeypatch.setenv("REMIE_HOME", str(tmp_path))
    scope = {"project_id": "project", "tab_id": "tab"}
    token = memory_context.set(scope)
    try:
        item = todo_tool("add", title="Implement and test")
        ident = item["id"]
        todo_tool("update", item_id=ident, status="in_progress")
        assert "error" in todo_tool("update", item_id=ident, status="completed", summary="done")
        ticket = begin_output("run_command", {"command": "pytest"})
        assert "result" not in todo_tool()["items"][0]["outputs"][0]
        assert "Implement and test" in todo_snapshot(scope)
        finish_output(ticket, {"exit_code": 0, "stdout": "tests passed"})
        assert "error" in todo_tool("update", item_id=ident, status="completed", summary="tests passed")
        assert todo_tool()["items"][0]["outputs"][0]["result"]["stdout"] == "tests passed"
        assert todo_tool("update", item_id=ident, status="completed", summary="pytest output: tests passed")["status"] == "completed"
        assert todo_snapshot(scope) == ""
        memory_context.set({**scope, "tab_id": "other"})
        assert todo_tool()["items"] == []
        memory_context.set(scope)
        assert todo_tool()["items"][0]["status"] == "completed"
    finally:
        memory_context.reset(token)


def test_executor_records_outputs(tmp_path, monkeypatch):
    from remie.storage.memory import memory_context
    from remie.storage.todo import todo_tool
    monkeypatch.setenv("REMIE_HOME", str(tmp_path))
    async def exercise():
        async def ask(question, options):
            raise AssertionError(question)
        token = memory_context.set({"project_id": "p", "tab_id": "t"})
        try:
            item = todo_tool("add", title="Read and validate")
            todo_tool("update", item_id=item["id"], status="in_progress")
            executor = ToolExecutor(ask_user=ask, run=lambda n, a: {"content": "evidence"}, project_root=tmp_path)
            await executor.execute("read_file", {"filename": str(tmp_path / "file")})
            assert todo_tool()["items"][0]["outputs"][0]["result"] == {"content": "evidence"}
        finally:
            memory_context.reset(token)
    asyncio.run(exercise())
