"""Regression coverage for removal of todo planning and its approval gate."""
import asyncio

from remie.prompts import build_system_prompt
from remie.tools.registry import TOOL_REGISTRY, get_tool_schemas
from remie.tools.executor import ToolExecutor


def test_no_todo_tool_or_prompt():
    assert "todo" not in TOOL_REGISTRY
    assert "todo" not in {schema["name"] for schema in get_tool_schemas()}
    for native in (False, True):
        assert "todo" not in build_system_prompt(native_tools=native).lower()


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
