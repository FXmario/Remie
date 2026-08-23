"""Provider-independent dispatch for model-requested tools."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from remie.tools.registry import TOOL_REGISTRY
from remie.tools.files import (
    edit_file_tool,
    glob_files_tool,
    list_files_tool,
    read_file_tool,
    tree_files_tool,
)
from remie.tools.commands import run_command_tool
from remie.tools.memory import memory_tool


def execute_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one validated tool invocation to its registered handler."""
    if name not in TOOL_REGISTRY:
        return {"action": f"unknown_tool_{name}", "args": args}
    try:
        if name == "read_file":
            filename = args.get("filename") or args.get("path") or "."
            return read_file_tool(filename)
        elif name == "list_files":
            return list_files_tool(args.get("path", "."))
        elif name == "edit_file":
            return edit_file_tool(
                args.get("path", "."),
                args.get("old_str", ""),
                args.get("new_str", ""),
            )
        elif name == "run_command":
            return run_command_tool(
                args.get("command", ""),
                args.get("cwd", "."),
            )
        elif name == "glob_files":
            return glob_files_tool(
                args.get("pattern", ""),
                args.get("path", "."),
            )
        elif name == "tree_files":
            return tree_files_tool(
                args.get("path", "."),
                args.get("max_depth", 3),
            )
        elif name == "ask_user":
            return {"action": "ask_user_interactive", "args": args}
        elif name == "memory":
            return memory_tool(
                args.get("action", "read"),
                args.get("text", ""),
                args.get("id", ""),
                args.get("name", ""),
            )
        return TOOL_REGISTRY[name](**args)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


# Historical name retained for external callers.
run_tool = execute_tool_call


AskUser = Callable[[str, list[str]], Awaitable[str | None]]
ToolFunction = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class ToolExecutor:
    """Execute tools without coupling the agent loop to a particular UI."""

    ask_user: AskUser
    run: ToolFunction = execute_tool_call

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "ask_user":
            answer = await self.ask_user(
                str(args.get("question", "")), list(args.get("options") or [])
            )
            if answer is None:
                return {"answer": None, "cancelled": True}
            return {"answer": answer}
        return await asyncio.to_thread(self.run, name, args)
