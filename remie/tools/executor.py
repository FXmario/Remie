"""Provider-independent dispatch for model-requested tools."""

import asyncio
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remie.tools.registry import TOOL_REGISTRY
from remie.tools.files import (
    edit_file_tool,
    glob_files_tool,
    list_files_tool,
    read_file_tool,
    tree_files_tool,
)
from remie.tools.commands import get_blocked_command_reason, run_command_tool
from remie.tools.memory import memory_tool
from remie.tools.common import _project_root, resolve_abs_path


_PATH_ARGUMENTS = {
    "read_file": ("filename", "path"),
    "list_files": ("path",),
    "edit_file": ("path",),
    "glob_files": ("path",),
    "tree_files": ("path",),
    "web_fetch": ("save_to",),
    "run_test_shards": ("cwd",),
}
_COMMAND_PATH = re.compile(r"(?<![\w:/])(?:~(?:/|$)|/|\.\.(?:/|$))[^\s;|&<>]*")


def _is_within(path: Path, project_root: Path) -> bool:
    try:
        path.resolve().relative_to(project_root.resolve())
        return True
    except ValueError:
        return False


def _outside_project_paths(
    name: str, args: dict[str, Any], project_root: Path
) -> list[Path]:
    """Return explicit paths a tool invocation targets outside its project."""
    candidates: list[Path] = []
    for key in _PATH_ARGUMENTS.get(name, ()):
        value = args.get(key)
        if isinstance(value, str) and value:
            candidates.append(resolve_abs_path(value))

    if name == "run_command":
        cwd = resolve_abs_path(str(args.get("cwd", ".")))
        candidates.append(cwd)
        command = str(args.get("command", ""))
        # Inspect shell words as well as embedded forms such as --file=/tmp/x.
        try:
            command = " ".join(shlex.split(command))
        except ValueError:
            pass
        for match in _COMMAND_PATH.finditer(command):
            raw = match.group(0).rstrip(",)]}'\"")
            path = Path(raw).expanduser()
            candidates.append((cwd / path).resolve() if not path.is_absolute() else path.resolve())

    outside: list[Path] = []
    for path in candidates:
        if not _is_within(path, project_root) and path not in outside:
            outside.append(path)
    return outside


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
    project_root: Path = field(default_factory=_project_root)

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "ask_user":
            answer = await self.ask_user(
                str(args.get("question", "")), list(args.get("options") or [])
            )
            if answer is None:
                return {"answer": None, "cancelled": True}
            return {"answer": answer}

        # Reject known-destructive commands before asking for path permission.
        # Otherwise `rm -rf /` is mistaken for an outside-path request and the
        # UI never receives the structured `blocked` result it should display.
        if name == "run_command" and get_blocked_command_reason(
            str(args.get("command", ""))
        ) is not None:
            return await asyncio.to_thread(self.run, name, args)

        outside = _outside_project_paths(name, args, self.project_root)
        if outside:
            paths = "\n".join(f"• {path}" for path in outside)
            answer = await self.ask_user(
                f"The agent wants to access path(s) outside the current project "
                f"({self.project_root}):\n\n{paths}\n\nAllow this operation once?",
                ["Allow once", "Deny"],
            )
            if answer != "Allow once":
                return {
                    "error": "Permission denied: outside-project access was not approved",
                    "paths": [str(path) for path in outside],
                }
        return await asyncio.to_thread(self.run, name, args)
