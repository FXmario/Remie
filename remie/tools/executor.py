"""Provider-independent dispatch for model-requested tools."""

import asyncio
import re
import shlex
import subprocess
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
from remie.tools.common import _project_root, resolve_abs_path, tool_working_directory


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


def _git_worktree_identity(path: Path) -> Path | None:
    """Return the shared Git directory if path belongs to a worktree.

    Use the closest existing directory so new files are covered, but verify
    the requested path is under the reported worktree (not merely beside it).
    A failed lookup must never relax the outside-project permission check.
    """
    path = path.resolve()
    directory = path if path.is_dir() else path.parent
    while not directory.is_dir() and directory != directory.parent:
        directory = directory.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--path-format=absolute",
             "--show-toplevel", "--git-common-dir"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        return None
    root, common_dir = (Path(line).resolve() for line in lines)
    if not _is_within(path, root):
        return None
    return common_dir


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
    project_identity: Path | None = None
    checked_project = False
    for path in candidates:
        if _is_within(path, project_root):
            continue
        if not checked_project:
            project_identity = _git_worktree_identity(project_root)
            checked_project = True
        if project_identity is not None and _git_worktree_identity(path) == project_identity:
            continue
        if path not in outside:
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
        return TOOL_REGISTRY[name](**args)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


# Historical name retained for external callers.
run_tool = execute_tool_call


AskUser = Callable[[str, list[str]], Awaitable[str | None]]
ToolFunction = Callable[[str, dict[str, Any]], dict[str, Any]]
TabStatus = Callable[[], dict[str, Any]]


@dataclass
class ToolExecutor:
    """Execute tools without coupling the agent loop to a particular UI."""

    ask_user: AskUser
    run: ToolFunction = execute_tool_call
    project_root: Path = field(default_factory=_project_root)
    tab_status: TabStatus | None = None
    permission_scope: Callable[[], str] | None = None
    _allowed_directories: dict[str, set[Path]] = field(default_factory=dict, init=False)
    _edit_locks: dict[Path, asyncio.Lock] = field(default_factory=dict, init=False)

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "tab_status" and self.tab_status is not None:
            return self.tab_status()
        if name == "memory" and (args.get("action") or "read") != "read":
            from remie.storage.memory import memory_context

            scope = args.get("scope") or "tab"
            current = await asyncio.to_thread(self.run, "memory", {"scope": scope})
            if "error" in current:
                return current
            action = args.get("action")
            entry = ""
            if action in {"replace", "remove"}:
                old_text = args.get("old_text") or ""
                matches = [part for part in current["content"].split("\n\n---\n\n")
                           if old_text and old_text in part]
                if len(matches) != 1:
                    return {"error": "old_text must identify exactly one entry"}
                entry = matches[0]
            answer = await self.ask_user(
                f"Approve {action} in {scope} memory?\n"
                f"Existing whole entry: {entry}\n"
                f"New content: {args.get('content') or ''}\nDo not save secrets.",
                ["Save memory", "Cancel"],
            )
            if answer != "Save memory":
                return {"error": "Memory update cancelled"}
            # The entry may change while approval is open in another tab.
            # Refuse rather than applying approval to content not reviewed.
            context = memory_context.get()
            if context is None:
                return {"error": "No active memory scope"}
            token = memory_context.set({**context, "expected_memory": current["content"]})
            try:
                return await asyncio.to_thread(self.run, name, args)
            finally:
                memory_context.reset(token)
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

        project_root = tool_working_directory.get() or self.project_root
        outside = _outside_project_paths(name, args, project_root)
        scope = self.permission_scope() if self.permission_scope is not None else "session"
        allowed = self._allowed_directories.setdefault(scope, set())
        outside = [path for path in outside
                   if not any(_is_within(path, directory) for directory in allowed)]
        if outside:
            directory_tool = name in {"list_files", "glob_files", "tree_files", "run_test_shards"}
            command_cwd = (resolve_abs_path(str(args.get("cwd", ".")))
                           if name == "run_command" else None)
            directories = {
                path if directory_tool or path.is_dir() or path == command_cwd else path.parent
                for path in outside
            }
            paths = "\n".join(f"• {path}" for path in outside)
            answer = await self.ask_user(
                f"The agent wants to access path(s) outside the current project "
                f"({project_root}):\n\n{paths}\n\n"
                "Allow once, or always allow these directories and their descendants "
                "for this tab/session:\n"
                + "\n".join(f"• {directory}" for directory in sorted(directories)),
                ["Allow once", "Always allow", "Deny"],
            )
            if answer == "Always allow":
                allowed.update(directories)
            elif answer != "Allow once":
                return {
                    "error": "Permission denied: outside-project access was not approved",
                    "paths": [str(path) for path in outside],
                }
        if name == "edit_file":
            path = resolve_abs_path(str(args.get("path", ".")))
            lock = self._edit_locks.setdefault(path, asyncio.Lock())
            async with lock:
                return await asyncio.to_thread(self.run, name, args)
        return await asyncio.to_thread(self.run, name, args)
