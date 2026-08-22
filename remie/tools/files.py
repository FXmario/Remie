"""File inspection and editing tools."""

import difflib
import fnmatch
import os
from pathlib import Path
from typing import Any

from remie.tools.common import resolve_abs_path

def resolve_abs_path(path_str: str) -> Path:
    """
    file.py -> /Users/home/username/project/file.py
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def read_file_tool(filename: str) -> dict[str, Any]:
    """
    Gets the full content of a file provided by the user.
    :param filename: The name of the file to read.
    :return: The full content of the file.
    """
    full_path = resolve_abs_path(filename)
    with open(str(full_path), "r") as f:
        content = f.read()
    return {"file_path": str(full_path), "content": content}


def list_files_tool(path: str) -> dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    :param path: The path to a directory to list files from.
    :return: A list of files in the directory.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append(
            {"filename": item.name, "type": "file" if item.is_file() else "dir"}
        )
    return {"path": str(full_path), "files": all_files}


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

IGNORED_SUFFIXES = (".egg-info",)


def _is_ignored(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(IGNORED_SUFFIXES)


GLOB_MAX_RESULTS = 200
TREE_MAX_ENTRIES = 500
TREE_MAX_DEPTH = 6


def glob_files_tool(pattern: str, path: str = ".") -> dict[str, Any]:
    """
    Finds files matching a glob pattern, searching recursively.
    :param pattern: The glob pattern to match (e.g. '*.py' or 'src/*.py').
    :param path: The directory to search from (defaults to the project).
    :return: A list of matching file paths relative to the search root.
    """
    full_path = resolve_abs_path(path)
    matches: list[str] = []
    for root, dirs, files in os.walk(full_path):
        dirs[:] = sorted(d for d in dirs if not _is_ignored(d))
        rel_root = os.path.relpath(root, full_path)
        for filename in sorted(files):
            rel_path = os.path.join(rel_root, filename) if rel_root != "." else filename
            rel_path = rel_path.replace(os.sep, "/")
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel_path, pattern):
                matches.append(rel_path)
            if len(matches) >= GLOB_MAX_RESULTS:
                break
        if len(matches) >= GLOB_MAX_RESULTS:
            break
    truncated = len(matches) >= GLOB_MAX_RESULTS
    return {
        "path": str(full_path),
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
        "truncated": truncated,
    }


def _tree_walk(
    directory: Path,
    prefix: str,
    depth: int,
    max_depth: int,
    lines: list[str],
    limit: int,
) -> None:
    entries = sorted(
        (entry for entry in directory.iterdir() if not _is_ignored(entry.name)),
        key=lambda entry: (not entry.is_dir(), entry.name.lower()),
    )
    for index, entry in enumerate(entries):
        if len(lines) >= limit:
            return
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir() and depth < max_depth:
            _tree_walk(
                entry,
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                max_depth,
                lines,
                limit,
            )


def tree_files_tool(path: str = ".", max_depth: int = 3) -> dict[str, Any]:
    """
    Shows the directory tree of a path.
    :param path: The directory to show the tree of (defaults to the project).
    :param max_depth: How many levels of subdirectories to descend into (max 6).
    :return: The tree rendered as text.
    """
    full_path = resolve_abs_path(path)
    depth = max(1, min(int(max_depth), TREE_MAX_DEPTH))
    lines = [f"{full_path.name}/"]
    _tree_walk(full_path, "", 1, depth, lines, TREE_MAX_ENTRIES)
    truncated = len(lines) >= TREE_MAX_ENTRIES
    if truncated:
        lines.append("[... truncated ...]")
    return {
        "path": str(full_path),
        "tree": "\n".join(lines),
        "truncated": truncated,
    }


DIFF_MAX_CHARS = 4000


def _make_diff(original: str, edited: str, path: str) -> str:
    """
    Build a unified diff between two file contents, truncated to DIFF_MAX_CHARS.
    """
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            edited.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    diff = "".join(lines)
    if len(diff) > DIFF_MAX_CHARS:
        diff = diff[:DIFF_MAX_CHARS].rstrip("\n") + "\n...diff truncated...\n"
    return diff


def edit_file_tool(path: str, old_str: str, new_str: str) -> dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file. If old_str is empty,
    create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path to the file, the action taken, and a diff.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        original = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
        full_path.write_text(new_str, encoding="utf-8")
        return {
            "path": str(full_path),
            "action": "created_file",
            "diff": _make_diff(original, new_str, str(full_path)),
        }
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {"path": str(full_path), "action": "old_str not found"}
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {
        "path": str(full_path),
        "action": "edited",
        "diff": _make_diff(original, edited, str(full_path)),
    }


