"""Tool registry, summaries, string representations, and JSON schemas."""

import inspect
from typing import Any

from remie.tools.ask_user import ask_user_tool
from remie.tools.commands import run_command_tool
from remie.tools.files import (
    edit_file_tool,
    glob_files_tool,
    list_files_tool,
    read_file_tool,
    tree_files_tool,
)
from remie.tools.memory import memory_tool
from remie.tools.test_runner import run_test_shards_tool
from remie.tools.web import web_fetch_tool, web_search_tool

TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
    "run_command": run_command_tool,
    "run_test_shards": run_test_shards_tool,
    "glob_files": glob_files_tool,
    "tree_files": tree_files_tool,
    "ask_user": ask_user_tool,
    "memory": memory_tool,
    "web_fetch": web_fetch_tool,
    "web_search": web_search_tool,
}

TOOL_SUMMARIES = {
    "read_file": "read a file",
    "list_files": "list the files in a directory",
    "edit_file": "edit a file",
    "run_command": "run a shell command",
    "run_test_shards": "run a test suite locally or in four parallel shards",
    "glob_files": "find files matching a glob pattern",
    "tree_files": "show the directory tree",
    "ask_user": "ask the user a question",
    "memory": "saving or recalling a memory note",
    "web_fetch": "fetch a URL over HTTP(S) with curl",
    "web_search": "search the web with DuckDuckGo",
}


def get_tool_summary(name: str) -> str:
    """
    Return a short human-readable summary for a tool name.
    """
    return TOOL_SUMMARIES.get(name, name)


def get_tool_str_representation(tool_name: str) -> str:
    tool = TOOL_REGISTRY[tool_name]
    return f"""
    Name: {tool_name}
    Description: {tool.__doc__}
    Signature: {inspect.signature(tool)}
    """


# JSON Schema for each tool's arguments, used for native function calling
# (Responses API) where the model receives structured tool definitions.
TOOL_PARAMETERS = {
    "read_file": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Path to the file to read.",
            }
        },
        "required": ["filename"],
    },
    "list_files": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list. Defaults to '.'.",
            }
        },
    },
    "glob_files": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern such as '**/*.py'.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search. Defaults to '.'.",
            },
        },
        "required": ["pattern"],
    },
    "tree_files": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root directory. Defaults to '.'.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth of the tree. Defaults to 3.",
            },
        },
    },
    "edit_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit."},
            "old_str": {
                "type": "string",
                "description": "Exact text to replace (must match uniquely).",
            },
            "new_str": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_str", "new_str"],
    },
    "run_command": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "cwd": {
                "type": "string",
                "description": "Working directory. Defaults to '.'.",
            },
        },
        "required": ["command"],
    },
    "run_test_shards": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Test command; shard paths are appended. Defaults to pytest."},
            "cwd": {"type": "string", "description": "Project directory. Defaults to '.'."},
            "threshold_seconds": {"type": "integer", "description": "Minimum estimated sequential duration for four-way execution. Defaults to 120."},
            "estimated_seconds": {"type": "number", "description": "Optional known or historical sequential duration."},
            "patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional test-file discovery glob patterns."},
            "targets": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit shard units (files, packages, modules, or projects); takes precedence over discovery."},
            "worker_timeout_seconds": {"type": "integer", "description": "Timeout for each shard. Defaults to 600 seconds."},
        },
    },
    "ask_user": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question for the user."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional predefined choices to offer.",
            },
        },
        "required": ["question"],
    },
    "memory": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "read", "clear", "delete", "list"],
                "description": "'add' appends a note, 'read' returns it, 'clear' wipes it, 'delete' removes a memory, 'list' lists memories.",
            },
            "text": {"type": "string", "description": "Note text (action 'add' only)."},
            "id": {"type": "string", "description": "Memory uuid; wins over name."},
            "name": {
                "type": "string",
                "description": "Memory name to target or create.",
            },
        },
        "required": ["action"],
    },
    "web_fetch": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http:// or https:// URL to fetch.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "HEAD"],
                "description": "HTTP method. Defaults to GET.",
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional extra request headers.",
            },
            "data": {
                "type": "string",
                "description": "Request body. For GET it is merged into the query string; otherwise sent as the request body.",
            },
            "save_to": {
                "type": "string",
                "description": "Optional project-relative path to save the raw response body to instead of returning it inline (e.g. downloading a file).",
            },
        },
        "required": ["url"],
    },
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query text."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 10).",
            },
        },
        "required": ["query"],
    },
}


def _tool_description(name: str) -> str:
    """First paragraph of the tool's docstring, without param/return lines."""
    doc = TOOL_REGISTRY[name].__doc__ or name
    lines = []
    for raw_line in doc.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith((":param", ":return")):
            break
        lines.append(line)
    return " ".join(lines) or name


def get_tool_schemas() -> list[dict[str, Any]]:
    """Responses-API function-tool definitions for native function calling."""
    return [
        {
            "type": "function",
            "name": name,
            "description": _tool_description(name),
            "parameters": TOOL_PARAMETERS[name],
        }
        for name in TOOL_REGISTRY
    ]
