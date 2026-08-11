import inspect
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import RenderableType
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

load_dotenv()

openai_client = AsyncOpenAI(
    base_url=os.environ["LLAMA_BASE_URL"],
    api_key=os.environ.get("LLAMA_API_KEY", "llama-cpp"),
    http_client=httpx.AsyncClient(verify=False),
)

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks. 
You have access to a series of tools you can execute. Hear are the tools you can execute:

{tool_list_repr}

When you want to use a tool, first provide a short 'thinking:' line explaining your reasoning, then reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.
"""


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


def edit_file_tool(path: str, old_str: str, new_str: str) -> dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file. If old_str is empty,
    create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path to the file and the action taken.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        full_path.write_text(new_str, encoding="utf-8")
        return {"path": str(full_path), "action": "created_file"}
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {"path": str(full_path), "action": "old_str not found"}
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {"path": str(full_path), "action": "edited"}


TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
}

TOOL_SUMMARIES = {
    "read_file": "read a file",
    "list_files": "list the files in a directory",
    "edit_file": "edit a file",
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


def get_full_system_prompt():
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
        tool_str_repr += f"\n{'=' * 15}\n"
    return SYSTEM_PROMPT.format(tool_list_repr=tool_str_repr)


def extract_thinking(text: str) -> str:
    """
    Return the content of all 'thinking:' lines joined together.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("thinking:"):
            continue
        lines.append(line[len("thinking:") :].strip())
    return "\n".join(lines)


def extract_tool_invocations(text: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Return list of (tool_name, args) requested in 'tool: name({...})' lines.
    The parser expects single-line, compact JSON in parentheses.
    """
    invocations = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("tool:"):
            continue
        try:
            after = line[len("tool:") :].strip()
            name, rest = after.split("(", 1)
            name = name.strip()
            if not rest.endswith(")"):
                continue
            json_str = rest[:-1].strip()
            args = json.loads(json_str)
            invocations.append((name, args))
        except ValueError, json.JSONDecodeError:
            continue
    return invocations


async def stream_llm_call(
    conversation: list[ChatCompletionMessageParam],
) -> AsyncIterator[str]:
    stream = await openai_client.chat.completions.create(
        model=os.environ.get("LLAMA_MODEL", "local-model"),
        messages=conversation,
        max_tokens=2000,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def get_connection_error_message(error: Exception) -> str | None:
    """Return a user-facing message for an LLM timeout or connection error."""
    try:
        request = getattr(error, "request", None)
    except RuntimeError:
        request = None
    url = str(getattr(request, "url", "")) or os.environ.get(
        "LLAMA_BASE_URL", "unknown URL"
    )
    if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
        return (
            f"The LLM request to {url} timed out. "
            "Check that the model server is responding."
        )
    if isinstance(error, (APIConnectionError, httpx.TransportError)):
        return (
            f"Could not connect to the LLM server at {url}. Check that it is running."
        )
    return None


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch a tool invocation to the registered tool function.
    """
    if name not in TOOL_REGISTRY:
        return {"action": f"unknown_tool_{name}", "args": args}
    if name == "read_file":
        return read_file_tool(args.get("filename", "."))
    elif name == "list_files":
        return list_files_tool(args.get("path", "."))
    elif name == "edit_file":
        return edit_file_tool(
            args.get("path", "."),
            args.get("old_str", ""),
            args.get("new_str", ""),
        )
    return TOOL_REGISTRY[name](**args)


def render_user_message(text: str) -> Panel:
    """
    Render a user message as a bordered panel.
    """
    return Panel(Text(escape(text)), title="You", border_style="blue", padding=(0, 1))


def render_assistant_message(
    text: str, code_theme: str = "ansi_dark"
) -> RenderableType:
    """Render an assistant response as Markdown with highlighted code."""
    return Markdown(text, code_theme=code_theme, hyperlinks=True)


def render_assistant_panel(text: str, code_theme: str = "ansi_dark") -> Panel:
    """
    Render an assistant message as a bordered panel with code highlighting.
    """
    return Panel(
        render_assistant_message(text, code_theme),
        title="Assistant",
        border_style="yellow",
        padding=(0, 1),
    )
