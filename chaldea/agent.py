import ast
import inspect
import json
import os
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
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

CONFIG_DIR = Path(
    os.environ.get("CHALDEA_CONFIG_DIR", "~/.config/fuiagent")
).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.json"

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

OPENCODE_GO_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "grok-4.5",
    "glm-5.2",
    "glm-5.1",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "hy3",
]


class UnsupportedModelError(RuntimeError):
    """Raised when a configured provider needs an unsupported API format."""


@dataclass
class ConnectionConfig:
    base_url: str
    api_key: str
    model: str


def _default_config() -> ConnectionConfig:
    return ConnectionConfig(
        base_url=os.environ.get("LLAMA_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.environ.get("LLAMA_API_KEY", "llama-cpp"),
        model=os.environ.get("LLAMA_MODEL", "local-model"),
    )


def load_config() -> ConnectionConfig:
    """Load saved connection config, falling back to environment defaults."""
    try:
        data = json.loads(CONFIG_FILE.read_text())
        return ConnectionConfig(
            base_url=data.get("base_url", ""),
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
        )
    except OSError, json.JSONDecodeError:
        return _default_config()


def save_config(config: ConnectionConfig) -> None:
    """Persist connection config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(config), indent=2))


_config = load_config()

openai_client = AsyncOpenAI(
    base_url=_config.base_url,
    api_key=_config.api_key,
    http_client=httpx.AsyncClient(verify=False),
)


def get_config() -> ConnectionConfig:
    """Return the current active connection config."""
    return _config


def configure_openai(base_url: str, api_key: str, model: str) -> ConnectionConfig:
    """Rebuild the OpenAI client with a new connection configuration."""
    global _config, openai_client
    _config = ConnectionConfig(base_url=base_url, api_key=api_key, model=model)
    openai_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.AsyncClient(verify=False),
    )
    return _config


async def fetch_opencode_go_models(api_key: str) -> list[str]:
    """Fetch the OpenCode Go model list, falling back to a hardcoded list."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(
                f"{OPENCODE_GO_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            models = [
                item["id"]
                for item in payload.get("data", [])
                if item.get("id") in OPENCODE_GO_MODELS
            ]
            return models or list(OPENCODE_GO_MODELS)
        except httpx.HTTPError, ValueError, KeyError, TypeError:
            return list(OPENCODE_GO_MODELS)


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
    Supports compact JSON and Python-style keyword arguments.
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
            args_text = rest[:-1].strip()
            try:
                args = json.loads(args_text)
            except json.JSONDecodeError:
                call = ast.parse(f"tool({args_text})", mode="eval").body
                if not isinstance(call, ast.Call):
                    continue
                if call.args:
                    args = ast.literal_eval(call.args[0])
                else:
                    args = {
                        keyword.arg: ast.literal_eval(keyword.value)
                        for keyword in call.keywords
                        if keyword.arg is not None
                    }
            if not isinstance(args, dict):
                continue
            invocations.append((name, args))
        except SyntaxError, ValueError, TypeError, json.JSONDecodeError:
            continue
    return invocations


async def stream_llm_call(
    conversation: list[ChatCompletionMessageParam],
) -> AsyncIterator[str]:
    if (
        _config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL
        and _config.model not in OPENCODE_GO_MODELS
    ):
        raise UnsupportedModelError(
            f"The model '{_config.model}' does not support Chat Completions yet."
        )
    stream = await openai_client.chat.completions.create(
        model=_config.model,
        messages=conversation,
        max_tokens=2000,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def get_connection_error_message(error: Exception) -> str | None:
    """Return a user-facing message for an LLM timeout or connection error."""
    try:
        request = getattr(error, "request", None)
    except RuntimeError:
        request = None
    url = str(getattr(request, "url", "")) or _config.base_url
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
        return TOOL_REGISTRY[name](**args)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


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
