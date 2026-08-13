import ast
import json
import os
import re
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

from fuica.tools import (
    RUN_COMMAND_MAX_OUTPUT,
    RUN_COMMAND_TIMEOUT,
    TOOL_REGISTRY,
    TOOL_SUMMARIES,
    ask_user_tool,
    edit_file_tool,
    get_blocked_command_reason,
    get_custom_blocked_commands,
    get_tool_str_representation,
    get_tool_summary,
    glob_files_tool,
    list_files_tool,
    read_file_tool,
    resolve_abs_path,
    run_command_tool,
    tree_files_tool,
)

load_dotenv()

CONFIG_DIR = Path(
    os.environ.get("FUICA_CONFIG_DIR", "~/.config/fuiagent")
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

TRUNCATED_REASONS = {"length", "max_tokens", "max_completion_tokens"}


def get_max_output_tokens(provider: str = "local") -> int:
    """Per-provider output token budget. Env override wins; OpenCode Go gets a
    large budget so long responses aren't cut off by an artificial cap."""
    env_value = os.environ.get("FUICA_MAX_OUTPUT_TOKENS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 32_768 if provider == "opencode-go" else 8_192

OPENCODE_GO_DEFAULT_CONTEXT_LIMIT = 128_000
PROJECT_CONTEXT_MAX_CHARS = 8000
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "grok-4.5": 256_000,
    "glm-5.2": 128_000,
    "glm-5.1": 128_000,
    "kimi-k3": 256_000,
    "kimi-k2.7-code": 128_000,
    "kimi-k2.6": 128_000,
    "mimo-v2.5": 128_000,
    "mimo-v2.5-pro": 128_000,
    "hy3": 128_000,
    "deepseek-v4-pro": 128_000,
    "deepseek-v4-flash": 128_000,
}


class UnsupportedModelError(RuntimeError):
    """Raised when a configured provider needs an unsupported API format."""


@dataclass
class ConnectionConfig:
    base_url: str
    api_key: str
    model: str
    provider: str = "local"
    reasoning_effort: str = "medium"


def _default_config() -> ConnectionConfig:
    base_url = os.environ.get("LLAMA_BASE_URL", "http://localhost:7070/v1")
    provider = (
        "opencode-go" if base_url.rstrip("/") == OPENCODE_GO_BASE_URL else "local"
    )
    return ConnectionConfig(
        base_url=base_url,
        api_key=os.environ.get("LLAMA_API_KEY", "llama-cpp"),
        model=os.environ.get("LLAMA_MODEL", "local-model"),
        provider=provider,
        reasoning_effort=os.environ.get("FUICA_REASONING_EFFORT", "medium"),
    )


def load_config() -> ConnectionConfig:
    """Load saved connection config, falling back to environment defaults."""
    try:
        data = json.loads(CONFIG_FILE.read_text())
        base_url = data.get("base_url", "")
        provider = data.get(
            "provider",
            "opencode-go"
            if base_url.rstrip("/") == OPENCODE_GO_BASE_URL
            else "local",
        )
        return ConnectionConfig(
            base_url=base_url,
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
            provider=provider,
            reasoning_effort=data.get("reasoning_effort", "medium"),
        )
    except (OSError, json.JSONDecodeError):
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


def configure_openai(
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "local",
    reasoning_effort: str = "medium",
) -> ConnectionConfig:
    """Rebuild the OpenAI client with a new connection configuration."""
    global _config, openai_client
    _config = ConnectionConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
    )
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
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return list(OPENCODE_GO_MODELS)


def get_model_context_limit(model: str, provider: str = "local") -> int | None:
    """Best-known context window for a model/provider pair (used for compaction)."""
    if provider != "opencode-go":
        return None
    return MODEL_CONTEXT_LIMITS.get(model, OPENCODE_GO_DEFAULT_CONTEXT_LIMIT)


SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks. 
You have access to a series of tools you can execute. Hear are the tools you can execute:

{tool_list_repr}

When you want to use a tool, first provide a short 'thinking:' line explaining your reasoning, then reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.

When multiple valid approaches have meaningful tradeoffs or require a user preference, do not choose silently. Briefly explain the options and ask the user which they prefer. Continue autonomously for routine implementation details or when one option clearly dominates. Do not ask unnecessary confirmation questions.
To ask the user a question, call the 'ask_user' tool and wait for its result instead of ending your turn.
"""


def load_project_context() -> str:
    """
    Load project instructions from AGENTS.md in the launch directory.
    Returns an empty string when there is no AGENTS.md.
    """
    agents_file = Path.cwd() / "AGENTS.md"
    if not agents_file.is_file():
        return ""
    try:
        content = agents_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    if len(content) > PROJECT_CONTEXT_MAX_CHARS:
        content = (
            content[:PROJECT_CONTEXT_MAX_CHARS].rstrip()
            + "\n\n(AGENTS.md truncated for context.)\n"
        )
    return f"\n\n## Project instructions (from AGENTS.md)\n{content}"


def get_full_system_prompt():
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
        tool_str_repr += f"\n{'=' * 15}\n"
    return SYSTEM_PROMPT.format(tool_list_repr=tool_str_repr) + load_project_context()


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


def _normalize_tool_line(line: str) -> str | None:
    """
    Normalize a tool invocation line to 'tool: name(args)' form, or None if the
    line is not a tool call. Handles plain, angle-bracket-wrapped, and
    self-closing variants like '<tool: name(args)>' and '<tool: name(args) />'.
    """
    if line.startswith("</tool") or line.startswith("tool>"):
        return None
    if line.startswith("<tool:"):
        inner = line[1:]
        if inner.endswith(">"):
            inner = inner[:-1]
        inner = inner.rstrip()
        if inner.endswith("/"):
            inner = inner[:-1].rstrip()
        return inner
    if line.startswith("tool:"):
        return line
    return None


def extract_tool_invocations(text: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Return list of (tool_name, args) requested in 'tool: name({...})' lines.
    Supports plain and angle-bracket-wrapped forms, compact JSON, and
    Python-style keyword arguments. Dashed names are normalized to underscores.
    """
    invocations = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = _normalize_tool_line(line)
        if normalized is None:
            continue
        try:
            after = normalized[len("tool:") :].strip()
            name, rest = after.split("(", 1)
            name = name.strip().replace("-", "_")
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
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
    invocations.extend(extract_dsml_invocations(text))
    return invocations


def extract_dsml_invocations(text: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Parse DSML tool-call markup like:

        <|DSML|>invoke name="list-files">
        <|DSML|>parameter path="." />

    Returns a list of (tool_name, args) with dashed names normalized to
    underscores (list-files -> list_files).
    """
    invocations: list[tuple[str, dict[str, Any]]] = []
    current_name: str | None = None
    current_args: dict[str, Any] = {}

    def flush() -> None:
        nonlocal current_name, current_args
        if current_name is not None:
            invocations.append((current_name, current_args))
        current_name = None
        current_args = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("<|DSML|>"):
            continue
        body = line[len("<|DSML|>") :].strip()
        if body.startswith("invoke name="):
            flush()
            match = re.match(r'invoke name="([^"]+)"', body)
            if not match:
                continue
            current_name = match.group(1).replace("-", "_")
            current_args = {}
        elif body.startswith("parameter") and current_name is not None:
            param_text = body[len("parameter") :].strip()
            param_text = re.sub(r"/?>\s*$", "", param_text).strip()
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", param_text)
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value.strip('"').strip("'")
            current_args[key] = parsed
        elif body.startswith("/tool_calls") or body.startswith("tool_calls"):
            flush()
    flush()
    return invocations


async def stream_llm_call(
    conversation: list[ChatCompletionMessageParam],
    usage_box: dict[str, int] | None = None,
    reasoning_box: list[str] | None = None,
    finish_box: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    if (
        _config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL
        and _config.model not in OPENCODE_GO_MODELS
    ):
        raise UnsupportedModelError(
            f"The model '{_config.model}' does not support Chat Completions yet."
        )
    kwargs: dict[str, Any] = {
        "model": _config.model,
        "messages": conversation,
        "max_tokens": get_max_output_tokens(_config.provider),
        "stream": True,
    }
    if _config.reasoning_effort != "off":
        kwargs["reasoning_effort"] = _config.reasoning_effort
    if usage_box is not None and _config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL:
        kwargs["stream_options"] = {"include_usage": True}
    stream = await openai_client.chat.completions.create(**kwargs)
    async for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage_box is not None and usage is not None:
            usage_box["prompt_tokens"] = usage.prompt_tokens or 0
            usage_box["completion_tokens"] = usage.completion_tokens or 0
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if finish_box is not None:
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                finish_box["finish_reason"] = finish_reason
                finish_box["truncated"] = finish_reason in TRUNCATED_REASONS
        if reasoning_box is not None:
            reason = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reason:
                reasoning_box.append(reason)
        content = delta.content
        if content:
            yield content


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


def render_user_message(text: str) -> Panel:
    """
    Render a user message as a bordered panel.
    """
    return Panel(Text(escape(text)), title="You", border_style="blue", padding=(0, 1))


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate for a piece of text, used when the API does not
    report exact usage. Based on the ~4 chars/token heuristic.
    """
    if not text:
        return 0
    return max(1, len(text) // 4 + text.count("\n") // 3)


def estimate_conversation_tokens(
    conversation: list[ChatCompletionMessageParam],
) -> int:
    """
    Rough token estimate for the whole conversation, summing string content
    (including tool_result(...) messages).
    """
    total = 0
    for message in conversation:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
                elif hasattr(part, "text") and part.text:
                    total += estimate_tokens(part.text)
    return total


def strip_protocol_lines(text: str) -> str:
    """
    Remove 'thinking:', 'tool:', angle-wrapped tool, and DSML protocol lines for
    display. They stay in the conversation history sent to the model.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            line.startswith("thinking:")
            or line.startswith("tool:")
            or line.startswith("<tool:")
            or line.startswith("</tool")
            or line.startswith("<|DSML|>")
        ):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


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
