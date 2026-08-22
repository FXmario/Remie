import ast
import json
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI
from rich.console import RenderableType
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from remie.model_names import ModelInfo, prettify_model_id, prettify_model_name
from remie.tools import (
    TOOL_REGISTRY,
    edit_file_tool,
    find_memory_by_id,
    get_active_memory_id,
    get_tool_str_representation,
    glob_files_tool,
    list_files_tool,
    memory_file_path,
    memory_tool,
    read_file_tool,
    run_command_tool,
    tree_files_tool,
)

load_dotenv()

CONFIG_DIR = Path(
    os.environ.get("REMIE_CONFIG_DIR", "~/.config/remie")
).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.json"

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_BASE_URL = "http://localhost:7070/v1"
CODEX_BACKEND_BASE = "https://chatgpt.com/backend-api/codex"
CONFIG_VERSION = 2
SUPPORTED_PROVIDERS = ("local", "opencode-go", "codex", "openrouter")
STATUS_ANIMATION_CONFIG_KEY = "status_animation"

# Codex (ChatGPT subscription) models. The live list for the signed-in account
# is fetched when connecting (GET /codex/models?client_version=...); this
# bundled list is only a fallback when the backend is unreachable.
CODEX_MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
]
CODEX_DEFAULT_CONTEXT_LIMIT = 272_000

# OpenRouter models, bundled only as a fallback: the live catalog (with each
# model's real context window) is fetched from the public /models endpoint
# when connecting. Ids follow OpenRouter's "vendor/model" convention.
OPENROUTER_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.6",
    "google/gemini-3-pro",
    "deepseek/deepseek-v4",
    "x-ai/grok-5",
    "qwen/qwen4-max",
]
OPENROUTER_DEFAULT_CONTEXT_LIMIT = 128_000

# Live context windows for OpenRouter models, populated from the public
# models API at connect time (used for context compaction).
_openrouter_model_context: dict[str, int] = {}

# Live context windows for Codex subscription models, populated from the
# account's model list at connect time.
_codex_model_context: dict[str, int] = {}

# Display metadata per model id, populated by the provider fetchers. Used by
# UI surfaces (badge, toasts) that only receive the raw id; ids never fetched
# fall back to heuristic prettification.
_model_info_cache: dict[str, ModelInfo] = {}


def _cache_model_info(info: ModelInfo) -> ModelInfo:
    if info.id:
        _model_info_cache[info.id] = info
    return info


def get_model_info(model_id: str) -> ModelInfo:
    """Display metadata for a raw model id.

    Returns the cached catalog row when a fetcher has seen the id; otherwise
    falls back to heuristic prettification so hand-typed local models still
    render readably.
    """
    cached = _model_info_cache.get(model_id)
    if cached is not None:
        return cached
    return prettify_model_id(model_id)

# Bundled fallback model list, used only when the OpenCode Go models API is
# unreachable. The live list (and each model's context window) is fetched from
# the API when connecting, so this does not restrict which models can be used.
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

# OpenCode Go models served via the Anthropic /messages or OpenAI /responses
# endpoints. They do not accept OpenAI's `reasoning_effort` parameter, so the
# reasoning-effort picker in the connection modal should be faded for them.
# Derived from the OpenCode Go endpoint mapping in the docs (aug 2026).
NON_REASONING_EFFORT_MODELS = {
    "grok-4.5",
    "gpt-5.6-luna",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.8-max",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.5-plus",
}


def supports_reasoning_effort(model: str, provider: str = "local") -> bool:
    """Return True when a connection can send `reasoning_effort`.

    Local (llama.cpp) servers vary by build and model, so reasoning effort is
    always offered there. For OpenCode Go, only models on the
    `/chat/completions` endpoint accept the parameter; unknown models default to
    supported.
    """
    if provider != "opencode-go":
        return True
    return model not in NON_REASONING_EFFORT_MODELS


def get_max_output_tokens(provider: str = "local") -> int:
    """Per-provider output token budget. Env override wins; OpenCode Go gets a
    large budget so long responses aren't cut off by an artificial cap."""
    env_value = os.environ.get("REMIE_MAX_OUTPUT_TOKENS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 32_768 if provider in ("opencode-go", "openrouter") else 8_192


OPENCODE_GO_DEFAULT_CONTEXT_LIMIT = 128_000
PROJECT_CONTEXT_MAX_CHARS = 8000
MEMORY_MAX_CHARS = 4000

# Live context windows for OpenCode Go models, populated from the models API
# (each model reports its own context_length). Used for context compaction so
# the budget tracks the actual model instead of a hardcoded table.
_opencode_go_model_context: dict[str, int] = {}


class UnsupportedModelError(RuntimeError):
    """Raised when a configured provider needs an unsupported API format."""


class LLMRequestError(RuntimeError):
    """Raised when the LLM server responds with a non-2xx status."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


@dataclass
class ConnectionConfig:
    base_url: str
    api_key: str
    model: str
    provider: str = "local"
    reasoning_effort: str = "medium"
    verify_ssl: bool = False


def _provider_defaults(provider: str) -> ConnectionConfig:
    if provider == "codex":
        return ConnectionConfig(
            CODEX_BACKEND_BASE,
            "",
            CODEX_MODELS[0],
            "codex",
            "medium",
            True,
        )
    if provider == "openrouter":
        return ConnectionConfig(
            OPENROUTER_BASE_URL,
            "",
            OPENROUTER_MODELS[0],
            "openrouter",
            "medium",
            True,
        )
    if provider == "opencode-go":
        return ConnectionConfig(
            OPENCODE_GO_BASE_URL,
            "",
            OPENCODE_GO_MODELS[0],
            "opencode-go",
            "medium",
            True,
        )
    return ConnectionConfig(
        os.environ.get("LLAMA_BASE_URL", LOCAL_BASE_URL),
        os.environ.get("LLAMA_API_KEY", "llama-cpp"),
        os.environ.get("LLAMA_MODEL", "local-model"),
        "local",
        os.environ.get("REMIE_REASONING_EFFORT", "medium"),
        False,
    )


def _default_config() -> ConnectionConfig:
    llama_base_url = os.environ.get("LLAMA_BASE_URL")
    if llama_base_url:
        base_url = llama_base_url
        provider = "local"
        api_key = os.environ.get("LLAMA_API_KEY", "llama-cpp")
        model = os.environ.get("LLAMA_MODEL", "local-model")
    else:
        base_url = LOCAL_BASE_URL
        provider = "local"
        api_key = os.environ.get("LLAMA_API_KEY", "llama-cpp")
        model = os.environ.get("LLAMA_MODEL", "local-model")
    return ConnectionConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        reasoning_effort=os.environ.get("REMIE_REASONING_EFFORT", "medium"),
        verify_ssl=False,
    )


def load_config() -> ConnectionConfig:
    """Load saved connection config, falling back to environment defaults."""
    try:
        data = json.loads(CONFIG_FILE.read_text())
        providers = data.get("providers")
        if isinstance(providers, dict):
            active_provider = data.get("active_provider") or data.get("provider")
            if active_provider in SUPPORTED_PROVIDERS:
                profile = providers.get(active_provider, {})
                defaults = _provider_defaults(active_provider)
                return ConnectionConfig(
                    profile.get("base_url", defaults.base_url),
                    profile.get("api_key", defaults.api_key),
                    profile.get("model", defaults.model),
                    active_provider,
                    profile.get("reasoning_effort", defaults.reasoning_effort),
                    bool(profile.get("verify_ssl", defaults.verify_ssl)),
                )
        base_url = data.get("base_url", "")
        provider = data.get(
            "provider",
            "opencode-go"
            if base_url.rstrip("/") == OPENCODE_GO_BASE_URL
            else "local",
        )
        if provider not in SUPPORTED_PROVIDERS:
            return _provider_defaults("local")
        return ConnectionConfig(
            base_url=base_url,
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
            provider=provider,
            reasoning_effort=data.get("reasoning_effort", "medium"),
            verify_ssl=bool(data.get("verify_ssl", False)),
        )
    except (OSError, json.JSONDecodeError):
        return _default_config()


def save_config(config: ConnectionConfig) -> None:
    """Persist the active connection while retaining all provider profiles."""
    profiles = load_provider_configs()
    profiles[config.provider] = config
    save_provider_configs(profiles, config.provider)


def load_provider_configs() -> dict[str, ConnectionConfig]:
    """Load provider profiles, migrating the old single-profile format."""
    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        active = _default_config()
        return {
            provider: _provider_defaults(provider) for provider in SUPPORTED_PROVIDERS
        } | {active.provider: active}

    stored = data.get("providers")
    if isinstance(stored, dict):
        # Every supported provider gets an entry so newly-added providers
        # (e.g. codex) have defaults even without a saved profile.
        profiles: dict[str, ConnectionConfig] = {
            provider: _provider_defaults(provider) for provider in SUPPORTED_PROVIDERS
        }
        for provider in SUPPORTED_PROVIDERS:
            profile = stored.get(provider)
            if not isinstance(profile, dict):
                continue
            defaults = profiles[provider]
            profiles[provider] = ConnectionConfig(
                profile.get("base_url", defaults.base_url),
                profile.get("api_key", defaults.api_key),
                profile.get("model", defaults.model),
                provider,
                profile.get("reasoning_effort", defaults.reasoning_effort),
                bool(profile.get("verify_ssl", defaults.verify_ssl)),
            )
        return profiles

    # Legacy config.json contained only the active connection.
    legacy = load_config()
    return {
        provider: _provider_defaults(provider) for provider in SUPPORTED_PROVIDERS
    } | {legacy.provider: legacy}


def save_provider_configs(
    profiles: dict[str, ConnectionConfig], active_provider: str
) -> None:
    """Persist all provider profiles and the provider currently in use."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CONFIG_VERSION,
        "active_provider": active_provider,
        STATUS_ANIMATION_CONFIG_KEY: load_status_animation_enabled(),
        "providers": {
            provider: {
                "base_url": config.base_url,
                "api_key": config.api_key,
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "verify_ssl": config.verify_ssl,
            }
            for provider, config in profiles.items()
            if provider in SUPPORTED_PROVIDERS
        },
    }
    CONFIG_FILE.write_text(json.dumps(payload, indent=2))


def load_status_animation_enabled() -> bool:
    """Return the persisted status GIF preference, enabled by default."""
    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get(STATUS_ANIMATION_CONFIG_KEY, True))


def save_status_animation_enabled(enabled: bool) -> None:
    """Persist the status GIF preference without changing provider settings."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    data[STATUS_ANIMATION_CONFIG_KEY] = bool(enabled)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


_config = load_config()

# Shared HTTP timeout for chat-completion requests. The read timeout applies
# between SSE chunks, so a stalled stream still errors out while a slow but
# alive stream keeps flowing.
HTTP_TIMEOUT = httpx.Timeout(connect=10, read=600, write=60, pool=10)

# Local (llama.cpp) servers commonly use self-signed certificates, so TLS
# verification stays disabled for the local client to match the previous SDK
# behavior. Remote providers (e.g. OpenCode Go) use a separate client with
# certificate verification enabled.
http_client = httpx.AsyncClient(
    verify=False,
    timeout=HTTP_TIMEOUT,
)

_verified_local_client: httpx.AsyncClient | None = None
_remote_client: httpx.AsyncClient | None = None
_local_openai_client: AsyncOpenAI | None = None
_local_openai_client_key: tuple[str, str, bool] | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return the HTTP client for the active connection.

    Local providers keep TLS verification disabled (self-signed certs are
    common on llama.cpp servers); any other provider uses a lazily-created
    client with certificate verification enabled.
    """
    global _remote_client, _verified_local_client
    if _config.provider == "local":
        if _config.verify_ssl:
            if _verified_local_client is None:
                _verified_local_client = httpx.AsyncClient(
                    verify=True,
                    timeout=HTTP_TIMEOUT,
                )
            return _verified_local_client
        return http_client
    if _remote_client is None:
        _remote_client = httpx.AsyncClient(
            verify=True,
            timeout=HTTP_TIMEOUT,
        )
    return _remote_client


def get_config() -> ConnectionConfig:
    """Return the current active connection config."""
    return _config


def configure_openai(
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "local",
    reasoning_effort: str = "medium",
    verify_ssl: bool = False,
) -> ConnectionConfig:
    """Update the active connection configuration."""
    global _config, _local_openai_client, _local_openai_client_key
    _config = ConnectionConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        verify_ssl=verify_ssl,
    )
    _local_openai_client = None
    _local_openai_client_key = None
    return _config


def _get_local_openai_client() -> AsyncOpenAI:
    """Return an OpenAI SDK client pointed at the local compatible server."""
    global _local_openai_client, _local_openai_client_key
    key = (_config.base_url, _config.api_key, _config.verify_ssl)
    if _local_openai_client is None or _local_openai_client_key != key:
        _local_openai_client = AsyncOpenAI(
            api_key=_config.api_key,
            base_url=_config.base_url,
            http_client=_get_http_client(),
            max_retries=0,
        )
        _local_openai_client_key = key
    return _local_openai_client


async def fetch_opencode_go_models(api_key: str) -> list[ModelInfo]:
    """Fetch the live OpenCode Go model list with each model's real context
    window (cached for compaction); fall back to prettified bundled ids when
    the API is unreachable. The catalog exposes raw ids only, so display names
    come from heuristics."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(
                f"{OPENCODE_GO_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            infos: list[ModelInfo] = []
            for item in payload.get("data", []):
                model_id = item.get("id")
                if not model_id:
                    continue
                context = item.get("context_length")
                if isinstance(context, int) and context > 0:
                    _opencode_go_model_context[model_id] = context
                infos.append(_cache_model_info(prettify_model_id(str(model_id))))
            if not infos:
                return [_cache_model_info(prettify_model_id(m)) for m in OPENCODE_GO_MODELS]
            return infos
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            return [_cache_model_info(prettify_model_id(m)) for m in OPENCODE_GO_MODELS]


async def fetch_codex_models() -> list[ModelInfo]:
    """Fetch the signed-in account's live Codex models as ModelInfo rows;
    fall back to the bundled list when offline or unsigned."""
    from remie.codex_client import fetch_codex_models as _fetch_live_models

    try:
        rows = await _fetch_live_models()
    except Exception:
        rows = []
    if not rows:
        return [
            _cache_model_info(
                ModelInfo(id=m, display=prettify_model_name(m), vendor="OpenAI")
            )
            for m in CODEX_MODELS
        ]
    infos: list[ModelInfo] = []
    for row in rows:
        context = row.get("context_window") or 0
        if isinstance(context, int) and context > 0:
            _codex_model_context[row["id"]] = context
        infos.append(
            _cache_model_info(
                ModelInfo(
                    id=row["id"],
                    display=row.get("display") or row["id"],
                    vendor="OpenAI",
                )
            )
        )
    return infos


async def fetch_openrouter_models() -> list[ModelInfo]:
    """Fetch the live OpenRouter catalog; fall back to prettified bundled ids
    on failure. Context windows are cached for compaction."""
    from remie.openrouter_client import fetch_openrouter_models as _fetch_live

    try:
        rows = await _fetch_live()
    except Exception:
        rows = []
    if not rows:
        return [_cache_model_info(prettify_model_id(m)) for m in OPENROUTER_MODELS]
    infos: list[ModelInfo] = []
    for row in rows:
        context = row.get("context_length") or 0
        if isinstance(context, int) and context > 0:
            _openrouter_model_context[row["id"]] = context
        infos.append(
            _cache_model_info(
                ModelInfo(
                    id=row["id"],
                    display=row.get("display") or row["id"],
                    vendor=row.get("vendor") or "",
                    free=bool(row.get("free")),
                )
            )
        )
    return infos


def get_model_context_limit(model: str, provider: str = "local") -> int | None:
    """Best-known context window for a model/provider pair (used for compaction).

    OpenCode Go and OpenRouter context windows come from the live model lists
    fetched at connect time; models without a reported window fall back to the
    provider default. Codex subscription models expose a 272k input window.
    """
    if provider == "codex":
        return CODEX_DEFAULT_CONTEXT_LIMIT
    if provider == "openrouter":
        return _openrouter_model_context.get(
            model, OPENROUTER_DEFAULT_CONTEXT_LIMIT
        )
    if provider != "opencode-go":
        return None
    return _opencode_go_model_context.get(model, OPENCODE_GO_DEFAULT_CONTEXT_LIMIT)


SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks. 
You have access to a series of tools you can execute. Here are the tools you can execute:

{tool_list_repr}

When you want to use a tool, first provide a short 'thinking:' line explaining your reasoning, then reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.

When multiple valid approaches have meaningful tradeoffs or require a user preference, do not choose silently. Briefly explain the options and ask the user which they prefer. Continue autonomously for routine implementation details or when one option clearly dominates. Do not ask unnecessary confirmation questions.
To ask the user a question, call the 'ask_user' tool and wait for its result instead of ending your turn.

Use the 'memory' tool to persist durable facts, decisions, user preferences, and open tasks that should be remembered across chats. Add a note when you learn something that will matter later; do not log routine progress and do not use memory as a chat transcript. Remie keeps an active project memory, so use memory(action="add", text=...) without a name to append to it. Use memory(action="list") to see older memories (each with an id and a name), and target one by name or id only when needed; memory(action="delete", name=...) removes a memory entirely.
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


def load_agent_memory() -> str:
    """
    Load the agent's active memory notes from .remie/memory/<uuid>.md in the
    launch directory. Returns an empty string when there is no active memory.
    """
    memory_id = get_active_memory_id()
    if not memory_id:
        return ""
    memory = find_memory_by_id(memory_id)
    if memory is None:
        return ""
    memory_file = memory_file_path(memory_id)
    if not memory_file.is_file():
        return ""
    try:
        content = memory_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    if not content.strip():
        return ""
    if len(content) > MEMORY_MAX_CHARS:
        content = (
            content[:MEMORY_MAX_CHARS].rstrip()
            + "\n\n(Memory truncated for context.)\n"
        )
    return f'\n\n## Agent memory (from .remie/memory: "{memory["name"]}")\n{content}'


def get_full_system_prompt(native_tools: bool = False):
    """Build the system prompt.

    With ``native_tools`` (Codex provider) the text-protocol instructions are
    replaced by a note that tools are called natively through the API, and the
    textual tool list is dropped since schemas travel with the request.
    """
    if native_tools:
        protocol = (
            "You have access to the function tools provided with each request. "
            "Call them natively instead of describing tool usage in plain text; "
            "results arrive as tool outputs between your turns.\n"
        )
        tool_list_repr = ""
    else:
        tool_list_repr = ""
        for tool_name in TOOL_REGISTRY:
            tool_list_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
            tool_list_repr += f"\n{'=' * 15}\n"
        protocol = (
            "When you want to use a tool, first provide a short 'thinking:' line "
            "explaining your reasoning, then reply with exactly one line in the "
            "format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.\n"
            "Use compact single-line JSON with double quotes. After receiving a "
            "tool_result(...) message, continue the task.\n"
            "If no tool is needed, respond normally.\n"
        )
    return (
        _compose_system_prompt(tool_list_repr, protocol)
        + load_project_context()
        + load_agent_memory()
    )


_ASK_USER_PARAGRAPH = (
    "When multiple valid approaches have meaningful tradeoffs or require a user "
    "preference, do not choose silently. Briefly explain the options and ask the "
    "user which they prefer. Continue autonomously for routine implementation "
    "details or when one option clearly dominates. Do not ask unnecessary "
    "confirmation questions.\nTo ask the user a question, call the 'ask_user' "
    "tool and wait for its result instead of ending your turn."
)

_MEMORY_PARAGRAPH = (
    "Use the 'memory' tool to persist durable facts, decisions, user preferences, "
    "and open tasks that should be remembered across chats. Add a note when you "
    "learn something that will matter later; do not log routine progress and do "
    "not use memory as a chat transcript. Remie keeps an active project memory, "
    'so use memory(action="add", text=...) without a name to append to it. Use '
    'memory(action="list") to see older memories (each with an id and a name), '
    'and target one by name or id only when needed; memory(action="delete", '
    "name=...) removes a memory entirely."
)


def _compose_system_prompt(tool_list_repr: str, protocol: str) -> str:
    return (
        f"You are a coding assistant whose goal it is to help us solve coding tasks. \n"
        f"You have access to a series of tools you can execute. Here are the tools you can execute:\n\n"
        f"{tool_list_repr}\n"
        f"{protocol}\n"
        f"{_ASK_USER_PARAGRAPH}\n"
        f"{_MEMORY_PARAGRAPH}"
    )


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


async def _stream_local_sdk_call(
    payload: dict[str, Any],
    usage_box: dict[str, int] | None,
    reasoning_box: list[str] | None,
    finish_box: dict[str, Any] | None,
) -> AsyncIterator[str]:
    try:
        stream = await _get_local_openai_client().chat.completions.create(**payload)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                usage = getattr(chunk, "usage", None)
                if usage_box is not None and usage is not None:
                    usage_box["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                    usage_box["completion_tokens"] = (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_box is not None and finish_reason is not None:
                finish_box["finish_reason"] = finish_reason
                finish_box["truncated"] = finish_reason in TRUNCATED_REASONS
            delta = getattr(choice, "delta", None)
            if reasoning_box is not None and delta is not None:
                reason = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reason:
                    reasoning_box.append(reason)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content
        if finish_box is not None:
            finish_box["stream_complete"] = True
    except APIStatusError as error:
        raise LLMRequestError(error.status_code, str(error)) from error


async def stream_llm_call(
    conversation: list[dict[str, Any]],
    usage_box: dict[str, int] | None = None,
    reasoning_box: list[str] | None = None,
    finish_box: dict[str, Any] | None = None,
    tool_calls_box: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    if _config.provider == "codex":
        # Imported lazily to avoid a module-level import cycle. Codex uses the
        # OpenAI SDK with native function calling: schemas travel with the
        # request and the model's function_call items land in tool_calls_box.
        from remie.codex_client import stream_codex_call
        from remie.tools import get_tool_schemas

        async for content in stream_codex_call(
            conversation,
            model=_config.model,
            reasoning_effort=_config.reasoning_effort,
            tools=get_tool_schemas(),
            usage_box=usage_box,
            reasoning_box=reasoning_box,
            finish_box=finish_box,
            tool_calls_box=tool_calls_box,
        ):
            yield content
        return

    if _config.provider == "openrouter":
        # OpenRouter: plain httpx SSE against /chat/completions with native
        # function calling (no SDK). Tool calls land in tool_calls_box.
        from remie.openrouter_client import stream_openrouter_call
        from remie.tools import get_tool_schemas

        async for content in stream_openrouter_call(
            _config.api_key,
            conversation,
            model=_config.model,
            reasoning_effort=_config.reasoning_effort,
            tools=get_tool_schemas(),
            max_tokens=get_max_output_tokens(_config.provider),
            usage_box=usage_box,
            reasoning_box=reasoning_box,
            finish_box=finish_box,
            tool_calls_box=tool_calls_box,
        ):
            yield content
        return

    max_output_tokens = get_max_output_tokens(_config.provider)
    reasoning_supported = supports_reasoning_effort(_config.model, _config.provider)
    payload: dict[str, Any] = {
        "model": _config.model,
        "messages": conversation,
        "max_tokens": max_output_tokens,
        "stream": True,
    }
    if (
        _config.reasoning_effort != "off"
        and reasoning_supported
    ):
        payload["reasoning_effort"] = _config.reasoning_effort
    if usage_box is not None and _config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL:
        payload["stream_options"] = {"include_usage": True}

    # Keep the SDK local-only. The fake client used by unit tests also exercises
    # the protocol parser below; real local connections always take this path.
    if _config.provider == "local" and isinstance(_get_http_client(), httpx.AsyncClient):
        async for content in _stream_local_sdk_call(
            payload, usage_box, reasoning_box, finish_box
        ):
            yield content
        return

    url = f"{_config.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {_config.api_key}"}
    async with _get_http_client().stream(
        "POST", url, json=payload, headers=headers
    ) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", errors="replace").strip()
            raise LLMRequestError(
                response.status_code, body or f"HTTP {response.status_code}"
            )
        saw_done = False
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            if data == "[DONE]":
                saw_done = True
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                usage = chunk.get("usage")
                if usage_box is not None and usage is not None:
                    usage_box["prompt_tokens"] = usage.get("prompt_tokens") or 0
                    usage_box["completion_tokens"] = usage.get("completion_tokens") or 0
                continue
            choice = choices[0]
            if finish_box is not None:
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    finish_box["finish_reason"] = finish_reason
                    finish_box["truncated"] = finish_reason in TRUNCATED_REASONS
            delta = choice.get("delta") or {}
            if reasoning_box is not None:
                reason = delta.get("reasoning_content") or delta.get("reasoning")
                if reason:
                    reasoning_box.append(reason)
            content = delta.get("content")
            if content:
                yield content
        if finish_box is not None:
            finish_box["stream_complete"] = (
                saw_done or bool(finish_box.get("finish_reason"))
            )


async def summarize_messages(messages: list[dict[str, Any]]) -> str:
    """
    Ask the model to condense a list of conversation messages into a compact
    "session memory" note. Returns "" when the call fails or yields nothing.
    """
    if not messages:
        return ""
    summary_messages = [
        {
            "role": "system",
            "content": (
                "Condense the following conversation excerpt into a compact "
                "'session memory' note. Preserve key facts, decisions, file "
                "paths, user preferences, and open tasks. Omit routine detail. "
                "Return only the note, under 300 words, no markdown headers."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(messages, default=str),
        },
    ]
    try:
        chunks = [
            chunk
            async for chunk in stream_llm_call(summary_messages)
        ]
    except Exception:
        return ""
    return "".join(chunks).strip()


async def generate_chat_title(messages: list[dict[str, Any]]) -> str:
    """Ask the active model for a short title for a completed task."""
    if not messages:
        return ""
    title_messages = [
        {
            "role": "system",
            "content": (
                "Create a concise title for this coding task. Return only 3 to 8 "
                "words, with no quotes, punctuation, markdown, or explanation."
            ),
        },
        {"role": "user", "content": json.dumps(messages, default=str)},
    ]
    try:
        title = "".join(chunk async for chunk in stream_llm_call(title_messages))
    except Exception:
        return ""
    return " ".join(title.split()).strip(" `\"'.,:;!?\n")


def get_connection_error_message(error: Exception) -> str | None:
    """Return a user-facing message for an LLM timeout or connection error."""
    try:
        request = getattr(error, "request", None)
    except RuntimeError:
        request = None
    url = str(getattr(request, "url", "")) or _config.base_url
    if isinstance(error, httpx.TimeoutException):
        return (
            f"The LLM request to {url} timed out. "
            "Check that the model server is responding."
        )
    if isinstance(error, httpx.TransportError):
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


def render_user_message(text: str) -> Panel:
    """
    Render a user message as a bordered panel.
    """
    return Panel(Text(escape(text)), title="You", border_style="blue", padding=(0, 1))


def estimate_tokens_from_counts(chars: int, newlines: int) -> int:
    """
    Rough token estimate from pre-computed character and newline counts, used
    when the API does not report exact usage. Based on the ~4 chars/token
    heuristic, with newlines counted separately because streams can cheaply
    track these counters without re-scanning the accumulated text.
    """
    if chars == 0:
        return 0
    return max(1, chars // 4 + newlines // 3)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate for a piece of text, used when the API does not
    report exact usage. Based on the ~4 chars/token heuristic.
    """
    if not text:
        return 0
    return estimate_tokens_from_counts(len(text), text.count("\n"))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """
    Rough token estimate for a single conversation message, summing string
    content (including tool_result(...) messages and multimodal parts).
    """
    content = message.get("content")
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                total += estimate_tokens(part["text"])
            elif hasattr(part, "text") and part.text:
                total += estimate_tokens(part.text)
        return total
    return 0


def estimate_conversation_tokens(
    conversation: list[dict[str, Any]],
) -> int:
    """
    Rough token estimate for the whole conversation, summing string content
    (including tool_result(...) messages).
    """
    return sum(estimate_message_tokens(message) for message in conversation)


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
