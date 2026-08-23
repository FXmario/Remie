"""Backward-compatible facade over Remie's focused core modules."""

# Imports in this facade intentionally re-export the historical public API.
# dotenv must load before config.py evaluates environment-backed defaults.
# ruff: noqa: E402, F401

import json
from importlib import import_module
from collections.abc import AsyncIterator
from typing import Any

import httpx
from dotenv import load_dotenv
from remie.model_names import ModelInfo

load_dotenv()

from remie.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    ConfigStore,
    ConnectionConfig,
    default_config as _config_default,
    provider_defaults as _config_provider_defaults,
)

from remie.providers.catalog import (
    fetch_codex_models as _fetch_codex_models,
    fetch_opencode_go_models as _fetch_opencode_go_models,
    fetch_openrouter_models as _fetch_openrouter_models,
    get_max_output_tokens,
    get_model_context_limit as _get_model_context_limit,
    get_model_info as _get_model_info,
    supports_reasoning_effort,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **{
        name: ("remie.config", name)
        for name in (
            "CODEX_BACKEND_BASE",
            "CODEX_MODELS",
            "LOCAL_BASE_URL",
            "OPENCODE_GO_BASE_URL",
            "OPENCODE_GO_MODELS",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_MODELS",
            "STATUS_ANIMATION_CONFIG_KEY",
            "SUPPORTED_PROVIDERS",
        )
    },
    "LLMRequestError": ("remie.errors", "LLMRequestError"),
    "UnsupportedModelError": ("remie.errors", "UnsupportedModelError"),
    **{
        name: ("remie.providers.catalog", name)
        for name in (
            "CODEX_DEFAULT_CONTEXT_LIMIT",
            "NON_REASONING_EFFORT_MODELS",
            "OPENCODE_GO_DEFAULT_CONTEXT_LIMIT",
            "OPENROUTER_DEFAULT_CONTEXT_LIMIT",
        )
    },
    **{
        name: ("remie.prompts", name)
        for name in (
            "MEMORY_MAX_CHARS",
            "PROJECT_CONTEXT_MAX_CHARS",
            "SYSTEM_PROMPT",
            "build_system_prompt",
            "get_full_system_prompt",
            "load_agent_memory",
            "load_project_context",
        )
    },
    **{
        name: ("remie.protocol", name)
        for name in (
            "extract_dsml_invocations",
            "extract_thinking",
            "extract_tool_invocations",
            "strip_protocol_lines",
        )
    },
    **{
        name: ("remie.tokens", name)
        for name in (
            "estimate_conversation_tokens",
            "estimate_message_tokens",
            "estimate_tokens",
            "estimate_tokens_from_counts",
        )
    },
    "execute_tool_call": ("remie.tools.executor", "execute_tool_call"),
    "run_tool": ("remie.tools.executor", "run_tool"),
    "render_assistant_message": ("remie.rendering", "render_assistant_message"),
    "render_assistant_panel": ("remie.rendering", "render_assistant_panel"),
    "render_user_message": ("remie.rendering", "render_user_message"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


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

# Live context windows for OpenCode Go models, populated from the models API
# (each model reports its own context_length). Used for context compaction so
# the budget tracks the actual model instead of a hardcoded table.
_opencode_go_model_context: dict[str, int] = {}


def get_model_info(model_id: str) -> ModelInfo:
    return _get_model_info(model_id, _model_info_cache)


async def fetch_opencode_go_models(api_key: str) -> list[ModelInfo]:
    return await _fetch_opencode_go_models(
        api_key, _opencode_go_model_context, _model_info_cache
    )


async def fetch_codex_models() -> list[ModelInfo]:
    return await _fetch_codex_models(_codex_model_context, _model_info_cache)


async def fetch_openrouter_models() -> list[ModelInfo]:
    return await _fetch_openrouter_models(_openrouter_model_context, _model_info_cache)


def get_model_context_limit(model: str, provider: str = "local") -> int | None:
    return _get_model_context_limit(
        model,
        provider,
        _opencode_go_model_context,
        _openrouter_model_context,
        _codex_model_context,
    )


def _config_store() -> ConfigStore:
    return ConfigStore(CONFIG_DIR, CONFIG_FILE)


def _provider_defaults(provider: str) -> ConnectionConfig:
    return _config_provider_defaults(provider)


def _default_config() -> ConnectionConfig:
    return _config_default()


def load_config() -> ConnectionConfig:
    return _config_store().load()


def save_config(config: ConnectionConfig) -> None:
    _config_store().save(config)


def load_provider_configs() -> dict[str, ConnectionConfig]:
    return _config_store().load_profiles()


def save_provider_configs(
    profiles: dict[str, ConnectionConfig], active_provider: str
) -> None:
    _config_store().save_profiles(profiles, active_provider)


def load_status_animation_enabled() -> bool:
    return _config_store().load_status_animation_enabled()


def save_status_animation_enabled(enabled: bool) -> None:
    _config_store().save_status_animation_enabled(enabled)


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
_local_openai_client: Any | None = None
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


def set_active_connection(
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "local",
    reasoning_effort: str = "medium",
    verify_ssl: bool = False,
) -> ConnectionConfig:
    """Update the active model-provider connection."""
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


# Historical name retained for external callers and saved test integrations.
configure_openai = set_active_connection


def _get_local_openai_client() -> Any:
    """Return an OpenAI SDK client pointed at the local compatible server."""
    global _local_openai_client, _local_openai_client_key
    from openai import AsyncOpenAI

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


async def stream_llm_call(
    conversation: list[dict[str, Any]],
    usage_box: dict[str, int] | None = None,
    reasoning_box: list[str] | None = None,
    finish_box: dict[str, Any] | None = None,
    tool_calls_box: list[dict[str, str]] | None = None,
    reasoning_items_box: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Compatibility facade over the typed provider event stream."""
    from remie.providers.events import (
        FinishEvent,
        ReasoningDelta,
        TextDelta,
        ToolCallEvent,
        UsageEvent,
    )
    from remie.providers.router import RoutedProvider

    provider = RoutedProvider(
        _config,
        get_http_client=_get_http_client,
        get_local_openai_client=_get_local_openai_client,
        max_output_tokens=get_max_output_tokens(_config.provider),
        reasoning_supported=supports_reasoning_effort(_config.model, _config.provider),
    )
    async for event in provider.stream(conversation):
        if isinstance(event, TextDelta):
            yield event.text
        elif isinstance(event, ReasoningDelta) and reasoning_box is not None:
            reasoning_box.append(event.text)
        elif isinstance(event, UsageEvent) and usage_box is not None:
            usage_box["prompt_tokens"] = event.input_tokens
            usage_box["completion_tokens"] = event.output_tokens
        elif isinstance(event, ToolCallEvent) and tool_calls_box is not None:
            tool_calls_box.append(
                {
                    "id": event.id,
                    "name": event.name,
                    "arguments": event.arguments,
                }
            )
        elif isinstance(event, FinishEvent):
            if finish_box is not None:
                finish_box["finish_reason"] = event.reason
                finish_box["truncated"] = event.truncated
                finish_box["stream_complete"] = event.complete
            if reasoning_items_box is not None:
                reasoning_items_box.extend(
                    event.provider_metadata.get("reasoning_items", [])
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
        chunks = [chunk async for chunk in stream_llm_call(summary_messages)]
    except Exception:
        return ""
    return "".join(chunks).strip()


async def generate_chat_title(messages: list[dict[str, Any]]) -> str:
    """Ask the active model for a concise title for a completed conversation."""
    if not messages:
        return ""
    title_messages = [
        {
            "role": "system",
            "content": (
                "Create a concise title describing the conversation's main theme "
                "or the work actually completed. The input may include a current "
                "chat title: return that exact title when the central topic has not "
                "meaningfully changed; update it when the conversation has shifted "
                "to a different main topic. Return only 3 to 7 words, with no "
                "quotes, terminal punctuation, markdown, or explanation."
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
