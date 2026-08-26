"""Remie's Textual TUI with lazy compatibility exports.

Importing :mod:`remie.tui` no longer initializes Textual, Pillow, every modal,
and the provider stack. Public names from the former single-module TUI are
resolved and cached only when requested.
"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Entry points and application.
    "AgentApp": ("remie.tui.app", "AgentApp"),
    "AgentScreen": ("remie.tui.app", "AgentScreen"),
    "main": ("remie.tui.app", "main"),
    "run_tui": ("remie.tui.app", "run_tui"),
    # Compatibility modules and image objects.
    "codex_auth": ("remie.codex_auth", ""),
    "PILImage": ("PIL.Image", ""),
    "ImageGrab": ("PIL.ImageGrab", ""),
    # Agent compatibility exports.
    **{
        name: ("remie.agent", name)
        for name in (
            "CODEX_BACKEND_BASE",
            "CODEX_MODELS",
            "estimate_conversation_tokens",
            "estimate_message_tokens",
            "estimate_tokens",
            "fetch_codex_models",
            "fetch_openrouter_models",
            "fetch_opencode_go_models",
            "generate_chat_title",
            "render_assistant_panel",
            "run_tool",
            "save_provider_configs",
            "stream_llm_call",
        )
    },
    # Constants.
    **{
        name: ("remie.tui.constants", name)
        for name in (
            "COMPACTION_CONTEXT_RATIO",
            "COMPACTION_KEEP_MESSAGES",
            "LIVE_REASONING_TICK",
            "MAX_AUTO_CONTINUATIONS",
            "MAX_EMPTY_RESPONSE_RETRIES",
            "PROMPT_HISTORY_LIMIT",
            "PROVIDER_BASE_URLS",
            "REASONING_EFFORTS",
            "STATUS_ANIMATION_MAX_FPS",
            "STREAM_PREVIEW_MAX_CHARS",
            "STREAM_RENDER_COALESCE_WINDOW",
            "STREAM_UPDATE_CHARS_PER_SECOND",
            "STREAM_UPDATE_LARGE_PREVIEW_CHARS",
            "STREAM_UPDATE_MIN_INTERVAL",
            "STREAM_UPDATE_MIN_INTERVAL_LARGE",
        )
    },
    "CSS": ("remie.tui.css", "CSS"),
    # Helpers and rendering.
    **{
        name: ("remie.tui.helpers", name)
        for name in (
            "_coerce_model_info",
            "_detect_terminal_background",
            "_fallback_memory_name",
            "_format_tokens",
            "_has_tool_call",
            "_is_tmux",
            "_model_option",
            "_preview_window",
            "_safe_reasoning_markdown",
            "_safe_stream_markdown",
            "_should_update_stream",
            "_stream_update_interval",
        )
    },
    **{
        name: ("remie.tui.render", name)
        for name in (
            "TOOL_RESULT_MAX_CHARS",
            "_PlainWrite",
            "_command_body",
            "_command_output_lexer",
            "_format_tool_result",
            "_guess_lexer_name",
            "_make_syntax",
            "_plain_tool_panel",
            "_render_diff",
            "_render_read_file_result",
            "_render_run_command_result",
            "_render_tool_result",
            "_truncate_body",
        )
    },
    # Widgets and screens.
    **{
        name: ("remie.tui.widgets", name)
        for name in (
            "InputRow",
            "ModelBadge",
            "ModelRow",
            "PromptBox",
            "PromptSubmitted",
            "PromptTextArea",
            "StatusIndicator",
            "StreamingRichLog",
            "ThinkingIndicator",
            "_load_status_gif",
        )
    },
    "AskUserScreen": ("remie.tui.screens", "AskUserScreen"),
    "ChatScreen": ("remie.tui.screens", "ChatScreen"),
    "ConnectionScreen": ("remie.tui.screens", "ConnectionScreen"),
    "MemoryScreen": ("remie.tui.screens", "MemoryScreen"),
    "ModelScreen": ("remie.tui.screens", "ModelScreen"),
    "SlashCommandPopup": ("remie.tui.widgets", "SlashCommandPopup"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load and cache a historical package export on first access."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name)
    value = getattr(module, attribute) if attribute else module
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
