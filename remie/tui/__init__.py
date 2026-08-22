"""Remie's Textual TUI.

This package is a refactor of the former single-module ``remie/tui.py``.
Every public name is re-exported here so existing imports
(``from remie.tui import ...``) keep working unchanged.
"""

from PIL import Image as PILImage
from PIL import ImageGrab  # noqa: F401 -- re-exported; tests patch tui.ImageGrab

from remie import codex_auth  # noqa: F401 -- re-exported for parity with the
# former single-module remie/tui.py, whose globals included these names.
from remie.agent import (
    CODEX_BACKEND_BASE,
    CODEX_MODELS,
    estimate_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens,
    fetch_codex_models,
    fetch_openrouter_models,
    fetch_opencode_go_models,
    generate_chat_title,
    render_assistant_panel,
    run_tool,
    save_provider_configs,
    stream_llm_call,
)
from remie.tui import _agent_app_registry as _registry  # noqa: F401
from remie.tui.app import AgentApp, AgentScreen, main, run_tui

# Make AgentApp resolvable as a bare name inside widgets and screens
# without a circular import at module load time.
_registry.register(AgentApp)
from remie.tui.constants import (
    COMPACTION_CONTEXT_RATIO,
    COMPACTION_KEEP_MESSAGES,
    MAX_AUTO_CONTINUATIONS,
    MAX_EMPTY_RESPONSE_RETRIES,
    PROMPT_HISTORY_LIMIT,
    PROVIDER_BASE_URLS,
    REASONING_EFFORTS,
    STREAM_PREVIEW_MAX_CHARS,
    STREAM_UPDATE_CHARS_PER_SECOND,
    STREAM_UPDATE_MIN_INTERVAL,
)
from remie.tui.css import CSS
from remie.tui.helpers import (
    _coerce_model_info,
    _detect_terminal_background,
    _fallback_memory_name,
    _format_tokens,
    _has_tool_call,
    _is_tmux,
    _model_option,
    _preview_window,
    _safe_reasoning_markdown,
    _safe_stream_markdown,
    _should_update_stream,
    _stream_update_interval,
)
from remie.tui.render import (
    TOOL_RESULT_MAX_CHARS,
    _PlainWrite,
    _render_diff,
    _render_tool_result,
    _command_body,
    _command_output_lexer,
    _format_tool_result,
    _guess_lexer_name,
    _make_syntax,
    _plain_tool_panel,
    _render_diff,
    _render_read_file_result,
    _render_run_command_result,
    _render_tool_result,
    _truncate_body,
)
from remie.tui.widgets import (
    InputRow,
    ModelBadge,
    ModelRow,
    PromptBox,
    PromptSubmitted,
    PromptTextArea,
    StatusIndicator,
    StreamingRichLog,
    ThinkingIndicator,
    _load_status_gif,
)
from remie.tui.screens import (
    AskUserScreen,
    ChatScreen,
    ConnectionScreen,
    MemoryScreen,
)

__all__ = [
    "AgentApp",
    "AgentScreen",
    "AskUserScreen",
    "ChatScreen",
    "ConnectionScreen",
    "InputRow",
    "MAX_AUTO_CONTINUATIONS",
    "MemoryScreen",
    "ModelBadge",
    "ModelRow",
    "PromptBox",
    "PromptSubmitted",
    "PromptTextArea",
    "StatusIndicator",
    "StreamingRichLog",
    "ThinkingIndicator",
    "_format_tokens",
    "_has_tool_call",
    "_is_tmux",
    "_load_status_gif",
    "main",
    "run_tui",
]


def __getattr__(name: str):
    """Fallback attribute access while the package is partially initialized."""
    if name == "AgentApp":
        from remie.tui.app import AgentApp

        return AgentApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
