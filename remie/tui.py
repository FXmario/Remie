import asyncio
import base64
import io
import json
import os
from pathlib import Path
import re
import select
import sys
import time
from typing import Any, ClassVar

import httpx
from PIL import Image as PILImage
from PIL import ImageGrab
from PIL import ImageSequence
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.markup import escape
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text
from textual import work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.keys import format_key
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Size
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.strip import Strip
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Switch,
    TextArea,
)
from textual_image.widget import SixelImage as TerminalImage

from remie.agent import (
    ConnectionConfig,
    LLMRequestError,
    OPENAI_BASE_URL,
    OPENAI_MODELS,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
    UnsupportedModelError,
    clear_session,
    configure_openai,
    estimate_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens,
    estimate_tokens_from_counts,
    extract_thinking,
    extract_tool_invocations,
    fetch_openai_models,
    fetch_codex_models,
    fetch_opencode_go_models,
    get_config,
    get_connection_error_message,
    get_full_system_prompt,
    get_model_context_limit,
    load_provider_configs,
    render_assistant_panel,
    render_user_message,
    run_tool,
    save_provider_configs,
    save_session,
    stream_llm_call,
    strip_protocol_lines,
    summarize_messages,
    supports_reasoning_effort,
)

from remie.tools import (
    create_launch_memory,
    delete_memory,
    ensure_general_memory,
    find_memory_by_id,
    get_active_memory_id,
    get_tool_summary,
    list_memories,
    set_active_memory_id,
)

REASONING_EFFORTS = ("off", "low", "medium", "high", "max")
PROMPT_HISTORY_LIMIT = 100
MAX_AUTO_CONTINUATIONS = int(
    os.environ.get("REMIE_MAX_AUTO_CONTINUATIONS", "10")
)
MAX_EMPTY_RESPONSE_RETRIES = int(
    os.environ.get("REMIE_MAX_EMPTY_RESPONSE_RETRIES", "2")
)
COMPACTION_CONTEXT_RATIO = 0.8
COMPACTION_KEEP_MESSAGES = 10

# The streaming preview re-renders the accumulated Markdown, which is
# expensive (parse + Pygments + layout). Two mitigations keep the UI
# responsive: throttling with an interval that grows with the text size, and
# rendering only a bounded tail window of the text so the per-update cost stays
# roughly constant no matter how long the generated answer gets.
#
# Because the per-update render cost is bounded by the preview window, the
# throttle interval is clamped to that same window: it never grows beyond the
# minimum once the text is long enough that the preview is capped, so long
# answers keep streaming at a steady rate instead of slowing down over time.
STREAM_UPDATE_MIN_INTERVAL = 0.1
STREAM_UPDATE_CHARS_PER_SECOND = 50_000
STREAM_PREVIEW_MAX_CHARS = 3000
PROVIDER_BASE_URLS = {
    "openai": OPENAI_BASE_URL,
    "opencode-go": OPENCODE_GO_BASE_URL,
    "codex": "",
}

CSS = """
Screen {
    layout: vertical;
}

#log {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
    border: round $primary;
    margin: 0 1;
}

#prompt {
    height: 3;
    width: 1fr;
    margin: 0;
    border: round $primary;
    border-title-align: right;
}

#prompt .text-area--placeholder {
    color: grey;
}

#prompt-box {
    height: 4;
    width: 1fr;
}

#input-row {
    height: 5;
    width: 100%;
    padding: 0 1 1 1;
    align: left middle;
}

#status {
    width: 8;
    height: 4;
    margin-right: 0;
    content-align: center middle;
    background: $panel;
}

#status-gif {
    width: 8;
    height: 4;
}

#tmux-spinner {
    width: 1;
    height: 1;
    margin-right: 0;
    content-align: center middle;
    display: none;
}

#model-row {
    dock: top;
    align: right top;
    height: 1;
    width: 100%;
    padding: 0 1;
}

#model-badge {
    height: 1;
    width: auto;
    padding: 0 1;
    margin-right: 1;
    content-align: center middle;
    border: none;
    color: $text;
    background: $panel;
}

#model-badge:hover {
    background: $primary 20%;
}
"""


def _detect_terminal_background() -> str | None:
    """
    Query the terminal background color via OSC 11 and return 'light' or 'dark'.

    Returns None when the terminal does not respond or is not interactive.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    sys.stdout.write("\x1b]11;?\x1b\\")
    sys.stdout.flush()
    response = b""
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 0.02)
        if ready:
            try:
                data = os.read(sys.stdin.fileno(), 1024)
            except (OSError, ValueError):
                break
            if not data:
                break
            response += data
            if b"\x1b\\" in response:
                break
    match = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", response)
    if not match:
        return None
    channels = []
    for component in match.groups():
        value = int(component[:2], 16)
        channels.append(value / 255.0)
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "light" if luminance >= 0.5 else "dark"


def _is_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def _safe_stream_markdown(
    text: str, code_theme: str, style: str = ""
) -> RenderableType:
    """Render partial Markdown safely during streaming.

    Auto-closes incomplete code fences so Pygments highlighting engages
    as soon as the opening fence and language arrive.  Falls back to
    plain escaped text if the Markdown parser rejects the content.

    When the text contains no code fence, the full Markdown + Pygments path is
    skipped entirely (it is the dominant cost of streaming re-renders) and the
    text is shown as plain styled text instead.
    """
    fence_count = text.count("\n```") + (1 if text.startswith("```") else 0)
    if fence_count == 0:
        return Text.from_markup(escape(text), style=style or None)
    if fence_count % 2 == 1:
        text = text + "\n```"
    try:
        return Markdown(
            text, code_theme=code_theme, hyperlinks=True, style=style or "none"
        )
    except Exception:
        return Text.from_markup(escape(text), style=style or None)


def _safe_reasoning_markdown(text: str, code_theme: str) -> RenderableType:
    """Render reasoning content gray so it reads as secondary text."""
    return _safe_stream_markdown(text, code_theme, style="grey62")


def _has_tool_call(text: str) -> bool:
    """Return True if any complete line is a tool invocation (or DSML markup)."""
    if "<|DSML|>" in text and "invoke name=" in text:
        return True
    return any(
        line.strip().startswith(("tool:", "<tool:"))
        for line in text.splitlines()
        if line.strip()
    )


def _stream_update_interval(text_len: int) -> float:
    """Minimum seconds between streaming preview re-renders for a given size.

    The length is clamped to the preview window because that bounds the actual
    render cost: `_preview_window` never renders more than
    `STREAM_PREVIEW_MAX_CHARS`, so the throttle interval must not keep growing
    with the (unbounded) accumulated text.
    """
    return max(
        STREAM_UPDATE_MIN_INTERVAL,
        min(text_len, STREAM_PREVIEW_MAX_CHARS)
        / STREAM_UPDATE_CHARS_PER_SECOND,
    )


def _should_update_stream(
    accumulated_len: int, last_update: float, now: float
) -> bool:
    """Whether the streaming preview should re-render now (throttled)."""
    return now - last_update >= _stream_update_interval(accumulated_len)


def _preview_window(text: str, limit: int = STREAM_PREVIEW_MAX_CHARS) -> str:
    """Return a bounded tail window of text for the live preview.

    Keeps per-update rendering cost roughly constant regardless of how long
    the generated answer gets. The cut is moved forward to the next newline so
    the preview never starts mid-line; empty and short inputs are unchanged.
    """
    if len(text) <= limit:
        return text
    start = text.rfind("\n", 0, len(text) - limit)
    if start == -1:
        start = len(text) - limit
    else:
        start += 1
    return text[start:]


def _format_tokens(count: int) -> str:
    """Format a token count compactly, e.g. 1234 -> '1.2k'."""
    if count >= 1000:
        value = count / 1000.0
        if value == int(value):
            return f"{int(value)}k"
        return f"{value:.1f}k"
    return str(count)


def _model_option(model: str) -> tuple[str, str]:
    return model, model


class _PlainWrite:
    """Wrap a rich renderable so RichLog can still extract plain text for
    selection and tests, while the actual rendering stays highlighted."""

    def __init__(self, plain: str, renderable: RenderableType) -> None:
        self.plain = plain
        self._renderable = renderable

    def __rich_console__(self, console, options) -> Any:
        yield from console.render(self._renderable, options)


def _render_diff(diff: str, code_theme: str = "ansi_dark") -> Panel:
    """Render a unified diff as a highlightable panel."""
    return Panel(
        _make_syntax(diff, "diff", code_theme),
        title="Diff",
        border_style="cyan",
        padding=(0, 1),
    )


def _make_syntax(
    code: str, language: str, code_theme: str
) -> RenderableType:
    """Highlight code with Pygments; fall back to escaped text on any error."""
    try:
        syntax = Syntax(code, language, theme=code_theme, word_wrap=False)
        # Rich does not raise for unknown lexer names; it leaves lexer None.
        if syntax.lexer is None:
            raise ValueError(f"Unknown lexer: {language}")
        return syntax
    except Exception:
        return Text.from_markup(escape(code))


def _guess_lexer_name(filename: str) -> str | None:
    """Guess a Pygments lexer name from a filename, or None when unknown."""
    try:
        from pygments.lexers import get_lexer_for_filename
        from pygments.util import ClassNotFound

        return get_lexer_for_filename(filename, "").name
    except (ClassNotFound, TypeError, ValueError, OSError):
        return None


def _command_body(result: dict[str, Any]) -> str:
    """Raw stdout/stderr joined the same way the summary shows them."""
    output = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    return "\n".join(part for part in (output, stderr) if part)


def _command_output_lexer(output: str) -> str | None:
    """Detect a lexer for shell command output: json, unified diff, or a
    Python traceback; None (plain text) otherwise."""
    sample = output.strip()
    if len(sample) > 500:
        sample = sample[:500]
    if not sample:
        return None
    stripped = sample.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return "json"
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if sample.startswith(("--- ", "+++ ", "@@ ")) or "\n--- " in sample:
        return "diff"
    lines = sample.splitlines()[:3]
    if len(lines) >= 2 and lines[0].startswith("Traceback") and 'File "' in sample:
        return "pytb"
    return None


TOOL_RESULT_MAX_CHARS = 2000


def _format_tool_result(name: str, result: dict[str, Any]) -> str:
    """Return a readable text summary of a tool result."""
    if result.get("blocked"):
        return ""
    if "error" in result:
        return f"Error: {result['error']}"
    if name == "read_file":
        content = result.get("content", "")
        return f"Read {result.get('file_path')} ({len(content)} chars)"
    if name == "edit_file":
        return f"{result.get('action', 'edited')}: {result.get('path')}"
    if name == "run_command":
        parts = [f"exit {result.get('exit_code')}"]
        if result.get("timed_out"):
            parts.append("timed out")
        output = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        body = "\n".join(part for part in (output, stderr) if part)
        summary = " · ".join(parts)
        return f"{summary}\n{body}" if body else summary
    if name == "list_files":
        entries = result.get("files", [])
        names = ", ".join(e.get("filename", "") for e in entries[:50])
        suffix = f" (+{len(entries) - 50} more)" if len(entries) > 50 else ""
        return f"{len(entries)} entries: {names}{suffix}"
    if name == "glob_files":
        matches = result.get("matches", [])
        names = ", ".join(matches[:50])
        suffix = f" (+{len(matches) - 50} more)" if len(matches) > 50 else ""
        return f"{result.get('count', len(matches))} matches: {names}{suffix}"
    if name == "tree_files":
        return result.get("tree", "")
    if name == "ask_user":
        answer = result.get("answer")
        return f"Answer: {answer}" if answer is not None else "Cancelled"
    return json.dumps(result, default=str)


def _truncate_body(
    text: str, limit: int = TOOL_RESULT_MAX_CHARS
) -> tuple[str, bool]:
    """Truncate a body to `limit` chars; return (text, was_truncated)."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n… (result truncated)", True


def _plain_tool_panel(name: str, text: str) -> Panel:
    """Render a plain (unhighlighted) tool result panel."""
    return Panel(
        Text.from_markup(escape(text)),
        title=f"Tool result · {name}",
        border_style="blue",
        padding=(0, 1),
    )


def _render_read_file_result(
    result: dict[str, Any], code_theme: str
) -> Panel:
    """Render a read_file result: a summary line plus the file content,
    syntax-highlighted by extension when a lexer can be guessed."""
    content = result.get("content", "")
    path = str(result.get("file_path", ""))
    body_text, truncated = _truncate_body(content)
    summary = f"Read {path} ({len(content)} chars)"
    if truncated:
        summary += " \u00b7 (result truncated)"
    lexer = _guess_lexer_name(path)
    if lexer is not None and body_text.strip():
        body: RenderableType = _PlainWrite(
            body_text, _make_syntax(body_text, lexer, code_theme)
        )
    else:
        body = Text.from_markup(escape(body_text))
    return Panel(
        Group(Text(summary, style="bold"), Text(), body),
        title="Tool result · read_file",
        border_style="blue",
        padding=(0, 1),
    )


def _render_run_command_result(
    result: dict[str, Any], code_theme: str
) -> RenderableType:
    """Render a run_command result: summary plus output, highlighted when the
    output looks like JSON, a unified diff, or a Python traceback."""
    output = _command_body(result)
    lexer = _command_output_lexer(output) if output else None
    parts = [f"exit {result.get('exit_code')}"]
    if result.get("timed_out"):
        parts.append("timed out")
    summary = " · ".join(parts)
    if lexer is not None:
        body_text, truncated = _truncate_body(output)
        if truncated:
            summary += " \u00b7 (result truncated)"
        body = _PlainWrite(
            body_text, _make_syntax(body_text, lexer, code_theme)
        )
        return Panel(
            Group(Text(summary, style="bold"), Text(), body),
            title="Tool result · run_command",
            border_style="blue",
            padding=(0, 1),
        )
    text = f"{summary}\n{output}" if output else summary
    body_text, truncated = _truncate_body(text)
    if truncated:
        text = body_text
    return _plain_tool_panel("run_command", text)


def _render_tool_result(
    name: str, result: dict[str, Any], code_theme: str = "ansi_dark"
) -> RenderableType | None:
    """Render a readable, truncated panel for a tool result, or None."""
    if result.get("blocked"):
        return None
    if "error" in result:
        return _plain_tool_panel(name, f"Error: {result['error']}")
    if name == "read_file":
        return _render_read_file_result(result, code_theme)
    if name == "run_command":
        return _render_run_command_result(result, code_theme)
    text = _format_tool_result(name, result)
    if not text:
        return None
    body_text, _ = _truncate_body(text)
    return _plain_tool_panel(name, body_text)


class StreamingRichLog(RichLog):
    """A RichLog that can stream text in place at the bottom of the log."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_start: int | None = None

    def get_selection(self, selection) -> tuple[str, str] | None:
        """Extract selected text from the rendered log lines."""
        text = "\n".join(strip.text for strip in self.lines)
        return selection.extract(text), "\n"

    def begin_stream(self) -> None:
        self._stream_start = len(self.lines)

    def update_stream(
        self,
        content: str | RenderableType,
        *,
        title: str = "Assistant",
        border_style: str = "yellow",
    ) -> None:
        if self._stream_start is None:
            self.begin_stream()
        inner = (
            content
            if not isinstance(content, str)
            else Text.from_markup(escape(content))
        )
        renderable = Panel(
            inner,
            title=title,
            border_style=border_style,
            padding=(0, 1),
        )
        console = self.app.console
        width = max(self.scrollable_content_region.width, 1)
        segments = console.render(renderable, console.options.update_width(width))
        lines = list(Segment.split_lines(segments))
        strips = Strip.from_lines(lines)
        for strip in strips:
            strip.adjust_cell_length(width)
        del self.lines[self._stream_start :]
        self.lines.extend(strips)
        self._widest_line_width = max(
            self._widest_line_width,
            max(
                (sum(segment.cell_length for segment in strip) for strip in strips),
                default=0,
            ),
        )
        self._line_cache.clear()
        self.virtual_size = Size(self._widest_line_width, len(self.lines))
        self.scroll_end(animate=False, immediate=False, x_axis=False)
        self.refresh()

    def end_stream(self) -> None:
        self._stream_start = None

    def replace_stream(self, *renderables: object) -> None:
        if self._stream_start is not None:
            del self.lines[self._stream_start :]
            self._stream_start = None
        for renderable in renderables:
            self.write(renderable)
        self._line_cache.clear()
        self.refresh()


def _load_status_gif(name: str) -> tuple[list[PILImage.Image], list[float]]:
    """Load GIF frames and their original durations from the project assets."""
    asset = Path(__file__).resolve().parent.parent / "assets" / name
    if not asset.exists():
        asset = Path.cwd() / "assets" / name
    with PILImage.open(asset) as image:
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGBA").copy())
            durations.append(max(0.1, frame.info.get("duration", 100) / 1000))
    return frames, durations


class StatusIndicator(Vertical):
    """Animated Sixel/unicode status indicator with a text fallback."""

    STATUSES = ("ready", "working", "done")

    def __init__(self) -> None:
        super().__init__(id="status")
        self._frames: dict[str, tuple[list[PILImage.Image], list[float]]] = {}
        self._state = "ready"
        self._frame_index = 0
        self._timer = None

    def _ensure_loaded(self, status: str) -> tuple[list[PILImage.Image], list[float]]:
        """Return the frames/durations for a status, loading them on first use.

        GIFs are loaded lazily per state so startup only decodes the initial
        "ready" frames instead of all three animations up front.
        """
        if status not in self._frames:
            self._frames[status] = _load_status_gif(f"{status}.gif")
        return self._frames[status]

    def compose(self) -> ComposeResult:
        yield TerminalImage(
            self._ensure_loaded(self._state)[0][0], id="status-gif"
        )

    def on_mount(self) -> None:
        if not _is_tmux():
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        durations = self._ensure_loaded(self._state)[1]
        self._timer = self.set_timer(durations[self._frame_index], self._advance)

    def _advance(self) -> None:
        frames = self._ensure_loaded(self._state)[0]
        self._frame_index = (self._frame_index + 1) % len(frames)
        self.query_one("#status-gif", TerminalImage).image = frames[self._frame_index]
        if not _is_tmux():
            self._schedule_next_frame()

    def set_status(self, status: str) -> None:
        if status not in self.STATUSES:
            raise ValueError(f"Unknown status: {status}")
        if self._timer is not None:
            self._timer.stop()
        self._state = status
        self._frame_index = 0
        self.query_one("#status-gif", TerminalImage).image = (
            self._ensure_loaded(status)[0][0]
        )
        if not _is_tmux():
            self._schedule_next_frame()


class ThinkingIndicator(Label):
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        super().__init__(id="tmux-spinner")
        self._frame_index = 0
        self._working = False

    def on_mount(self) -> None:
        if _is_tmux():
            self.set_interval(0.1, self._advance)

    def _advance(self) -> None:
        if self._working:
            self._frame_index = (self._frame_index + 1) % len(self.FRAMES)
            self.update(f"{self.FRAMES[self._frame_index]}")

    def set_status(self, status: str) -> None:
        self._working = status == "working"
        self.display = self._working and _is_tmux()
        if self._working:
            self._frame_index = 0
            self.update(self.FRAMES[0])


class ModelBadge(Label):
    """Clickable label showing the active model name, vendor, and token usage."""

    def __init__(self) -> None:
        super().__init__(id="model-badge")
        self._model_text = ""
        self._vendor_text = ""
        self._reasoning_effort = "off"
        self._input_tokens = 0
        self._output_tokens = 0
        self._speed: float | None = None
        self.update_config(get_config())

    def update_config(self, config) -> None:
        provider = config.provider
        # Preserve compatibility with callers that construct a config from a
        # remote base URL without setting the provider field.
        if provider == "local":
            provider = (
                "opencode-go"
                if config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL
                else "openai"
                if config.base_url.rstrip("/") == OPENAI_BASE_URL
                else "local"
            )
        vendor = {
            "codex": "Codex CLI",
            "openai": "OpenAI",
            "opencode-go": "OpenCode Go",
        }.get(provider, "Local")
        self._model_text = config.model
        self._vendor_text = vendor
        self._reasoning_effort = config.reasoning_effort
        self._show(self._input_tokens, self._output_tokens)

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._show(input_tokens, output_tokens)

    def set_speed(self, speed: float | None) -> None:
        self._speed = speed
        self._show(self._input_tokens, self._output_tokens)

    def _show(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        text = (
            f"{self._model_text}  {self._vendor_text}"
            if self._model_text
            else self._vendor_text
        )
        if self._reasoning_effort != "off":
            text += f" · effort {self._reasoning_effort}"
        if input_tokens or output_tokens:
            total = input_tokens + output_tokens
            text += f" · {_format_tokens(total)} tok"
        if self._speed is not None:
            text += f" · {self._speed:.1f} tok/s"
        self.update(text)

    async def on_click(self) -> None:
        app = self.app
        if isinstance(app, AgentApp):
            await app.action_open_connection()


class PromptSubmitted(Message):
    """Posted when the user submits the prompt box."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PromptTextArea(TextArea):
    """Multiline prompt: Enter submits, Shift+Enter / Ctrl+J insert newlines."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            id="prompt",
            placeholder="Type a prompt here...",
            soft_wrap=True,
            show_line_numbers=False,
            **kwargs,
        )

    async def on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self._submit()
        elif event.key in {"shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif event.key == "ctrl+v":
            if self._paste_clipboard_image():
                event.stop()
                event.prevent_default()
        elif event.key in {"up", "down"}:
            if self._maybe_history_navigate(event.key):
                event.stop()
                event.prevent_default()

    def _maybe_history_navigate(self, key: str) -> bool:
        """Navigate prompt history when at a line boundary, like a shell."""
        app = self.app
        if not isinstance(app, AgentApp):
            return False
        at_boundary = (
            self.cursor_at_first_line if key == "up" else self.cursor_at_last_line
        )
        if not at_boundary:
            return False
        direction = -1 if key == "up" else 1
        text = app.recall_prompt_history(direction, self.text)
        if text is None:
            return False
        self.load_text(text)
        self.move_cursor(self.document.end, center=False)
        return True

    def _submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.post_message(PromptSubmitted(text))
        self.load_text("")

    def _paste_clipboard_image(self) -> bool:
        """Attach an image from the clipboard; return True if one was attached."""
        try:
            grabbed = ImageGrab.grabclipboard()
        except Exception:
            grabbed = None
        if not isinstance(grabbed, PILImage.Image):
            return False
        app = self.app
        if isinstance(app, AgentApp):
            app.set_pending_image(grabbed)
            app.notify("Image attached — press Enter to send", title="Clipboard")
        return True


class InputRow(Horizontal):
    def compose(self) -> ComposeResult:
        yield StatusIndicator()
        yield PromptBox()


class ModelRow(Horizontal):
    """Top-right row holding the TMUX spinner next to the model badge."""

    def compose(self) -> ComposeResult:
        yield ThinkingIndicator()
        yield ModelBadge()


class PromptBox(Vertical):
    def __init__(self) -> None:
        super().__init__(id="prompt-box")

    def compose(self) -> ComposeResult:
        yield ModelRow(id="model-row")
        yield PromptTextArea()


class ConnectionScreen(ModalScreen):
    """Modal to select a provider and connect to the LLM API."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._stashed_effort: str | None = None
        self._profiles = load_provider_configs()
        self._profiles.setdefault("codex", ConnectionConfig("", "", "", "codex"))
        current = get_config()
        self._profiles[current.provider] = current
        self._active_provider = current.provider

    CSS = """
    ConnectionScreen {
        align: center middle;
    }

    #connection-dialog {
        width: 60;
        height: 24;
        max-height: 90%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #connection-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    #connection-dialog Label {
        margin-top: 1;
    }

    #connection-dialog .row {
        height: 3;
        width: 100%;
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        current = self._profiles[self._active_provider]
        with Vertical(id="connection-dialog"):
            with VerticalScroll(id="connection-scroll"):
                yield Label("Connection", id="dialog-title")
                yield Select(
                    [
                        ("Local (llama.cpp)", "local"),
                        ("OpenAI API", "openai"),
                        ("OpenCode Go", "opencode-go"),
                        ("Codex CLI", "codex"),
                    ],
                    value=self._active_provider,
                    id="provider-select",
                    prompt="Choose provider...",
                )
                yield Label("Base URL", id="base-url-label")
                yield Input(
                    current.base_url,
                    placeholder="http://localhost:7070/v1",
                    id="base-url-input",
                )
                yield Label("API Key", id="api-key-label")
                yield Input(
                    current.api_key,
                    password=True,
                    placeholder="API key",
                    id="api-key-input",
                )
                yield Label("Model")
                model_list = (
                    ["codex-default"]
                    if current.provider == "codex"
                    else list(OPENAI_MODELS if current.provider == "openai" else OPENCODE_GO_MODELS)
                )
                if current.provider == "local":
                    model_list = list(OPENCODE_GO_MODELS)
                if (
                    current.provider != "local"
                    and current.model
                    and current.model not in model_list
                ):
                    model_list = [current.model] + model_list
                yield Select(
                    [_model_option(model) for model in model_list],
                    value=(
                        current.model
                        if current.model in model_list
                        else "codex-default"
                        if current.provider == "codex"
                        else model_list[0]
                    ),
                    id="model-select",
                    prompt="Select model...",
                )
                yield Input(
                    current.model if current.provider == "local" else "",
                    placeholder="Enter the local model name",
                    id="local-model-input",
                )
                yield Label("Reasoning effort", id="reasoning-effort-label")
                yield Select(
                    [(effort.title(), effort) for effort in REASONING_EFFORTS],
                    value=current.reasoning_effort
                    if current.reasoning_effort in REASONING_EFFORTS
                    else "medium",
                    id="reasoning-effort-select",
                    prompt="Select reasoning effort...",
                )
                yield Label("Verify local SSL certificates", id="verify-ssl-label")
                yield Switch(
                    value=current.verify_ssl,
                    id="verify-ssl-switch",
                    animate=False,
                )
            with Horizontal(classes="row"):
                yield Button("Submit", variant="primary", id="submit-button")
                yield Button("Cancel", id="cancel-button")

    def on_mount(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        self._set_provider_fields(provider)
        self._update_reasoning_fields()
        if provider == "codex":
            self.query_one("#model-select", Select).focus()
        else:
            self.query_one("#api-key-input", Input).focus()
        if provider in {"openai", "opencode-go", "codex"}:
            api_key = self.query_one("#api-key-input", Input).value.strip()
            if provider == "codex" or api_key:
                self.run_worker(
                    self._refresh_models(api_key, str(provider)), exclusive=False
                )

    def _set_provider_fields(self, provider: object) -> None:
        is_local = provider == "local"
        has_provider = provider in {"local", "openai", "opencode-go", "codex"}
        base_url_input = self.query_one("#base-url-input", Input)
        base_url_label = self.query_one("#base-url-label", Label)
        base_url_input.display = is_local
        base_url_label.display = is_local
        base_url_input.disabled = not is_local
        api_key_input = self.query_one("#api-key-input", Input)
        api_key_label = self.query_one("#api-key-label", Label)
        api_key_input.display = provider != "codex"
        api_key_label.display = provider != "codex"
        api_key_input.disabled = provider == "codex"
        model_select = self.query_one("#model-select", Select)
        local_model_input = self.query_one("#local-model-input", Input)
        model_select.display = has_provider and not is_local
        model_select.disabled = not has_provider or is_local
        local_model_input.display = is_local
        local_model_input.disabled = not is_local
        reasoning_select = self.query_one("#reasoning-effort-select", Select)
        reasoning_label = self.query_one("#reasoning-effort-label", Label)
        reasoning_select.display = has_provider
        reasoning_label.display = has_provider
        verify_label = self.query_one("#verify-ssl-label", Label)
        verify_switch = self.query_one("#verify-ssl-switch", Switch)
        verify_label.display = is_local
        verify_switch.display = is_local
        verify_switch.disabled = not is_local

    def _capture_profile(self) -> None:
        """Keep edits made to the current provider before switching away."""
        provider = self._active_provider
        model = (
            self.query_one("#local-model-input", Input).value.strip()
            if provider == "local"
            else self._selected_model()
        )
        effort = self.query_one("#reasoning-effort-select", Select).value
        if not isinstance(effort, str):
            effort = "medium"
        self._profiles[provider] = ConnectionConfig(
            self.query_one("#base-url-input", Input).value.strip()
            if provider == "local"
            else PROVIDER_BASE_URLS[provider],
            self.query_one("#api-key-input", Input).value.strip(),
            model,
            provider,
            effort,
            self.query_one("#verify-ssl-switch", Switch).value
            if provider == "local"
            else True,
        )

    def _apply_profile(self, provider: str) -> None:
        profile = self._profiles[provider]
        self.query_one("#base-url-input", Input).value = profile.base_url
        self.query_one("#api-key-input", Input).value = profile.api_key
        self.query_one("#local-model-input", Input).value = profile.model
        reasoning = self.query_one("#reasoning-effort-select", Select)
        reasoning.value = profile.reasoning_effort
        self.query_one("#verify-ssl-switch", Switch).value = profile.verify_ssl

    def _update_reasoning_fields(self, selected_model: str | None = None) -> None:
        """Enable or fade the reasoning-effort picker for the selected model.

        Models that don't accept `reasoning_effort` get the effort snapped to
        "off" and the control disabled (dimmed); the prior effort is stashed
        and restored when a supported model is selected again.
        """
        model = selected_model or self._selected_model()
        provider = self.query_one("#provider-select", Select).value
        if provider not in {"local", "openai", "opencode-go", "codex"}:
            return
        supported = supports_reasoning_effort(model, provider)
        select = self.query_one("#reasoning-effort-select", Select)
        label = self.query_one("#reasoning-effort-label", Label)
        if supported:
            select.disabled = False
            label.disabled = False
            if self._stashed_effort is not None:
                select.value = self._stashed_effort
                self._stashed_effort = None
        else:
            if select.value != "off":
                self._stashed_effort = select.value
            select.value = "off"
            select.disabled = True
            label.disabled = True

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-select":
            self._capture_profile()
            self._active_provider = str(event.value)
            self._apply_profile(self._active_provider)
            self._set_provider_fields(event.value)
            if event.value in PROVIDER_BASE_URLS:
                self.query_one("#base-url-input", Input).value = PROVIDER_BASE_URLS[
                    event.value
                ]
                model_select = self.query_one("#model-select", Select)
                fallback_models = list(
                    ["codex-default"]
                    if event.value == "codex"
                    else OPENAI_MODELS
                    if event.value == "openai"
                    else OPENCODE_GO_MODELS
                )
                profile = self._profiles[self._active_provider]
                if profile.model and profile.model not in fallback_models:
                    fallback_models.insert(0, profile.model)
                model_select.set_options(
                    [(model, model) for model in fallback_models]
                )
                model_select.value = profile.model or "codex-default"
                api_key = self.query_one("#api-key-input", Input).value.strip()
                if event.value == "codex" or api_key:
                    await self._refresh_models(api_key, str(event.value))
        if event.select.id in {"provider-select", "model-select"}:
            selected_model = (
                str(event.value) if event.select.id == "model-select" else None
            )
            self._update_reasoning_fields(selected_model)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss()
            return
        if event.button.id == "submit-button":
            self._connect()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit modal fields without leaking them to the chat input."""
        event.stop()
        self._connect()

    async def _refresh_models(self, api_key: str, provider: str = "opencode-go") -> None:
        select = self.query_one("#model-select", Select)
        previously_selected = select.value
        select.loading = True
        models = (
            await fetch_openai_models(api_key)
            if provider == "openai"
            else await fetch_codex_models(
                self._profiles["codex"].codex_binary,
                self._profiles["codex"].codex_home,
            )
            if provider == "codex"
            else await fetch_opencode_go_models(api_key)
        )
        select.loading = False
        select.set_options([_model_option(model) for model in models])
        # Keep the user's selection when it is still offered by the live list;
        # only fall back to the first model when it is gone.
        if previously_selected in models:
            select.value = previously_selected
        elif models:
            select.value = models[0]
        self._update_reasoning_fields()

    def _connect(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        if provider not in {"local", "openai", "opencode-go", "codex"}:
            self.notify("Choose a provider first", severity="warning")
            return
        if provider == "codex":
            base_url = ""
            api_key = ""
            model = self._selected_model()
        elif provider != "local":
            base_url = PROVIDER_BASE_URLS.get(str(provider), OPENCODE_GO_BASE_URL)
            api_key = self.query_one("#api-key-input", Input).value.strip()
            model = self._selected_model()
            if not api_key:
                self.notify(
                    f"Enter your {str(provider).replace('-', ' ').title()} API key",
                    title="Missing API key",
                    severity="error",
                )
                return
        else:
            base_url = self.query_one("#base-url-input", Input).value.strip()
            api_key = self.query_one("#api-key-input", Input).value.strip()
            model = self._selected_model()
            if not model:
                self.notify("Enter the local model name", severity="error")
                return
        verify_ssl = (
            self.query_one("#verify-ssl-switch", Switch).value
            if provider == "local"
            else True
        )
        effort = self.query_one("#reasoning-effort-select", Select).value
        if not isinstance(effort, str):
            effort = "medium"
        if not supports_reasoning_effort(model, str(provider)):
            effort = "off"
        config = configure_openai(
            base_url,
            api_key,
            model,
            provider=str(provider),
            reasoning_effort=effort,
            verify_ssl=verify_ssl,
            codex_binary=self._profiles[str(provider)].codex_binary,
            codex_home=self._profiles[str(provider)].codex_home,
        )
        self._profiles[str(provider)] = config
        save_provider_configs(self._profiles, str(provider))
        app = self.app
        if isinstance(app, AgentApp):
            app.query_one(ModelBadge).update_config(config)
        self.dismiss()
        self.app.notify(f"Connected to {model}", title="Connection updated")

    def _selected_model(self) -> str:
        provider = self.query_one("#provider-select", Select).value
        if provider == "local":
            return self.query_one("#local-model-input", Input).value.strip()
        value = self.query_one("#model-select", Select).value
        if isinstance(value, str):
            return "" if value == "codex-default" else value
        return OPENCODE_GO_MODELS[0]


class AskUserScreen(ModalScreen):
    """Modal asking the user a question with optional predefined choices."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    CSS = """
    AskUserScreen {
        align: center middle;
    }

    #ask-dialog {
        width: 60;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #ask-question {
        margin-bottom: 1;
    }

    #ask-dialog #ask-options {
        margin-bottom: 1;
    }

    #ask-dialog Button {
        margin-right: 1;
    }

    #ask-dialog #ask-input {
        margin-top: 1;
    }
    """

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__()
        self.question = question
        self.options = options or []

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-dialog"):
            yield Label(self.question, id="ask-question")
            if self.options:
                yield Select(
                    [(option, option) for option in self.options],
                    prompt="Select an option...",
                    id="ask-options",
                )
            yield Input(placeholder="Type an answer...", id="ask-input")
            with Horizontal(classes="row"):
                yield Button("Submit", variant="primary", id="ask-submit")
                yield Button("Cancel", id="ask-cancel")

    def on_mount(self) -> None:
        self.query_one("#ask-input", Input).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ask-options" and event.value is not Select.NULL:
            self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "ask-cancel":
            self.dismiss(None)
            return
        if button_id == "ask-submit":
            answer = self.query_one("#ask-input", Input).value.strip()
            if answer:
                self.dismiss(answer)
            else:
                self.notify("Enter an answer first", severity="warning")
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        answer = self.query_one("#ask-input", Input).value.strip()
        if answer:
            self.dismiss(answer)


class MemoryScreen(ModalScreen):
    """Modal to view and switch between named agent memories."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    CSS = """
    MemoryScreen {
        align: center middle;
    }

    #memory-dialog {
        width: 60;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #memory-dialog Label {
        margin-top: 1;
    }

    #memory-dialog Button {
        margin-right: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._memories: list[dict] = []

    def _refresh_memories(self) -> None:
        self._memories = list_memories()

    def compose(self) -> ComposeResult:
        # Auto-create the default memory (and activate it) when none exist, so
        # the picker always has something to select and switch to. Memories are
        # also created on the fly by the agent's memory tool when it saves a
        # note under a new name.
        if not list_memories():
            memory = ensure_general_memory()
            if not get_active_memory_id():
                set_active_memory_id(memory["id"])
            app = self.app
            if isinstance(app, AgentApp):
                app._refresh_system_prompt()
        self._refresh_memories()
        active = get_active_memory_id()
        with Vertical(id="memory-dialog"):
            yield Label("Memories", id="dialog-title")
            yield Select(
                [(memory["name"], memory["id"]) for memory in self._memories],
                value=active if active else Select.NULL,
                prompt="Select active memory...",
                id="memory-select",
            )
            with Horizontal(classes="row"):
                yield Button("Switch", variant="primary", id="memory-switch")
                yield Button("Delete", variant="error", id="memory-delete")
                yield Button("Cancel", id="memory-cancel")

    def on_mount(self) -> None:
        self.query_one("#memory-select", Select).focus()

    def _selected_id(self) -> str | None:
        value = self.query_one("#memory-select", Select).value
        return value if isinstance(value, str) and value else None

    def _switch(self, memory_id: str | None) -> None:
        if not memory_id:
            self.notify("Pick a memory to switch to", severity="warning")
            return
        memory = find_memory_by_id(memory_id)
        if memory is None:
            self.notify("Unknown memory", severity="warning")
            return
        set_active_memory_id(memory_id)
        app = self.app
        if isinstance(app, AgentApp):
            app._refresh_system_prompt()
            app.notify(f"Active memory: {memory['name']}", title="Memory")
        self.dismiss()

    async def _delete_current(self) -> None:
        memory_id = self._selected_id()
        if not memory_id:
            self.notify("Select a memory to delete", severity="warning")
            return
        memory = find_memory_by_id(memory_id)
        if memory is None:
            self.notify("Unknown memory", severity="warning")
            return
        answer = await self.app.push_screen_wait(
            AskUserScreen(
                f"Delete memory '{memory['name']}'? This cannot be undone.",
                ["Delete", "Cancel"],
            )
        )
        if answer != "Delete":
            return
        delete_memory(memory_id)
        self._refresh_memories()
        select = self.query_one("#memory-select", Select)
        select.set_options(
            [(memory["name"], memory["id"]) for memory in self._memories]
        )
        active = get_active_memory_id()
        select.value = active if active else Select.NULL
        app = self.app
        if isinstance(app, AgentApp):
            app._refresh_system_prompt()
        self.notify(f"Deleted memory '{memory['name']}'", title="Memory")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "memory-select" and event.value is not Select.NULL:
            # Select posts a Changed with the preset value on mount; skip it so
            # the modal doesn't immediately dismiss itself.
            if event.value != get_active_memory_id():
                self._switch(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "memory-cancel":
            self.dismiss()
            return
        if event.button.id == "memory-switch":
            self._switch(self._selected_id())
        elif event.button.id == "memory-delete":
            self.app.run_worker(self._delete_current(), exclusive=False)


class AgentScreen(Screen):
    """Default screen. Overrides ctrl+c copy to confirm the selection copy."""

    def action_copy_text(self) -> None:
        selection = self.get_selected_text()
        if selection is None:
            raise SkipAction()
        self.app.copy_to_clipboard(selection)
        self.app.notify("Copied to clipboard", title="Selection")


class AgentApp(App):
    """Textual TUI for the Remie coding assistant."""

    TITLE = "Remie"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c,super+c", "copy_or_quit", "Copy/Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
        ("ctrl+o", "open_memory", "Memories"),
        ("ctrl+p", "open_connection", "Connect"),
        ("ctrl+t", "toggle_theme", "Toggle theme"),
        ("escape", "stop_agent", "Stop agent"),
    ]

    def get_key_display(self, binding: Binding) -> str:
        """Render keys like `Ctrl+p` in the footer instead of Textual's `^p`."""
        modifiers, key = binding.parse_key()
        key = format_key(key)
        display_mods = [modifier.title() for modifier in modifiers]
        return "+".join([*display_mods, key])

    def get_default_screen(self) -> Screen:
        return AgentScreen(id="_default")

    def __init__(self) -> None:
        super().__init__()
        self.conversation: list[dict[str, Any]] = []
        self._cached_conv_tokens = 0
        self.theme = "ansi-dark"
        self._agent_running = False
        self._stop_requested = False
        self._input_queue: asyncio.Queue[str | list | None] = asyncio.Queue()
        self._pending_image: PILImage.Image | None = None
        self._prompt_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.debug_mode = os.environ.get("REMIE_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def compose(self) -> ComposeResult:
        yield Header()
        yield StreamingRichLog(id="log", markup=True, wrap=True)
        yield InputRow(id="input-row")
        yield Footer()

    def _code_theme(self) -> str:
        return "ansi_light" if self.theme == "ansi-light" else "ansi_dark"

    def _set_status(self, status: str) -> None:
        self.query_one(StatusIndicator).set_status(status)
        self.query_one(ThinkingIndicator).set_status(status)

    def on_mount(self) -> None:
        self.sub_title = ""
        prompt = self.query_one("#prompt", PromptTextArea)
        self.query_one(ModelBadge).update_config(get_config())
        create_launch_memory()
        clear_session()
        self.conversation = [{"role": "system", "content": get_full_system_prompt()}]
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)
        prompt.focus()
        self._prefetch_model_context()

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system message from the current prompt (incl. memory) and
        keep the conversation token cache in sync."""
        new_system = {"role": "system", "content": get_full_system_prompt()}
        if self.conversation and self.conversation[0]["role"] == "system":
            self._cached_conv_tokens += estimate_message_tokens(
                new_system
            ) - estimate_message_tokens(self.conversation[0])
            self.conversation[0] = new_system
        else:
            self.conversation.insert(0, new_system)
            self._cached_conv_tokens += estimate_message_tokens(new_system)

    def _save_session(self) -> None:
        save_session(self.conversation)

    def on_unmount(self) -> None:
        """Persist the conversation so a later launch can resume it."""
        try:
            self._save_session()
        except Exception:
            pass

    @work(exclusive=False)
    async def _prefetch_model_context(self) -> None:
        """Populate the live context-window cache when connected to OpenCode Go,
        so compaction uses the actual model window without opening the picker."""
        config = get_config()
        if config.provider not in {"openai", "opencode-go"} or not config.api_key:
            return
        try:
            if config.provider == "openai":
                await fetch_openai_models(config.api_key)
            else:
                await fetch_opencode_go_models(config.api_key)
        except Exception:
            pass

    def set_pending_image(self, image: PILImage.Image) -> None:
        self._pending_image = image

    def _record_prompt_history(self, text: str) -> None:
        if not text:
            return
        if self._prompt_history and self._prompt_history[-1] == text:
            return
        self._prompt_history.append(text)
        if len(self._prompt_history) > PROMPT_HISTORY_LIMIT:
            del self._prompt_history[:-PROMPT_HISTORY_LIMIT]
        self._history_index = None
        self._history_draft = ""

    def recall_prompt_history(self, direction: int, current_text: str) -> str | None:
        """Recall prompt history; returns the text to show or None to abort."""
        if not self._prompt_history:
            return None
        if self._history_index is None:
            if direction < 0:
                self._history_draft = current_text
                self._history_index = len(self._prompt_history) - 1
                return self._prompt_history[self._history_index]
            return None
        target = self._history_index + direction
        if target < 0:
            self._history_index = 0
            return self._prompt_history[0]
        if target >= len(self._prompt_history):
            self._history_index = None
            return self._history_draft
        self._history_index = target
        return self._prompt_history[target]

    def _image_to_content(self, image: PILImage.Image) -> list[dict[str, Any]]:
        buffer = io.BytesIO()
        image.convert("RGBA").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        user_input = event.text.strip()
        if not user_input:
            return
        if user_input.lower() in {"exit", "quit", "keluar"}:
            self.exit()
            return
        self._record_prompt_history(user_input)
        log = self.query_one("#log", StreamingRichLog)
        content: str | list = user_input
        if self._pending_image is not None:
            content = [
                {"type": "text", "text": user_input},
                self._image_to_content(self._pending_image),
            ]
            self._pending_image = None
            log.write(render_user_message(user_input))
            log.write("[dim]📷 image attached[/]")
        else:
            log.write(render_user_message(user_input))
        log.write("")
        self._input_queue.put_nowait(content)
        if not self._agent_running:
            _ = self.message_worker()

    @work(exclusive=True)
    async def message_worker(self) -> None:
        """Consume queued user messages, processing them one at a time."""
        try:
            while True:
                user_content = await self._input_queue.get()
                self._input_queue.task_done()
                if user_content is None or self._stop_requested:
                    self._drain_queue()
                    break
                self._set_status("working")
                await self.run_agent_turn(user_content)
                if self._stop_requested:
                    self._drain_queue()
                    break
        finally:
            self._stop_requested = False
            try:
                self.query_one("#prompt", PromptTextArea).focus()
            except Exception:
                pass

    def _drain_queue(self) -> None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
                self._input_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def _context_limit(self) -> int | None:
        config = get_config()
        return get_model_context_limit(config.model, config.provider)

    def _push_message(self, role: str, content: Any) -> None:
        """Append a message to the conversation and keep the running token
        estimate current so compaction checks stay O(1)."""
        message = {"role": role, "content": content}
        self.conversation.append(message)
        self._cached_conv_tokens += estimate_message_tokens(message)

    def _conversation_too_large(self, limit: int | None) -> bool:
        if not limit:
            return False
        tokens = self._cached_conv_tokens or estimate_conversation_tokens(
            self.conversation
        )
        return tokens >= limit * COMPACTION_CONTEXT_RATIO

    async def _compact_conversation(self) -> None:
        """Trim old context when the window is nearly full so long tasks continue.

        The messages being dropped are summarized into a compact "session
        memory" note that stays in the conversation; the terse omitted-note
        fallback is used when the summary call fails or yields nothing.
        """
        if len(self.conversation) <= 2:
            return
        dropped = self.conversation[1:-COMPACTION_KEEP_MESSAGES]
        summary = await summarize_messages(dropped) if dropped else ""
        note = summary or (
            "(Earlier conversation was omitted because the context window was "
            "nearly full. Continue based on the most recent messages below.)"
        )
        tail = self.conversation[1:][-COMPACTION_KEEP_MESSAGES:]
        self.conversation = self.conversation[:1] + [
            {"role": "system", "content": note}
        ] + tail
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)

    async def run_agent_turn(self, user_content: str | list) -> None:
        log = self.query_one("#log", StreamingRichLog)
        completed = False
        try:
            self._agent_running = True
            self._push_message("user", user_content)
            continuations = 0
            empty_retries = 0
            while True:
                if self._stop_requested:
                    log.write("[dim]Stopped by user[/]")
                    break
                if self._conversation_too_large(self._context_limit()):
                    await self._compact_conversation()
                    log.write(
                        "[dim]Context window nearly full — older messages compacted.[/]"
                    )
                full_text = ""
                full_chars = 0
                full_newlines = 0
                log.begin_stream()
                tool_detected = False
                tool_rendered = False
                usage_box: dict[str, int] = {}
                reasoning_box: list[str] = []
                finish_box: dict[str, Any] = {}
                badge = self.query_one(ModelBadge)
                stream_started = time.monotonic()
                last_preview_update = stream_started
                reasoning_text = ""
                reasoning_chunks_done = 0
                async for delta in stream_llm_call(
                    self.conversation, usage_box, reasoning_box, finish_box
                ):
                    if self._stop_requested:
                        break
                    full_text += delta
                    # Keep O(1) counters for the token-speed estimate so the
                    # badge does not re-scan the whole accumulation per update.
                    full_chars += len(delta)
                    full_newlines += delta.count("\n")
                    # A complete tool-call line only exists once its trailing
                    # newline has arrived, so only rescan at line boundaries
                    # instead of re-scanning the whole accumulating text on
                    # every token.
                    if not tool_detected:
                        if "\n" in delta or _has_tool_call(delta):
                            tool_detected = _has_tool_call(full_text)
                    now = time.monotonic()
                    # Re-rendering the entire accumulated Markdown (parse +
                    # Pygments + layout) per token is the main CPU cost of
                    # streaming. Throttle the preview with an interval that
                    # grows with the text size, but force an immediate render
                    # when a tool call is first detected so the preview mode
                    # switches without delay.
                    should_render = _should_update_stream(
                        len(full_text), last_preview_update, now
                    ) or (tool_detected and not tool_rendered)
                    if tool_detected:
                        tool_rendered = True
                    if not should_render:
                        continue
                    last_preview_update = now
                    if len(reasoning_box) > reasoning_chunks_done:
                        reasoning_text += "".join(
                            reasoning_box[reasoning_chunks_done:]
                        )
                        reasoning_chunks_done = len(reasoning_box)
                    if tool_detected:
                        shown = reasoning_text or extract_thinking(full_text)
                    elif reasoning_text:
                        shown = reasoning_text
                    else:
                        shown = ""
                    elapsed = now - stream_started
                    if elapsed > 0:
                        badge.set_speed(
                            estimate_tokens_from_counts(full_chars, full_newlines)
                            / elapsed
                        )
                    preview = _preview_window(full_text)
                    if shown:
                        preview_shown = _preview_window(shown)
                        log.update_stream(
                            _safe_reasoning_markdown(preview_shown, self._code_theme()),
                            title="Reasoning",
                            border_style="dim",
                        )
                    elif preview:
                        log.update_stream(
                            _safe_stream_markdown(preview, self._code_theme()),
                        )
                if self._stop_requested:
                    log.replace_stream()
                    log.write("[dim]Stopped by user[/]")
                    break
                self.query_one(ModelBadge).set_speed(None)
                reasoning_text = "".join(reasoning_box) or extract_thinking(full_text)
                input_tokens = usage_box.get("prompt_tokens") or self._cached_conv_tokens
                output_tokens = usage_box.get("completion_tokens") or estimate_tokens(
                    full_text
                )
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self.query_one(ModelBadge).set_tokens(
                    self._total_input_tokens, self._total_output_tokens
                )
                tool_invocations = extract_tool_invocations(full_text)
                content = strip_protocol_lines(full_text).strip()
                if not tool_invocations and not content:
                    # The model produced no usable output (e.g. only reasoning,
                    # or the stream ended prematurely). Don't silently mark the
                    # turn done: retry a bounded number of times.
                    if empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
                        empty_retries += 1
                        log.replace_stream()
                        log.write(
                            "[dim]Agent produced no output — retrying…[/]"
                        )
                        self._push_message(
                            "assistant", reasoning_text or "(no output)"
                        )
                        continue
                    log.replace_stream()
                    if reasoning_text:
                        log.write(
                            Panel(
                                _safe_reasoning_markdown(
                                    reasoning_text, self._code_theme()
                                ),
                                title="Reasoning",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
                    log.write("[bold red]Agent stopped: empty response[/]")
                    break
                if (
                    finish_box.get("truncated")
                    and continuations < MAX_AUTO_CONTINUATIONS
                    and not tool_invocations
                ):
                    self._push_message("assistant", full_text)
                    partial = strip_protocol_lines(full_text).strip()
                    if partial:
                        log.replace_stream(
                            render_assistant_panel(partial, self._code_theme())
                        )
                    else:
                        log.replace_stream()
                    continuations += 1
                    continue
                if not tool_invocations:
                    content = strip_protocol_lines(full_text).strip()
                    renderables = []
                    if reasoning_text:
                        renderables.append(
                            Panel(
                                _safe_reasoning_markdown(
                                    reasoning_text, self._code_theme()
                                ),
                                title="Reasoning",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
                    if content:
                        renderables.append(
                            render_assistant_panel(content, self._code_theme())
                        )
                    log.replace_stream(*renderables)
                    self._push_message("assistant", full_text)
                    completed = True
                    self._set_status("done")
                    return
                replacements: list[RenderableType] = []
                if reasoning_text:
                    replacements.append(
                        Panel(
                            _safe_reasoning_markdown(
                                reasoning_text, self._code_theme()
                            ),
                            title="Reasoning",
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )
                for name, args in tool_invocations:
                    if self.debug_mode:
                        tool_line = (
                            f"[bold cyan]Agent {escape(name)}"
                            f"({escape(json.dumps(args))})[/]"
                        )
                    else:
                        tool_line = (
                            "[bold cyan]Agent "
                            f"{escape(get_tool_summary(name))}[/]"
                        )
                    replacements.append(tool_line)
                log.replace_stream(*replacements)
                self._push_message("assistant", full_text)
                for name, args in tool_invocations:
                    if self._stop_requested:
                        log.write("[dim]Stopped by user[/]")
                        break
                    if name == "ask_user":
                        question = str(args.get("question", ""))
                        options = args.get("options") or []
                        log.write(
                            f"[bold cyan]Agent asking you:[/] {escape(question)}"
                        )
                        answer = await self.push_screen_wait(
                            AskUserScreen(question, options)
                        )
                        if answer is None:
                            result = {"answer": None, "cancelled": True}
                        else:
                            result = {"answer": answer}
                    else:
                        result = await asyncio.to_thread(run_tool, name, args)
                    result_json = json.dumps(result, default=str)
                    if isinstance(result, dict) and result.get("diff"):
                        log.write(_render_diff(result["diff"]))
                    if isinstance(result, dict) and result.get("blocked"):
                        log.write(
                            "[bold red]Blocked command:[/] "
                            f"{escape(result.get('command', ''))} "
                            f"\u2014 {escape(str(result.get('reason', 'unsafe command')))}"
                        )
                    if isinstance(result, dict):
                        result_renderable = _render_tool_result(
                            name, result, self._code_theme()
                        )
                        if result_renderable is not None:
                            log.write(result_renderable)
                    if self.debug_mode:
                        log.write(
                            f"[bold magenta]tool_result:[/] {escape(result_json)}"
                        )
                    if name == "memory" and isinstance(result, dict) and (
                        result.get("action") in {"add", "clear"}
                    ):
                        self._refresh_system_prompt()
                    self._push_message("user", f"tool_result({result_json})")
        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as error:
            log.replace_stream()
            message = get_connection_error_message(error)
            if message is not None:
                self.notify(message, title="LLM connection error", severity="error")
        except UnsupportedModelError as error:
            log.replace_stream()
            self.notify(str(error), title="Unsupported model", severity="error")
        except LLMRequestError as error:
            log.replace_stream()
            message = str(error)
            if "context" in message.lower() or "maximum context length" in message.lower():
                self.notify(
                    "The conversation exceeded the model's context window. "
                    "Clear the log or start a new session.",
                    title="Context window full",
                    severity="error",
                )
            else:
                self.notify(message, title="Request error", severity="error")
        except Exception as error:
            log.replace_stream()
            self.notify(
                f"{type(error).__name__}: {error}",
                title="Agent error",
                severity="error",
            )
        finally:
            self._agent_running = False
            if completed:
                self._save_session()
            else:
                self._set_status("ready")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.query_one(ModelBadge).set_tokens(0, 0)
        clear_session()
        self.conversation = [{"role": "system", "content": get_full_system_prompt()}]
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)

    def action_copy_or_quit(self) -> None:
        """Copy selected text, or quit when nothing is selected."""
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.notify("Copied to clipboard", title="Selection")
            return
        self.exit()

    def action_toggle_theme(self) -> None:
        self.theme = "ansi-dark" if self.theme == "ansi-light" else "ansi-light"

    def action_stop_agent(self) -> None:
        """Request the running agent loop to stop and clear pending messages."""
        if self._agent_running:
            self._stop_requested = True
            self._input_queue.put_nowait(None)

    async def action_open_connection(self) -> None:
        """Open the connection/model picker. Ignored while the agent is busy."""
        if self._agent_running:
            return
        await self.push_screen(ConnectionScreen())

    async def action_open_memory(self) -> None:
        """Open the memory picker. Ignored while the agent is busy."""
        if self._agent_running:
            return
        await self.push_screen(MemoryScreen())


def run_tui() -> None:
    app = AgentApp()
    app.run()


def main() -> None:
    run_tui()
