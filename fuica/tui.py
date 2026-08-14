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
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text
from textual import work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import BindingType
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
    TextArea,
)
from textual_image.widget import SixelImage as TerminalImage

from fuica.agent import (
    LLMRequestError,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
    UnsupportedModelError,
    configure_openai,
    estimate_conversation_tokens,
    estimate_tokens,
    extract_thinking,
    extract_tool_invocations,
    fetch_opencode_go_models,
    get_config,
    get_connection_error_message,
    get_full_system_prompt,
    get_model_context_limit,
    render_assistant_panel,
    render_user_message,
    run_tool,
    save_config,
    stream_llm_call,
    strip_protocol_lines,
)

from fuica.tools import get_tool_summary

REASONING_EFFORTS = ("off", "low", "medium", "high", "max")
PROMPT_HISTORY_LIMIT = 100
MAX_AUTO_CONTINUATIONS = int(
    os.environ.get("FUICA_MAX_AUTO_CONTINUATIONS", "10")
)
COMPACTION_CONTEXT_RATIO = 0.8
COMPACTION_KEEP_MESSAGES = 10

# The streaming preview re-renders the whole accumulated Markdown, which is
# expensive (parse + Pygments + layout). Throttling with an interval that grows
# with the text size keeps the re-layout work roughly constant over time while
# staying responsive early on and for slow streams.
STREAM_UPDATE_MIN_INTERVAL = 0.05
STREAM_UPDATE_CHARS_PER_SECOND = 50_000
PROVIDER_BASE_URLS = {
    "opencode-go": OPENCODE_GO_BASE_URL,
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
    """
    fence_count = text.count("\n```") + (1 if text.startswith("```") else 0)
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
    """Minimum seconds between streaming preview re-renders for a given size."""
    return max(
        STREAM_UPDATE_MIN_INTERVAL,
        text_len / STREAM_UPDATE_CHARS_PER_SECOND,
    )


def _should_update_stream(
    accumulated_len: int, last_update: float, now: float
) -> bool:
    """Whether the streaming preview should re-render now (throttled)."""
    return now - last_update >= _stream_update_interval(accumulated_len)


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


def _render_diff(diff: str) -> Panel:
    """Render a unified diff as a colorized panel."""
    text = Text()
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            style = "dim"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = ""
        text.append(line + "\n", style=style)
    return Panel(
        text,
        title="Diff",
        border_style="cyan",
        padding=(0, 1),
    )


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


def _render_tool_result(name: str, result: dict[str, Any]) -> RenderableType | None:
    """Render a readable, truncated panel for a tool result, or None."""
    text = _format_tool_result(name, result)
    if not text:
        return None
    if len(text) > TOOL_RESULT_MAX_CHARS:
        text = text[:TOOL_RESULT_MAX_CHARS].rstrip() + "\n… (result truncated)"
    return Panel(
        Text.from_markup(escape(text)),
        title=f"Tool result · {name}",
        border_style="blue",
        padding=(0, 1),
    )


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

    def __init__(self) -> None:
        super().__init__(id="status")
        self._frames = {
            status: _load_status_gif(f"{status}.gif")
            for status in ("ready", "working", "done")
        }
        self._state = "ready"
        self._frame_index = 0
        self._timer = None

    def compose(self) -> ComposeResult:
        yield TerminalImage(self._frames[self._state][0][0], id="status-gif")

    def on_mount(self) -> None:
        if not _is_tmux():
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        durations = self._frames[self._state][1]
        self._timer = self.set_timer(durations[self._frame_index], self._advance)

    def _advance(self) -> None:
        frames = self._frames[self._state][0]
        self._frame_index = (self._frame_index + 1) % len(frames)
        self.query_one("#status-gif", TerminalImage).image = frames[self._frame_index]
        if not _is_tmux():
            self._schedule_next_frame()

    def set_status(self, status: str) -> None:
        if status not in self._frames:
            raise ValueError(f"Unknown status: {status}")
        if self._timer is not None:
            self._timer.stop()
        self._state = status
        self._frame_index = 0
        self.query_one("#status-gif", TerminalImage).image = self._frames[status][0][0]
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
        vendor = (
            "OpenCode Go"
            if config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL
            else "Local"
        )
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
        text = f"{self._model_text}  {self._vendor_text}"
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
        current = get_config()
        with Vertical(id="connection-dialog"):
            with VerticalScroll(id="connection-scroll"):
                yield Label("Connection", id="dialog-title")
                yield Select(
                    [
                        ("Local (llama.cpp)", "local"),
                        ("OpenCode Go", "opencode-go"),
                    ],
                    value=current.provider
                    if current.provider in {"local", "opencode-go"}
                    else "local",
                    id="provider-select",
                    prompt="Choose provider...",
                )
                yield Label("Base URL", id="base-url-label")
                yield Input(
                    current.base_url,
                    placeholder="http://localhost:7070/v1",
                    id="base-url-input",
                )
                yield Label("API Key")
                yield Input(
                    current.api_key,
                    password=True,
                    placeholder="API key",
                    id="api-key-input",
                )
                yield Label("Model")
                model_list = list(OPENCODE_GO_MODELS)
                if current.model and current.model not in model_list:
                    model_list = [current.model] + model_list
                yield Select(
                    [_model_option(model) for model in model_list],
                    value=current.model if current.model in model_list else model_list[0],
                    id="model-select",
                    prompt="Select model...",
                )
                yield Label("Reasoning effort")
                yield Select(
                    [(effort.title(), effort) for effort in REASONING_EFFORTS],
                    value=current.reasoning_effort
                    if current.reasoning_effort in REASONING_EFFORTS
                    else "medium",
                    id="reasoning-effort-select",
                    prompt="Select reasoning effort...",
                )
            with Horizontal(classes="row"):
                yield Button("Submit", variant="primary", id="submit-button")
                yield Button("Cancel", id="cancel-button")

    def on_mount(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        self._set_provider_fields(provider)
        self.query_one("#api-key-input", Input).focus()

    def _set_provider_fields(self, provider: object) -> None:
        is_local = provider == "local"
        base_url_input = self.query_one("#base-url-input", Input)
        base_url_label = self.query_one("#base-url-label", Label)
        base_url_input.display = is_local
        base_url_label.display = is_local
        base_url_input.disabled = not is_local
        self.query_one("#model-select", Select).disabled = is_local

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-select":
            self._set_provider_fields(event.value)
            if event.value in PROVIDER_BASE_URLS:
                self.query_one("#base-url-input", Input).value = PROVIDER_BASE_URLS[
                    event.value
                ]
                api_key = self.query_one("#api-key-input", Input).value.strip()
                if api_key:
                    await self._refresh_models(api_key)

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

    async def _refresh_models(self, api_key: str) -> None:
        select = self.query_one("#model-select", Select)
        previously_selected = select.value
        select.loading = True
        models = await fetch_opencode_go_models(api_key)
        select.loading = False
        select.set_options([_model_option(model) for model in models])
        # Keep the user's selection when it is still offered by the live list;
        # only fall back to the first model when it is gone.
        if previously_selected in models:
            select.value = previously_selected
        elif models:
            select.value = models[0]

    def _connect(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        if provider != "local":
            base_url = PROVIDER_BASE_URLS.get(str(provider), OPENCODE_GO_BASE_URL)
            api_key = self.query_one("#api-key-input", Input).value.strip()
            model = self._selected_model()
            if not api_key:
                self.notify(
                    "Enter your OpenCode Go API key",
                    title="Missing API key",
                    severity="error",
                )
                return
        else:
            base_url = self.query_one("#base-url-input", Input).value.strip()
            api_key = self.query_one("#api-key-input", Input).value.strip()
            model = self._selected_model()
        effort = self.query_one("#reasoning-effort-select", Select).value
        if not isinstance(effort, str):
            effort = "medium"
        config = configure_openai(
            base_url,
            api_key,
            model,
            provider=str(provider),
            reasoning_effort=effort,
        )
        save_config(config)
        app = self.app
        if isinstance(app, AgentApp):
            app.query_one(ModelBadge).update_config(config)
        self.dismiss()
        self.app.notify(f"Connected to {model}", title="Connection updated")

    def _selected_model(self) -> str:
        value = self.query_one("#model-select", Select).value
        if isinstance(value, str):
            return value
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

    #ask-dialog .option-row {
        height: auto;
        align: left middle;
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
                with Horizontal(classes="option-row"):
                    for index, option in enumerate(self.options):
                        yield Button(
                            option, id=f"ask-option-{index}", variant="primary"
                        )
            yield Input(placeholder="Type an answer...", id="ask-input")
            with Horizontal(classes="row"):
                yield Button("Submit", variant="primary", id="ask-submit")
                yield Button("Cancel", id="ask-cancel")

    def on_mount(self) -> None:
        self.query_one("#ask-input", Input).focus()

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
        if button_id and button_id.startswith("ask-option-"):
            try:
                index = int(button_id.rsplit("-", 1)[1])
                self.dismiss(self.options[index])
            except (ValueError, IndexError):
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        answer = self.query_one("#ask-input", Input).value.strip()
        if answer:
            self.dismiss(answer)


class AgentScreen(Screen):
    """Default screen. Overrides ctrl+c copy to confirm the selection copy."""

    def action_copy_text(self) -> None:
        selection = self.get_selected_text()
        if selection is None:
            raise SkipAction()
        self.app.copy_to_clipboard(selection)
        self.app.notify("Copied to clipboard", title="Selection")


class AgentApp(App):
    """Textual TUI for the FuiAgent coding assistant."""

    TITLE = "FuiAgent"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c,super+c", "copy_or_quit", "Copy/Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
        ("ctrl+p", "open_connection", "Connect"),
        ("ctrl+t", "toggle_theme", "Toggle theme"),
        ("escape", "stop_agent", "Stop agent"),
    ]

    def get_default_screen(self) -> Screen:
        return AgentScreen(id="_default")

    def __init__(self) -> None:
        super().__init__()
        self.conversation: list[dict[str, Any]] = []
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
        self.debug_mode = os.environ.get("FUICA_DEBUG", "").lower() in {
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
        self.conversation = [{"role": "system", "content": get_full_system_prompt()}]
        self.sub_title = ""
        prompt = self.query_one("#prompt", PromptTextArea)
        self.query_one(ModelBadge).update_config(get_config())
        prompt.focus()
        self._prefetch_model_context()

    @work(exclusive=False)
    async def _prefetch_model_context(self) -> None:
        """Populate the live context-window cache when connected to OpenCode Go,
        so compaction uses the actual model window without opening the picker."""
        config = get_config()
        if config.provider != "opencode-go" or not config.api_key:
            return
        try:
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

    def _conversation_too_large(self, limit: int | None) -> bool:
        if not limit:
            return False
        return estimate_conversation_tokens(self.conversation) >= limit * COMPACTION_CONTEXT_RATIO

    def _compact_conversation(self) -> None:
        """Trim old context when the window is nearly full so long tasks continue."""
        if len(self.conversation) <= 2:
            return
        tail = self.conversation[1:][-COMPACTION_KEEP_MESSAGES:]
        self.conversation = self.conversation[:1] + [
            {
                "role": "system",
                "content": (
                    "(Earlier conversation was omitted because the context "
                    "window was nearly full. Continue based on the most "
                    "recent messages below.)"
                ),
            }
        ] + tail

    async def run_agent_turn(self, user_content: str | list) -> None:
        log = self.query_one("#log", StreamingRichLog)
        completed = False
        try:
            self._agent_running = True
            self.conversation.append({"role": "user", "content": user_content})
            continuations = 0
            while True:
                if self._stop_requested:
                    log.write("[dim]Stopped by user[/]")
                    break
                if self._conversation_too_large(self._context_limit()):
                    self._compact_conversation()
                    log.write(
                        "[dim]Context window nearly full — older messages compacted.[/]"
                    )
                full_text = ""
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
                        badge.set_speed(estimate_tokens(full_text) / elapsed)
                    if shown:
                        log.update_stream(
                            _safe_reasoning_markdown(shown, self._code_theme()),
                            title="Reasoning",
                            border_style="dim",
                        )
                    else:
                        log.update_stream(
                            _safe_stream_markdown(full_text, self._code_theme()),
                        )
                if self._stop_requested:
                    log.replace_stream()
                    log.write("[dim]Stopped by user[/]")
                    break
                self.query_one(ModelBadge).set_speed(None)
                reasoning_text = "".join(reasoning_box) or extract_thinking(full_text)
                input_tokens = usage_box.get("prompt_tokens") or (
                    estimate_conversation_tokens(self.conversation)
                )
                output_tokens = usage_box.get("completion_tokens") or estimate_tokens(
                    full_text
                )
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self.query_one(ModelBadge).set_tokens(
                    self._total_input_tokens, self._total_output_tokens
                )
                tool_invocations = extract_tool_invocations(full_text)
                if (
                    finish_box.get("truncated")
                    and continuations < MAX_AUTO_CONTINUATIONS
                    and not tool_invocations
                ):
                    self.conversation.append(
                        {"role": "assistant", "content": full_text}
                    )
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
                    self.conversation.append(
                        {"role": "assistant", "content": full_text}
                    )
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
                            f"[bold cyan]Agent calling {escape(name)}"
                            f"({escape(json.dumps(args))})[/]"
                        )
                    else:
                        tool_line = (
                            "[bold cyan]Agent calling the "
                            f"{escape(get_tool_summary(name))}[/]"
                        )
                    replacements.append(tool_line)
                log.replace_stream(*replacements)
                self.conversation.append({"role": "assistant", "content": full_text})
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
                        result_renderable = _render_tool_result(name, result)
                        if result_renderable is not None:
                            log.write(result_renderable)
                    if self.debug_mode:
                        log.write(
                            f"[bold magenta]tool_result:[/] {escape(result_json)}"
                        )
                    self.conversation.append(
                        {"role": "user", "content": f"tool_result({result_json})"}
                    )
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
            if not completed:
                self._set_status("ready")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.query_one(ModelBadge).set_tokens(0, 0)

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


def run_tui() -> None:
    app = AgentApp()
    app.run()


def main() -> None:
    run_tui()
