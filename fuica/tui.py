import asyncio
import json
import os
from pathlib import Path
import re
import select
import sys
import time
from typing import ClassVar

import httpx
from PIL import Image as PILImage
from PIL import ImageSequence
from openai import APIConnectionError, APITimeoutError
from openai.types.chat import ChatCompletionMessageParam
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select
from textual_image.widget import SixelImage as TerminalImage

from fuica.agent import (
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
    get_tool_summary,
    render_assistant_panel,
    render_user_message,
    run_tool,
    save_config,
    stream_llm_call,
    strip_protocol_lines,
)

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
    width: 6;
    height: 3;
    margin-right: 0;
    content-align: center middle;
    background: $panel;
}

#status-gif {
    width: 6;
    height: 3;
}

#tmux-spinner {
    width: 1;
    height: 3;
    margin-right: 0;
    content-align: left middle;
    display: none;
}

#model-badge {
    dock: top;
    align: right top;
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
            except OSError, ValueError:
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


def _safe_stream_markdown(text: str, code_theme: str) -> RenderableType:
    """Render partial Markdown safely during streaming.

    Auto-closes incomplete code fences so Pygments highlighting engages
    as soon as the opening fence and language arrive.  Falls back to
    plain escaped text if the Markdown parser rejects the content.
    """
    fence_count = text.count("\n```") + (1 if text.startswith("```") else 0)
    if fence_count % 2 == 1:
        text = text + "\n```"
    try:
        return Markdown(text, code_theme=code_theme, hyperlinks=True)
    except Exception:
        return Text.from_markup(escape(text))


def _has_tool_call(text: str) -> bool:
    """Return True if any complete line starts with the tool: prefix."""
    return any(
        line.strip().startswith("tool:") for line in text.splitlines() if line.strip()
    )


def _format_tokens(count: int) -> str:
    """Format a token count compactly, e.g. 1234 -> '1.2k'."""
    if count >= 1000:
        value = count / 1000.0
        if value == int(value):
            return f"{int(value)}k"
        return f"{value:.1f}k"
    return str(count)


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


class StreamingRichLog(RichLog):
    """A RichLog that can stream text in place at the bottom of the log."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_start: int | None = None

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
        self.update_config(get_config())

    def update_config(self, config) -> None:
        vendor = (
            "OpenCode Go"
            if config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL
            else "Local"
        )
        self._model_text = config.model
        self._vendor_text = vendor
        self._show()

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._show(input_tokens, output_tokens)

    def _show(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        text = f"{self._model_text}  {self._vendor_text}"
        if input_tokens or output_tokens:
            total = input_tokens + output_tokens
            text += f" · {_format_tokens(total)} tok"
        self.update(text)

    async def on_click(self) -> None:
        app = self.app
        if isinstance(app, AgentApp):
            await app.action_open_connection()


class InputRow(Horizontal):
    def compose(self) -> ComposeResult:
        yield StatusIndicator()
        yield ThinkingIndicator()
        yield PromptBox()


class PromptBox(Vertical):
    def __init__(self) -> None:
        super().__init__(id="prompt-box")

    def compose(self) -> ComposeResult:
        yield Input(id="prompt", placeholder="Type a message and press Enter...")
        yield ModelBadge()


class ConnectionScreen(ModalScreen):
    """Modal to select a provider and connect to the LLM API."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    CSS = """
    ConnectionScreen {
        align: center middle;
    }

    #connection-dialog {
        width: 60;
        height: 20;
        max-height: 80%;
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

    #connection-dialog Button {
        margin-top: 1;
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
                    value="local",
                    id="provider-select",
                    prompt="Choose provider...",
                )
                yield Label("Base URL")
                yield Input(
                    current.base_url,
                    placeholder="http://localhost:1234/v1",
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
                yield Select(
                    [(model, model) for model in OPENCODE_GO_MODELS],
                    value=current.model
                    if current.model in OPENCODE_GO_MODELS
                    else OPENCODE_GO_MODELS[0],
                    id="model-select",
                    prompt="Select model...",
                )
                with Horizontal(classes="row"):
                    yield Button("Refresh models", id="refresh-button")
                    yield Button("Submit", variant="primary", id="submit-button")
                    yield Button("Cancel", id="cancel-button")

    def on_mount(self) -> None:
        self.query_one("#model-select", Select).disabled = True
        self.query_one("#api-key-input", Input).focus()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-select":
            is_go = event.value == "opencode-go"
            self.query_one("#base-url-input", Input).disabled = is_go
            self.query_one("#model-select", Select).disabled = not is_go
            if is_go:
                self.query_one("#base-url-input", Input).value = OPENCODE_GO_BASE_URL
                api_key = self.query_one("#api-key-input", Input).value.strip()
                if api_key:
                    await self._refresh_models(api_key)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss()
            return
        if event.button.id == "submit-button":
            self._connect()
        elif event.button.id == "refresh-button":
            api_key = self.query_one("#api-key-input", Input).value.strip()
            if api_key:
                await self._refresh_models(api_key)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit modal fields without leaking them to the chat input."""
        event.stop()
        self._connect()

    async def _refresh_models(self, api_key: str) -> None:
        select = self.query_one("#model-select", Select)
        select.loading = True
        models = await fetch_opencode_go_models(api_key)
        select.loading = False
        select.set_options([(model, model) for model in models])
        select.value = models[0] if models else OPENCODE_GO_MODELS[0]

    def _connect(self) -> None:
        provider = self.query_one("#provider-select", Select).value
        if provider == "opencode-go":
            base_url = OPENCODE_GO_BASE_URL
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
        config = configure_openai(base_url, api_key, model)
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


class AgentApp(App):
    """Textual TUI for the FuiAgent coding assistant."""

    TITLE = "FuiAgent"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
        ("ctrl+p", "open_connection", "Connect"),
        ("ctrl+t", "toggle_theme", "Toggle theme"),
        ("escape", "stop_agent", "Stop agent"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.conversation: list[ChatCompletionMessageParam] = []
        self.theme = "ansi-dark"
        self._agent_running = False
        self._stop_requested = False
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
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
        prompt = self.query_one("#prompt", Input)
        self.query_one(ModelBadge).update_config(get_config())
        prompt.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return
        if user_input.lower() in {"exit", "quit", "keluar"}:
            self.exit()
            return
        log = self.query_one("#log", StreamingRichLog)
        log.write(render_user_message(user_input))
        log.write("")
        event.input.value = ""
        self._input_queue.put_nowait(user_input)
        if not self._agent_running:
            _ = self.message_worker()

    @work(exclusive=True)
    async def message_worker(self) -> None:
        """Consume queued user messages, processing them one at a time."""
        try:
            while True:
                user_input = await self._input_queue.get()
                self._input_queue.task_done()
                if user_input is None or self._stop_requested:
                    self._drain_queue()
                    break
                self._set_status("working")
                await self.run_agent_turn(user_input)
                if self._stop_requested:
                    self._drain_queue()
                    break
        finally:
            self._stop_requested = False
            self.query_one("#prompt", Input).focus()

    def _drain_queue(self) -> None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
                self._input_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def run_agent_turn(self, user_input: str) -> None:
        log = self.query_one("#log", StreamingRichLog)
        completed = False
        try:
            self._agent_running = True
            self.conversation.append({"role": "user", "content": user_input})
            while True:
                if self._stop_requested:
                    log.write("[dim]Stopped by user[/]")
                    break
                full_text = ""
                log.begin_stream()
                tool_detected = False
                usage_box: dict[str, int] = {}
                reasoning_box: list[str] = []
                async for delta in stream_llm_call(
                    self.conversation, usage_box, reasoning_box
                ):
                    if self._stop_requested:
                        break
                    full_text += delta
                    if not tool_detected and _has_tool_call(full_text):
                        tool_detected = True
                    reasoning_text = "".join(reasoning_box)
                    if tool_detected:
                        shown = reasoning_text or extract_thinking(full_text)
                    elif reasoning_text:
                        shown = reasoning_text
                    else:
                        shown = ""
                    if shown:
                        log.update_stream(
                            Text.from_markup(escape(shown)),
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
                if not tool_invocations:
                    content = strip_protocol_lines(full_text).strip()
                    renderables = []
                    if reasoning_text:
                        renderables.append(
                            Panel(
                                Text.from_markup(escape(reasoning_text)),
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
                replacements: list[str] = []
                if reasoning_text:
                    replacements.append(f"[dim]Reasoning:[/] {escape(reasoning_text)}")
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
                    result = await asyncio.to_thread(run_tool, name, args)
                    result_json = json.dumps(result, default=str)
                    if isinstance(result, dict) and result.get("diff"):
                        log.write(_render_diff(result["diff"]))
                    if self.debug_mode:
                        log.write(
                            f"[bold magenta]tool_result:[/] {escape(result_json)}"
                        )
                    self.conversation.append(
                        {"role": "user", "content": f"tool_result({result_json})"}
                    )
        except (
            APITimeoutError,
            APIConnectionError,
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
