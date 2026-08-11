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
from rich.markup import escape
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical
from textual.geometry import Size
from textual.strip import Strip
from textual.widgets import Footer, Header, Input, Label, RichLog
from textual_image.widget import SixelImage as TerminalImage

from chaldea.agent import (
    extract_thinking,
    extract_tool_invocations,
    get_connection_error_message,
    get_full_system_prompt,
    get_tool_summary,
    render_assistant_panel,
    render_user_message,
    run_tool,
    stream_llm_call,
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

#input-row {
    height: 4;
    width: 100%;
    padding: 0 1 1 1;
    align: left middle;
}

#status {
    width: 7;
    height: 3;
    margin-right: 1;
    content-align: center middle;
    background: $panel;
}

#status-gif {
    width: 6;
    height: 3;
}

#tmux-spinner {
    width: 16;
    height: 3;
    margin-right: 1;
    content-align: left middle;
    display: none;
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


class StreamingRichLog(RichLog):
    """A RichLog that can stream text in place at the bottom of the log."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_start: int | None = None

    def begin_stream(self) -> None:
        self._stream_start = len(self.lines)

    def update_stream(self, text: str) -> None:
        if self._stream_start is None:
            self.begin_stream()
        renderable = Panel(
            Text.from_markup(escape(text)),
            title="Assistant",
            border_style="yellow",
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
            self.update(f"{self.FRAMES[self._frame_index]} Thinking...")

    def set_status(self, status: str) -> None:
        self._working = status == "working"
        self.display = self._working and _is_tmux()
        if self._working:
            self._frame_index = 0
            self.update(f"{self.FRAMES[0]} Thinking...")


class InputRow(Horizontal):
    def compose(self) -> ComposeResult:
        yield StatusIndicator()
        yield ThinkingIndicator()
        yield Input(id="prompt", placeholder="Type a message and press Enter...")


class AgentApp(App):
    """Textual TUI for the FuiAgent coding assistant."""

    TITLE = "FuiAgent"
    CSS = CSS
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
        ("ctrl+t", "toggle_theme", "Toggle theme"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.conversation: list[ChatCompletionMessageParam] = []
        self.theme = "ansi-dark"
        self.debug_mode = os.environ.get("CHALDEA_DEBUG", "").lower() in {
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
        prompt.border_title = os.environ.get("LLAMA_MODEL", "local-model")
        prompt.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return
        if user_input.lower() in {"exit", "quit", "keluar"}:
            self.exit()
            return
        self.conversation.append({"role": "user", "content": user_input})
        log = self.query_one("#log", StreamingRichLog)
        log.write(render_user_message(user_input))
        log.write("")
        event.input.value = ""
        event.input.disabled = True
        self._set_status("working")
        _ = self.run_agent_turn()

    @work(exclusive=True)
    async def run_agent_turn(self) -> None:
        log = self.query_one("#log", StreamingRichLog)
        prompt = self.query_one("#prompt", Input)
        completed = False
        try:
            while True:
                full_text = ""
                log.begin_stream()
                async for delta in stream_llm_call(self.conversation):
                    full_text += delta
                    log.update_stream(full_text)
                tool_invocations = extract_tool_invocations(full_text)
                if not tool_invocations:
                    log.replace_stream(
                        render_assistant_panel(full_text, self._code_theme())
                    )
                    self.conversation.append(
                        {"role": "assistant", "content": full_text}
                    )
                    completed = True
                    self._set_status("done")
                    return
                thinking = extract_thinking(full_text)
                replacements: list[str] = []
                if thinking:
                    replacements.append(f"[dim]Thinking:[/] {escape(thinking)}")
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
                    result = await asyncio.to_thread(run_tool, name, args)
                    result_json = json.dumps(result, default=str)
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
        finally:
            prompt.disabled = False
            prompt.focus()
            if not completed:
                self._set_status("ready")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def action_toggle_theme(self) -> None:
        self.theme = "ansi-dark" if self.theme == "ansi-light" else "ansi-light"


def run_tui() -> None:
    app = AgentApp()
    app.run()


def main() -> None:
    run_tui()
