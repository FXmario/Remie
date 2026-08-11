import asyncio
import json
import os
import re
import select
import sys
import time
from typing import ClassVar

from openai.types.chat import ChatCompletionMessageParam
from rich.markup import escape
from rich.segment import Segment
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.geometry import Size
from textual.strip import Strip
from textual.widgets import Footer, Header, Input, RichLog

from chaldea.agent import (
    extract_thinking,
    extract_tool_invocations,
    get_full_system_prompt,
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
    margin: 0 1 1 1;
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
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
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
        renderable = Text.from_markup(f"[bold yellow]Assistant:[/] {escape(text)}")
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

    def compose(self) -> ComposeResult:
        yield Header()
        yield StreamingRichLog(id="log", markup=True, wrap=True)
        yield Input(id="prompt", placeholder="Type a message and press Enter...")
        yield Footer()

    def on_mount(self) -> None:
        self.conversation = [{"role": "system", "content": get_full_system_prompt()}]
        self.sub_title = "Ready"
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return
        self.conversation.append({"role": "user", "content": user_input})
        log = self.query_one("#log", StreamingRichLog)
        log.write(f"[bold blue]You:[/] {escape(user_input)}")
        event.input.value = ""
        event.input.disabled = True
        self.sub_title = "Working..."
        _ = self.run_agent_turn()

    @work(exclusive=True)
    async def run_agent_turn(self) -> None:
        log = self.query_one("#log", StreamingRichLog)
        prompt = self.query_one("#prompt", Input)
        try:
            while True:
                full_text = ""
                log.begin_stream()
                async for delta in stream_llm_call(self.conversation):
                    full_text += delta
                    log.update_stream(full_text)
                tool_invocations = extract_tool_invocations(full_text)
                if not tool_invocations:
                    log.end_stream()
                    self.conversation.append(
                        {"role": "assistant", "content": full_text}
                    )
                    return
                thinking = extract_thinking(full_text)
                replacements: list[str] = []
                if thinking:
                    replacements.append(f"[dim]Thinking:[/] {escape(thinking)}")
                for name, _ in tool_invocations:
                    replacements.append(f"[bold cyan]Agent calling {escape(name)}[/]")
                log.replace_stream(*replacements)
                self.conversation.append({"role": "assistant", "content": full_text})
                for name, args in tool_invocations:
                    result = await asyncio.to_thread(run_tool, name, args)
                    result_json = json.dumps(result, default=str)
                    log.write(f"[bold magenta]tool_result:[/] {escape(result_json)}")
                    self.conversation.append(
                        {"role": "user", "content": f"tool_result({result_json})"}
                    )
        finally:
            prompt.disabled = False
            prompt.focus()
            self.sub_title = "Ready"

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def action_toggle_theme(self) -> None:
        self.theme = "ansi-dark" if self.theme == "ansi-light" else "ansi-light"


def run_tui() -> None:
    app = AgentApp()
    if _detect_terminal_background() == "light":
        app.theme = "ansi-light"
    app.run()


def main() -> None:
    run_tui()
