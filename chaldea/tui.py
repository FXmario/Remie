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
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

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

#chat {
    width: 1fr;
    height: 1fr;
    layout: vertical;
}

#log {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
    border: round $primary;
    margin: 0 1;
}

#stream {
    height: auto;
    max-height: 8;
    padding: 0 1;
    margin: 0 1;
    color: $text;
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
        with Vertical(id="chat"):
            yield RichLog(id="log", markup=True, wrap=True)
            yield Static(id="stream", markup=True)
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
        log = self.query_one("#log", RichLog)
        log.write(f"[bold blue]You:[/] {escape(user_input)}")
        event.input.value = ""
        event.input.disabled = True
        self.sub_title = "Working..."
        _ = self.run_agent_turn()

    @work(exclusive=True)
    async def run_agent_turn(self) -> None:
        log = self.query_one("#log", RichLog)
        stream = self.query_one("#stream", Static)
        prompt = self.query_one("#prompt", Input)
        try:
            while True:
                full_text = ""
                stream.update("")
                async for delta in stream_llm_call(self.conversation):
                    full_text += delta
                    stream.update(f"[bold yellow]Assistant:[/] {escape(full_text)}")
                tool_invocations = extract_tool_invocations(full_text)
                if not tool_invocations:
                    log.write(f"[bold yellow]Assistant:[/] {escape(full_text)}")
                    self.conversation.append(
                        {"role": "assistant", "content": full_text}
                    )
                    return
                thinking = extract_thinking(full_text)
                if thinking:
                    log.write(f"[dim]Thinking:[/] {escape(thinking)}")
                self.conversation.append({"role": "assistant", "content": full_text})
                for name, args in tool_invocations:
                    log.write(f"[bold cyan]{name}[/] {escape(json.dumps(args))}")
                    result = await asyncio.to_thread(run_tool, name, args)
                    result_json = json.dumps(result, default=str)
                    log.write(f"[bold magenta]tool_result:[/] {escape(result_json)}")
                    self.conversation.append(
                        {"role": "user", "content": f"tool_result({result_json})"}
                    )
        finally:
            stream.update("")
            prompt.disabled = False
            prompt.focus()
            self.sub_title = "Ready"

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def action_toggle_theme(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


def run_tui() -> None:
    app = AgentApp()
    if _detect_terminal_background() == "light":
        app.theme = "textual-light"
    app.run()


def main() -> None:
    run_tui()
