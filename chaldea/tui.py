import asyncio
import json
from typing import ClassVar

from openai.types.chat import ChatCompletionMessageParam
from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.widgets import Footer, Header, Input, RichLog

from chaldea.agent import (
    async_execute_llm_call,
    extract_thinking,
    extract_tool_invocations,
    get_full_system_prompt,
    run_tool,
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
    dock: bottom;
    height: 3;
    margin: 0 1 1 1;
}
"""


class AgentApp(App):
    """Textual TUI for the FuiAgent coding assistant."""

    TITLE = "FuiAgent"
    CSS = CSS
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.conversation: list[ChatCompletionMessageParam] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", markup=True, wrap=True)
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
        prompt = self.query_one("#prompt", Input)
        try:
            while True:
                assistant_response = await async_execute_llm_call(self.conversation)
                tool_invocations = extract_tool_invocations(assistant_response)
                if not tool_invocations:
                    log.write(
                        f"[bold yellow]Assistant:[/] {escape(assistant_response)}"
                    )
                    self.conversation.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                    return
                thinking = extract_thinking(assistant_response)
                if thinking:
                    log.write(f"[dim]Thinking:[/] {escape(thinking)}")
                self.conversation.append(
                    {"role": "assistant", "content": assistant_response}
                )
                for name, args in tool_invocations:
                    log.write(f"[bold cyan]{name}[/] {escape(json.dumps(args))}")
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


def run_tui() -> None:
    AgentApp().run()
