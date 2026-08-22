"""Modal asking the user a question with optional predefined choices."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static

from remie.tui.helpers import _safe_stream_markdown


class AskUserScreen(ModalScreen):
    """Modal asking the user a question with optional predefined choices."""

    BINDINGS = [("escape", "cancel", "Cancel")] + [
        (str(i + 1), f"pick({i})", str(i + 1)) for i in range(9)
    ]

    CSS = """
    AskUserScreen {
        align: center middle;
    }

    #ask-dialog {
        width: 72;
        max-width: 92%;
        height: auto;
        max-height: 80%;
        padding: 0;
        border: round $primary;
        background: $surface;
    }

    #ask-title {
        dock: top;
        background: $primary;
        color: $text;
        padding: 0 2;
        text-style: bold;
    }

    #ask-body {
        padding: 1 2;
    }

    #ask-question {
        border-left: thick $accent;
        margin-bottom: 1;
        padding-left: 1;
    }

    #ask-dialog #ask-options {
        height: auto;
        max-height: 12;
        margin-bottom: 1;
        background: transparent;
    }

    #ask-dialog #ask-input {
        margin-top: 1;
    }

    #ask-actions {
        align: right middle;
        margin-top: 1;
    }

    #ask-dialog Button {
        margin-left: 1;
    }

    #ask-footer {
        dock: bottom;
        padding: 0 2;
        text-style: dim;
    }
    """

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__()
        self.question = question
        self.options = options or []

    def _question_renderable(self):
        """Render the question as markdown, falling back safely to plain text."""
        from rich.text import Text

        try:
            code_theme = self.app._code_theme()
        except Exception:
            code_theme = "default"
        try:
            return _safe_stream_markdown(self.question, code_theme)
        except Exception:
            return Text(self.question)

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-dialog"):
            yield Label("❓ Agent needs your input", id="ask-title")
            with Vertical(id="ask-body"):
                yield Static(self._question_renderable(), id="ask-question")
                if self.options:
                    yield OptionList(
                        *self.options,
                        id="ask-options",
                    )
                yield Input(
                    placeholder="Or type your own answer…",
                    id="ask-input",
                )
            with Horizontal(id="ask-actions"):
                yield Button("Submit", variant="primary", id="ask-submit")
                yield Button("Cancel", id="ask-cancel")
            yield Label(
                "↑/↓ choose · 1–9 quick-pick · Enter select · Esc cancel",
                id="ask-footer",
            )

    def on_mount(self) -> None:
        if self.options:
            self.query_one("#ask-options", OptionList).focus()
        else:
            self.query_one("#ask-input", Input).focus()

    def action_pick(self, index: int) -> None:
        """Quick-pick an option by number key (1–9); ignored if out of range."""
        if not self.options or index >= len(self.options):
            return
        self.dismiss(self.options[index])

    def action_cancel(self) -> None:
        app = self.app
        if isinstance(app, AgentApp):
            app.action_stop_agent()
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "ask-options":
            self.dismiss(self.options[event.option_index])

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


from remie.tui import _agent_app_registry as _registry

_registry.register_module(__name__)
