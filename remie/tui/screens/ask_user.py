"""Modal asking the user a question with optional predefined choices."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList


class AskUserScreen(ModalScreen):
    """Modal asking the user a question with optional predefined choices."""

    BINDINGS = [("escape", "cancel", "Cancel")]

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
        height: auto;
        max-height: 12;
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
                yield OptionList(
                    *self.options,
                    id="ask-options",
                )
            yield Input(placeholder="Type an answer...", id="ask-input")
            with Horizontal(classes="row"):
                yield Button("Submit", variant="primary", id="ask-submit")
                yield Button("Cancel", id="ask-cancel")

    def on_mount(self) -> None:
        if self.options:
            self.query_one("#ask-options", OptionList).focus()
        else:
            self.query_one("#ask-input", Input).focus()

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
