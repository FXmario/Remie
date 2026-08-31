"""Modal asking the user a question with optional predefined choices."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static

from remie.tui.contracts import is_agent_app
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

    #ask-header {
        height: 3;
        background: $primary;
        color: $text;
    }

    #ask-title {
        width: 1fr;
        padding: 1 2 0 2;
        text-style: bold;
    }

    #ask-close {
        width: 5;
        min-width: 5;
        height: 3;
        border: none;
        background: $primary;
        color: $text;
        margin: 0;
    }

    #ask-close:hover {
        background: $error;
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

    #ask-dialog #ask-options > .option-list--option-highlighted {
        color: $text;
        background: $primary 35%;
        text-style: bold;
    }

    #ask-dialog #ask-input {
        margin-top: 1;
    }

    #ask-dialog .ask-custom-hidden {
        display: none;
    }

    #ask-actions {
        align: right middle;
        margin-top: 1;
    }

    #ask-dialog Button {
        margin-left: 1;
    }

    #ask-footer {
        height: 1;
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
            with Horizontal(id="ask-header"):
                yield Label("❓ Agent needs your input", id="ask-title")
                yield Button("✕", id="ask-close")
            with Vertical(id="ask-body"):
                yield Static(self._question_renderable(), id="ask-question")
                choices = [
                    f"{index}. {option}"
                    for index, option in enumerate(self.options, start=1)
                ]
                choices.append(f"{len(self.options) + 1}. Write your answer")
                option_list = OptionList(*choices, id="ask-options")
                # OptionList's automatic height can omit its bottom border.
                # Reserve one row per choice plus both border rows, capped by
                # the CSS maximum so longer lists remain scrollable.
                option_list.styles.height = min(len(choices) + 2, 12)
                yield option_list
                yield Input(
                    placeholder="Write your answer…",
                    id="ask-input",
                    classes="ask-custom-hidden",
                )
            with Horizontal(id="ask-actions"):
                yield Button(
                    "Submit",
                    variant="primary",
                    id="ask-submit",
                    classes="ask-custom-hidden",
                )
            yield Label(
                "↑/↓ choose · 1–9 quick-pick · Enter select · Esc cancel",
                id="ask-footer",
            )

    def on_mount(self) -> None:
        self.query_one("#ask-options", OptionList).focus()

    def _show_custom_answer(self) -> None:
        """Replace the choices with the free-form answer controls."""
        # Removing the option list frees its layout height before the input is
        # revealed, preventing the dialog's max-height from clipping the input.
        self.query_one("#ask-options", OptionList).add_class("ask-custom-hidden")
        answer_input = self.query_one("#ask-input", Input)
        answer_input.remove_class("ask-custom-hidden")
        self.query_one("#ask-submit", Button).remove_class("ask-custom-hidden")
        answer_input.focus()

    def action_pick(self, index: int) -> None:
        """Quick-pick an option or the numbered custom-answer entry."""
        if index < len(self.options):
            self.dismiss(self.options[index])
        elif index == len(self.options):
            self._show_custom_answer()

    def action_cancel(self) -> None:
        app = self.app
        if is_agent_app(app):
            app.action_stop_agent()
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "ask-options":
            return
        if event.option_index < len(self.options):
            self.dismiss(self.options[event.option_index])
        else:
            self._show_custom_answer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "ask-close":
            self.action_cancel()
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
