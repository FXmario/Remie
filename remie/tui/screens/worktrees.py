"""Picker returning an explicit open or delete action to the application."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList


class WorktreeScreen(ModalScreen):
    BINDINGS = [("escape", "dismiss", "Cancel")]
    CSS = """
    WorktreeScreen { align: center middle; }
    #worktree-dialog { width: 90; max-width: 95%; height: 75%;
        border: round $primary; background: $surface; padding: 1 2; }
    #worktree-list { height: 1fr; }
    #worktree-actions { height: auto; }
    """

    def __init__(self, items: list[dict], occupied: set):
        super().__init__()
        self.items = items
        self.occupied = occupied

    def compose(self) -> ComposeResult:
        with Vertical(id="worktree-dialog"):
            yield Label("Git worktrees — open in a new tab")
            labels = []
            for item in self.items:
                flags = [name for name, enabled in (
                    ("main", item["main"]), ("open", item["path"] in self.occupied),
                    ("dirty", item["dirty"]), ("locked", item["locked"])) if enabled]
                labels.append(Text(f'{item["path"].name} · {item["branch"]} [{", ".join(flags)}]\n{item["path"]}'))
            yield OptionList(*labels, id="worktree-list")
            with Horizontal(id="worktree-actions"):
                yield Button("Open", id="worktree-open", variant="primary")
                yield Button("Delete worktree", id="worktree-delete", variant="error")
                yield Button("Cancel", id="worktree-cancel")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(("open", self.items[event.option_index]["path"]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "worktree-cancel":
            self.dismiss(None)
            return
        index = self.query_one(OptionList).highlighted
        if index is not None:
            action = "delete" if event.button.id == "worktree-delete" else "open"
            self.dismiss((action, self.items[index]["path"]))
