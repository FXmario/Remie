"""Interactive manager for the active tab's saved memory scopes."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, Select, Static, TextArea

from remie.storage.memory import (
    memory_context, memory_path, memory_tool, read_memory,
)

SCOPES = ("global", "project", "tab")
SEPARATOR = "\n\n---\n\n"


class MemoryScreen(ModalScreen):
    """Manage individual notes; writes use optimistic concurrency checks."""

    BINDINGS = [("escape", "dismiss", "Close")]
    CSS = """
    MemoryScreen { align: center middle; }
    MemoryScreen #memory-dialog {
        width: 90%; max-width: 120; height: 95%;
        border: round $primary; background: $surface; padding: 1 2;
    }
    MemoryScreen #memory-title { text-style: bold; }
    MemoryScreen #memory-scroll { height: 1fr; min-height: 4; }
    MemoryScreen .memory-heading { text-style: bold; margin-top: 1; }
    MemoryScreen .memory-path { color: $text-muted; }
    MemoryScreen .memory-entry { height: auto; margin-bottom: 1; }
    MemoryScreen .memory-entry Checkbox { width: 5; }
    MemoryScreen .memory-entry Static { width: 1fr; }
    MemoryScreen .memory-actions { height: auto; }
    MemoryScreen #memory-editor { height: 6; }
    MemoryScreen #memory-status { height: auto; color: $warning; }
    """

    def __init__(self, context: dict) -> None:
        super().__init__()
        self.context = dict(context)
        self.snapshots = {}
        self.entries = []
        self.editing = None
        self.confirming_delete = False

    def _items(self):
        self.entries = []
        for scope in SCOPES:
            yield Label(f"{scope.title()} memory", classes="memory-heading")
            yield Static(str(memory_path(scope, self.context)), markup=False,
                         classes="memory-path")
            try:
                text = read_memory(scope, self.context)
                self.snapshots[scope] = text
            except (OSError, UnicodeError) as error:
                yield Static(f"Unable to read: {error}", markup=False)
                continue
            entries = [entry.strip() for entry in text.split(SEPARATOR) if entry.strip()]
            if not entries:
                yield Static("(empty)", id=f"memory-{scope}-content")
            for entry in entries:
                index = len(self.entries)
                self.entries.append((scope, entry))
                yield Horizontal(
                    Checkbox("", id=f"memory-check-{index}"),
                    Static(entry, markup=False), classes="memory-entry",
                )

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-dialog"):
            yield Label("Saved memories", id="memory-title")
            with VerticalScroll(id="memory-scroll"):
                yield from self._items()
            with Horizontal(classes="memory-actions"):
                yield Button("Create", id="memory-create")
                yield Button("Edit checked", id="memory-edit")
                yield Button("Delete checked", id="memory-delete", variant="error")
                yield Button("Refresh", id="memory-refresh")
                yield Button("Close", id="memory-close")
            yield Select([(s.title(), s) for s in SCOPES], value="project",
                         allow_blank=False, id="memory-scope")
            yield TextArea(id="memory-editor")
            with Horizontal(classes="memory-actions"):
                yield Button("Save", id="memory-save", variant="primary")
                yield Button("Cancel edit", id="memory-cancel")
            yield Static("Create a note, or check one note to edit.", id="memory-status")

    def _status(self, message):
        self.query_one("#memory-status", Static).update(message)

    def _selected(self):
        return [self.entries[i] for i in range(len(self.entries))
                if self.query_one(f"#memory-check-{i}", Checkbox).value]

    async def _refresh(self):
        container = self.query_one("#memory-scroll", VerticalScroll)
        await container.remove_children()
        self.snapshots = {}
        await container.mount(*self._items())
        self.confirming_delete = False
        self.query_one("#memory-delete", Button).label = "Delete checked"

    def _write(self, action, scope, content="", old_text=""):
        token = memory_context.set({**self.context, "expected_memory": self.snapshots[scope]})
        try:
            return memory_tool(action, scope, content, old_text)
        finally:
            memory_context.reset(token)

    def _reset_editor(self):
        self.editing = None
        self.query_one("#memory-editor", TextArea).load_text("")
        self.query_one("#memory-scope", Select).disabled = False

    def on_checkbox_changed(self, event: Checkbox.Changed):
        self.confirming_delete = False
        self.query_one("#memory-delete", Button).label = "Delete checked"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        name = event.button.id
        try:
            if name == "memory-close":
                self.dismiss()
            elif name == "memory-refresh":
                self._reset_editor()
                await self._refresh()
            elif name in {"memory-create", "memory-cancel"}:
                self._reset_editor()
                self._status("New note: choose a scope, enter text, then Save.")
            elif name == "memory-edit":
                selected = self._selected()
                if len(selected) != 1:
                    self._status("Check exactly one note to edit.")
                    return
                self.editing = selected[0]
                scope, text = self.editing
                self.query_one("#memory-scope", Select).value = scope
                self.query_one("#memory-scope", Select).disabled = True
                self.query_one("#memory-editor", TextArea).load_text(text)
                self._status("Editing checked note. Save to apply changes.")
            elif name == "memory-save":
                scope = self.query_one("#memory-scope", Select).value
                text = self.query_one("#memory-editor", TextArea).text
                if scope not in self.snapshots:
                    self._status("Scope unavailable. Refresh and retry.")
                    return
                result = self._write("replace" if self.editing else "add", scope,
                                     text, self.editing[1] if self.editing else "")
                if "error" in result:
                    self._status(result["error"])
                    return
                self._reset_editor()
                await self._refresh()
                self._status("Saved. Use /memory reload to refresh the agent's snapshot.")
            elif name == "memory-delete":
                selected = self._selected()
                if not selected:
                    self._status("Check one or more notes to delete.")
                    return
                if not self.confirming_delete:
                    self.confirming_delete = True
                    event.button.label = "Confirm delete"
                    self._status(f"Delete {len(selected)} checked notes? Click Confirm delete to proceed.")
                    return
                # Check every affected scope before making any changes.
                if any(read_memory(s, self.context) != self.snapshots[s]
                       for s in {s for s, _ in selected}):
                    self._status("Memory changed. Refresh and review before deleting.")
                    return
                for scope, entry in selected:
                    result = self._write("remove", scope, old_text=entry)
                    if "error" in result:
                        self._status(result["error"])
                        return
                    self.snapshots[scope] = read_memory(scope, self.context)
                self._reset_editor()
                await self._refresh()
                self._status("Checked notes deleted. Use /memory reload to refresh the agent's snapshot.")
        except (OSError, UnicodeError) as error:
            self._status(f"Memory operation failed: {error}")
