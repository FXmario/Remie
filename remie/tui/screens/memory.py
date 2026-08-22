"""Modal to view and switch between named agent memories."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from remie.tools import (
    delete_memory,
    ensure_general_memory,
    find_memory_by_id,
    get_active_memory_id,
    list_memories,
    set_active_memory_id,
)

class MemoryScreen(ModalScreen):
    """Modal to view and switch between named agent memories."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    CSS = """
    MemoryScreen {
        align: center middle;
    }

    #memory-dialog {
        width: 60;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #dialog-header {
        height: auto;
        margin-bottom: 1;
    }

    #dialog-title {
        width: 1fr;
        padding-top: 1;
        text-style: bold;
    }

    #memory-close {
        width: auto;
        min-width: 4;
        height: 3;
        border: none;
        background: $panel;
    }

    #memory-row {
        height: 3;
        width: 100%;
    }

    #memory-select {
        width: 1fr;
        margin-right: 1;
    }

    #memory-search {
        width: 1fr;
    }

    #memory-delete {
        width: auto;
        min-width: 4;
        height: 3;
        border: none;
        background: $panel;
        margin-left: 1;
        margin-right: 0;
    }

    #memory-dialog Button {
        margin-right: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._memories: list[dict] = []
        # Value the search filter programmatically assigned to the dropdown;
        # the resulting queued Select.Changed must not count as a user switch.
        self._programmatic_value: str | None = None
        # True while the search filter rebuilds options; Select.Changed fired
        # by programmatic value syncs must not count as a user switch.
        self._filtering = False

    def _refresh_memories(self) -> None:
        self._memories = list_memories()

    def _select_options(self) -> list[tuple[str, str]]:
        return [(memory["name"], memory["id"]) for memory in self._memories]

    def compose(self) -> ComposeResult:
        # Auto-create the default memory (and activate it) when none exist, so
        # the picker always has something to select and switch to. Memories are
        # also created on the fly by the agent's memory tool when it saves a
        # note under a new name.
        if not list_memories():
            memory = ensure_general_memory()
            if not get_active_memory_id():
                set_active_memory_id(memory["id"])
            app = self.app
            if isinstance(app, AgentApp):
                app._refresh_system_prompt()
        self._refresh_memories()
        # Pass the active memory as the initial value so the Select's own
        # mount-time init emits a Changed event that matches the active memory
        # (and is ignored by on_select_changed) instead of auto-picking the
        # first option and spuriously switching memories.
        active = get_active_memory_id()
        with Vertical(id="memory-dialog"):
            with Horizontal(id="dialog-header"):
                yield Label("Memories", id="dialog-title")
                yield Button("🗑", id="memory-delete")
                yield Button("✕", id="memory-close")
            with Horizontal(id="memory-row"):
                yield Select(
                    self._select_options(),
                    value=active,
                    id="memory-select",
                    prompt="Pick a memory",
                    allow_blank=False,
                )
                yield Input(
                    placeholder="Filter memories…",
                    id="memory-search",
                )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "memory-search":
            event.stop()
            self._apply_memory_filter(event.value or "")

    def _apply_memory_filter(self, query: str) -> None:
        """Narrow the memory dropdown to entries matching the query."""
        select = self.query_one("#memory-select", Select)
        q = query.strip().lower()
        options = self._select_options()
        if q:
            options = [
                (name, memory_id)
                for name, memory_id in options
                if q in name.lower() or q in memory_id.lower()
            ]
        previous = select.value
        select.set_options(options)
        values = [memory_id for _, memory_id in options]
        assigned: str | None = None
        if isinstance(previous, str) and previous in values:
            assigned = previous
        elif options:
            assigned = options[0][1]
        self._programmatic_value = assigned
        if assigned is not None:
            select.value = assigned

    def on_mount(self) -> None:
        self.query_one("#memory-select", Select).focus()

    def _selected_id(self) -> str | None:
        value = self.query_one("#memory-select", Select).value
        return str(value) if value is not Select.BLANK else None

    def _switch(self, memory_id: str | None) -> None:
        if not memory_id:
            self.notify("Pick a memory to switch to", severity="warning")
            return
        memory = find_memory_by_id(memory_id)
        if memory is None:
            self.notify("Unknown memory", severity="warning")
            return
        set_active_memory_id(memory_id)
        app = self.app
        if isinstance(app, AgentApp):
            app._refresh_system_prompt()
            app.notify(f"Active memory: {memory['name']}", title="Memory")
        self.dismiss()

    async def _delete_current(self) -> None:
        memory_id = self._selected_id()
        if not memory_id:
            self.notify("Select a memory to delete", severity="warning")
            return
        memory = find_memory_by_id(memory_id)
        if memory is None:
            self.notify("Unknown memory", severity="warning")
            return
        delete_memory(memory_id)
        self._refresh_memories()
        self._apply_memory_filter(self.query_one("#memory-search", Input).value or "")
        active = get_active_memory_id()
        if active is not None:
            memory_select = self.query_one("#memory-select", Select)
            if isinstance(active, str) and active in [
                memory_id for _, memory_id in memory_select._options
            ]:
                memory_select.value = active
        app = self.app
        if isinstance(app, AgentApp):
            app._refresh_system_prompt()
        self.notify(f"Deleted memory '{memory['name']}'", title="Memory")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "memory-select":
            return
        # Ignore programmatic syncs (search filter) and reselects of the
        # active memory; anything else is a genuine user switch.
        if event.value == self._programmatic_value:
            return
        self._programmatic_value = (
            event.value if isinstance(event.value, str) else None
        )
        if event.value == get_active_memory_id():
            return
        self._switch(str(event.value))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "memory-close":
            self.dismiss()
            return
        if event.button.id == "memory-delete":
            await self._delete_current()


from remie.tui import _agent_app_registry as _registry

_registry.register_module(__name__)
