"""Ctrl+P tabbed launcher reusing the existing management screens."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import TabbedContent, TabPane

from remie.tui.screens.chats import ChatScreen
from remie.tui.screens.connection import ConnectionScreen
from remie.tui.screens.memory import MemoryScreen
from remie.tui.screens.models import ModelScreen


class OpenScreen(ModalScreen):
    """Host the existing management modals in a keyboard-accessible tab set."""

    BINDINGS = [("escape", "dismiss", "Close")]

    CSS = """
    OpenScreen { align: center middle; }
    #open-dialog {
        width: 78;
        height: 34;
        max-width: 96%;
        max-height: 94%;
        border: round $primary;
        background: $surface;
    }
    #open-tabs { width: 100%; height: 100%; }
    #open-tabs ContentSwitcher { height: 1fr; }
    #open-tabs TabPane { padding: 0; align: center middle; }
    #open-tabs ChatScreen,
    #open-tabs MemoryScreen,
    #open-tabs ConnectionScreen,
    #open-tabs ModelScreen { width: 100%; height: 100%; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._loaded_tabs: set[str] = set()
        self._ready = False

    def compose(self) -> ComposeResult:
        with Vertical(id="open-dialog"):
            with TabbedContent(initial="open-chats", id="open-tabs"):
                with TabPane("Chats", id="open-chats"):
                    yield ChatScreen()
                with TabPane("Memories", id="open-memories"):
                    yield MemoryScreen()
                with TabPane("Providers", id="open-providers"):
                    yield ConnectionScreen()
                with TabPane("Models", id="open-models"):
                    yield ModelScreen()

    def on_mount(self) -> None:
        self.call_after_refresh(self._activate_chats)

    def _activate_chats(self) -> None:
        self.query_one("#open-tabs", TabbedContent).active = "open-chats"
        self._ready = True

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Fetch remote catalogs only when their copied tab is first opened."""
        if not self._ready:
            return
        pane_id = event.pane.id or ""
        if pane_id in self._loaded_tabs:
            return
        self._loaded_tabs.add(pane_id)
        if pane_id == "open-models":
            models = self.query_one(ModelScreen)
            if models._config.provider != "local":
                models.run_worker(models._load_live_models(), exclusive=False)
        elif pane_id == "open-providers":
            providers = self.query_one(ConnectionScreen)
            provider = providers._active_provider
            if provider == "codex":
                providers.run_worker(
                    providers._prefetch_codex_models(), exclusive=False
                )
            elif provider == "openrouter":
                providers.run_worker(
                    providers._prefetch_openrouter_models(), exclusive=False
                )
            elif provider == "opencode-go":
                api_key = providers.query_one("#api-key-input").value.strip()
                if api_key:
                    providers.run_worker(
                        providers._refresh_models(api_key, provider), exclusive=False
                    )
