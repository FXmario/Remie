"""Ctrl+P tabbed launcher reusing the existing management layouts."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Input, TabbedContent, TabPane

from remie.tui.screens.chats import ChatScreen
from remie.tui.screens.connection import ConnectionScreen
from remie.tui.screens.memory import MemoryScreen
from remie.tui.screens.models import ModelScreen


class _TabContent(Widget):
    """Widget host for behavior copied from an existing modal screen."""

    def dismiss(self, result=None) -> None:
        self.screen.dismiss(result)


class _ChatTab(_TabContent):
    def __init__(self) -> None:
        super().__init__(id="open-chat-content")
        ChatScreen._init_state(self)


class _MemoryTab(_TabContent):
    def __init__(self) -> None:
        super().__init__(id="open-memory-content")
        MemoryScreen._init_state(self)


class _ProviderTab(_TabContent):
    def __init__(self) -> None:
        super().__init__(id="open-provider-content")
        ConnectionScreen._init_state(self)


class _ModelTab(_TabContent):
    def __init__(self) -> None:
        super().__init__(id="open-model-content")
        ModelScreen._init_state(self)


def _copy_screen_behavior(target: type[Widget], source: type[ModalScreen]) -> None:
    """Share a modal's compose/event methods without nesting a Screen.

    Textual Screens only render when they are on the app screen stack, so a
    Screen placed directly in a TabPane has a zero-sized compositor region.
    Copying the behavior onto a normal Widget keeps one implementation of each
    layout while allowing it to render inside TabbedContent.
    """
    excluded = {"__dict__", "__doc__", "__init__", "__module__", "__weakref__"}
    for name, value in source.__dict__.items():
        if name not in excluded and name not in {"CSS", "BINDINGS"}:
            setattr(target, name, value)


_copy_screen_behavior(_ChatTab, ChatScreen)
_copy_screen_behavior(_MemoryTab, MemoryScreen)
_copy_screen_behavior(_ProviderTab, ConnectionScreen)
_copy_screen_behavior(_ModelTab, ModelScreen)


_TAB_CSS = (
    ChatScreen.CSS.replace("ChatScreen", "#open-chat-content")
    + MemoryScreen.CSS.replace("MemoryScreen", "#open-memory-content")
    + ConnectionScreen.CSS.replace("ConnectionScreen", "#open-provider-content")
    + ModelScreen.CSS.replace("ModelScreen", "#open-model-content")
)

# A standalone screen needs a centered, bordered dialog. Inside Ctrl+P the
# TabbedContent is already that dialog, so the same content containers should
# fill the pane without drawing a second modal shell.
_FLAT_TAB_CSS = """
#open-chat-content #chat-dialog,
#open-memory-content #memory-dialog,
#open-provider-content #connection-dialog,
#open-model-content #model-dialog {
    width: 100%;
    height: 100%;
    max-width: 100%;
    max-height: 100%;
    border: none;
    background: transparent;
}

#open-chat-content #chat-dialog,
#open-memory-content #memory-dialog,
#open-model-content #model-dialog {
    padding: 1 2;
}

#open-provider-content #connection-dialog {
    padding: 0;
}
"""


class OpenScreen(ModalScreen):
    """Host the existing management layouts in a keyboard-accessible tab set."""

    BINDINGS = [("escape", "dismiss", "Close")]

    CSS = (
        """
        OpenScreen { align: center middle; }
        #open-dialog {
            width: 78;
            height: 38;
            max-width: 96%;
            max-height: 94%;
            border: round $primary;
            background: $surface;
        }
        #open-tabs { width: 100%; height: 100%; }
        #open-tabs ContentSwitcher { height: 1fr; }
        #open-chats,
        #open-memories,
        #open-providers,
        #open-models {
            width: 100%;
            height: 100%;
            padding: 0;
            align: center middle;
        }
        #open-chat-content,
        #open-memory-content,
        #open-provider-content,
        #open-model-content { width: 100%; height: 100%; }
        """
        + _TAB_CSS
        + _FLAT_TAB_CSS
    )

    _TAB_CONTENT = {
        "open-chats": _ChatTab,
        "open-memories": _MemoryTab,
        "open-providers": _ProviderTab,
        "open-models": _ModelTab,
    }

    def __init__(self) -> None:
        super().__init__()
        self._mounted_tabs = {"open-chats"}
        self._ready = False

    def compose(self) -> ComposeResult:
        with Vertical(id="open-dialog"):
            with TabbedContent(initial="open-chats", id="open-tabs"):
                with TabPane("Chats", id="open-chats"):
                    # Only compose the visible tab. The other management
                    # layouts are relatively expensive and are mounted on
                    # first use by the activation handler below.
                    yield _ChatTab()
                yield TabPane("Memories", id="open-memories")
                yield TabPane("Providers", id="open-providers")
                yield TabPane("Models", id="open-models")

    def on_mount(self) -> None:
        self.query_one("#open-tabs", TabbedContent).focus()
        self._ready = True

    async def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Mount tabs on demand and refresh remote data on every activation."""
        if not self._ready:
            return
        pane_id = event.pane.id or ""
        if pane_id not in self._mounted_tabs:
            content_type = self._TAB_CONTENT.get(pane_id)
            if content_type is None:
                return
            self._mounted_tabs.add(pane_id)
            await event.pane.mount(content_type())
        if pane_id == "open-models":
            models = self.query_one("#open-model-content", _ModelTab)
            if models._config.provider != "local":
                models.run_worker(models._load_live_models(), exclusive=False)
        elif pane_id == "open-providers":
            providers = self.query_one("#open-provider-content", _ProviderTab)
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
                api_key = providers.query_one("#api-key-input", Input).value.strip()
                if api_key:
                    providers.run_worker(
                        providers._refresh_models(api_key, provider), exclusive=False
                    )
