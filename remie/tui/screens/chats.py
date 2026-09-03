"""Modal to switch between, create, or delete saved chats."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, RichLog
from textual.widgets.option_list import Option

from remie.tokens import estimate_conversation_tokens
from remie.storage.chats import (
    DEFAULT_CHAT_NAME,
    create_chat,
    delete_chat,
    list_chats,
    load_latest_chat,
)
from remie.tui.contracts import is_agent_app


class ChatScreen(ModalScreen):
    """Modal to switch between, create, or delete saved chats."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    CSS = """
    ChatScreen {
        align: center middle;
    }

    #chat-dialog {
        width: 60;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #chat-list {
        height: auto;
        max-height: 14;
        margin-bottom: 1;
    }

    #chat-dialog Label {
        margin-top: 1;
    }

    #chat-dialog Button {
        margin-right: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_state()

    def _init_state(self) -> None:
        self._chats: list[dict] = []
        self._delete_armed = False

    def _refresh_chats(self) -> None:
        self._chats = list_chats()

    def compose(self) -> ComposeResult:
        self._refresh_chats()
        with Vertical(id="chat-dialog"):
            yield Label("Chats", id="dialog-title")
            yield Input(
                placeholder="Filter chats…",
                id="chat-search",
            )
            yield OptionList(
                *[
                    Option(chat["name"] or DEFAULT_CHAT_NAME, id=chat["id"])
                    for chat in self._chats
                ],
                id="chat-list",
            )
            with Horizontal(classes="row"):
                yield Button("New", variant="primary", id="chat-new")
                yield Button("Switch", variant="primary", id="chat-switch")
                yield Button("Delete", variant="error", id="chat-delete")
                yield Button("Cancel", id="chat-cancel")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "chat-search":
            event.stop()
            self._apply_chat_filter(event.value or "")

    def _apply_chat_filter(self, query: str) -> None:
        """Narrow the chat list to chats whose name matches the query."""
        chat_list = self.query_one("#chat-list", OptionList)
        app = self.app
        current_id = app._chat_id if is_agent_app(app) else None
        q = query.strip().lower()
        if q:
            visible = [
                chat
                for chat in self._chats
                if q in (chat["name"] or DEFAULT_CHAT_NAME).lower()
            ]
        else:
            visible = self._chats
        chat_list.set_options(
            [
                Option(chat["name"] or DEFAULT_CHAT_NAME, id=chat["id"])
                for chat in visible
            ]
        )
        highlight = current_id or (visible[0]["id"] if visible else None)
        if highlight is not None:
            try:
                index = chat_list.get_option_index(highlight)
            except Exception:
                # Highlighted chat is filtered out; leave the list as-is.
                return
            chat_list.highlighted = index

    def on_mount(self) -> None:
        chat_list = self.query_one("#chat-list", OptionList)
        app = self.app
        current_id = app._chat_id if is_agent_app(app) else None
        if self._chats:
            highlight = current_id or self._chats[0]["id"]
            index = chat_list.get_option_index(highlight)
            if index is not None:
                chat_list.highlighted = index
        if self.parent is None or self.parent.__class__.__name__ != "TabPane":
            chat_list.focus()

    def _selected_id(self) -> str | None:
        option = self.query_one("#chat-list", OptionList).highlighted_option
        return option.id if option is not None else None

    def _disarm_delete(self) -> None:
        self._delete_armed = False
        self.query_one("#chat-delete", Button).label = "Delete"

    def _switch(self, chat_id: str | None) -> None:
        if not chat_id:
            self.notify("Pick a chat to switch to", severity="warning")
            return
        app = self.app
        if is_agent_app(app) and app._chat_id == chat_id:
            self.dismiss()
            return
        if is_agent_app(app) and app._load_chat_into_ui(chat_id):
            app.notify("Chat loaded", title="Chats")
        self.dismiss()

    def _reload_options(self) -> None:
        self._refresh_chats()
        self._disarm_delete()
        self._apply_chat_filter(self.query_one("#chat-search", Input).value or "")

    def _new_chat(self) -> None:
        app = self.app
        if is_agent_app(app):
            app.action_new_chat()
            app.notify("Started a new chat", title="Chats")
        self.dismiss()

    def _delete_current(self) -> None:
        chat_id = self._selected_id()
        if not chat_id:
            self.notify("Select a chat to delete", severity="warning")
            return
        if not self._delete_armed:
            self._delete_armed = True
            self.query_one("#chat-delete", Button).label = "Really delete?"
            return
        chat = delete_chat(chat_id)
        app = self.app
        if chat is None:
            self.notify("Unknown chat", severity="warning")
            self._disarm_delete()
            return
        if is_agent_app(app):
            removed_tabs = [
                tab for tab in app._tab_layout["tabs"] if tab["chat_id"] == chat_id
            ]
            app._tab_layout["tabs"] = [
                tab for tab in app._tab_layout["tabs"] if tab["chat_id"] != chat_id
            ]
            if app._chat_id == chat_id:
                latest = load_latest_chat()
                if latest is not None:
                    app._load_chat_into_ui(latest["id"])
                else:
                    app._active_tab_id = None
                    app.action_new_chat()
            elif removed_tabs:
                app._persist_tab_layout()
                app._refresh_tabs()
            app.notify(f"Deleted chat '{chat['name']}'", title="Chats")
        self._reload_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "chat-list" and event.option_id is not None:
            self._switch(event.option_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "chat-cancel":
            self.dismiss()
            return
        if button_id == "chat-new":
            self._new_chat()
        elif button_id == "chat-switch":
            self._switch(self._selected_id())
        elif button_id == "chat-delete":
            self._delete_current()
