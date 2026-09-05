"""Chat persistence and UI session lifecycle behavior."""

from typing import Any

from remie.prompts import build_system_prompt
from remie.protocol import strip_protocol_lines
from remie.rendering import render_assistant_panel, render_user_message
from remie.storage.chats import create_chat, load_chat, save_chat
from remie.storage.tabs import new_tab
from remie.tokens import estimate_conversation_tokens
from remie.tui.constants import PROMPT_HISTORY_LIMIT
from remie.tui.widgets import ModelBadge, StreamingRichLog


class ChatSessionMixin:
    def _close_dangling_tool_calls(self) -> None:
        """Repair interrupted native tool calls through the core runner."""
        rebuilt, changed = self._agent_runner.close_dangling_tool_calls(
            self.conversation
        )
        if changed:
            self.conversation = rebuilt
            self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)

    def _save_current_chat(self) -> None:
        if self._chat_id:
            # Never persist an unanswered tool call: a resumed chat would then
            # replay a dangling function_call forever (persistent 400).
            self._close_dangling_tool_calls()
            save_chat(
                self._chat_id,
                self.conversation,
                self._transcript,
                token_usage={
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                },
                keep_empty=True,
            )

    def on_unmount(self) -> None:
        """Persist the current chat so a later launch can resume it."""
        for tab_id in list(self._runtimes):
            if tab_id == "__bootstrap__":
                continue
            try:
                self._save_runtime(tab_id)
            except Exception:
                pass

    def _replay_transcript(self) -> None:
        """Replay a loaded chat's visible history into the log."""
        log = self._widget(StreamingRichLog)
        for message in self._transcript:
            role = message.get("role")
            content = message.get("content")
            if role == "tool":
                log.write("[dim]· tool result[/]")
                continue
            if role == "user":
                if isinstance(content, str):
                    if content.startswith("tool_result("):
                        log.write("[dim]· tool result[/]")
                        continue
                    log.write(render_user_message(content))
                elif isinstance(content, list):
                    text = " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    ).strip()
                    has_image = any(
                        isinstance(part, dict) and part.get("type") == "image_url"
                        for part in content
                    )
                    if text:
                        log.write(render_user_message(text))
                    if has_image:
                        log.write("[dim]📷 image attached[/]")
                log.write("")
            elif role == "assistant":
                text = strip_protocol_lines(str(content or "")).strip()
                if text:
                    log.write(render_assistant_panel(text, self._code_theme()))
                    log.write("")

    def _rebuild_prompt_history(self) -> None:
        """Seed prompt recall from the loaded transcript's user messages."""
        history: list[str] = []
        for message in self._transcript:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and not content.startswith("tool_result("):
                history.append(content)
        self._prompt_history = history[-PROMPT_HISTORY_LIMIT:]

    def _restore_chat_token_usage(self, chat: dict[str, Any]) -> None:
        """Load a chat's saved cumulative usage into the badge counters."""
        usage = chat.get("token_usage") or {}
        self._total_input_tokens = int(usage.get("input_tokens") or 0)
        self._total_output_tokens = int(usage.get("output_tokens") or 0)
        badge = self._widget(ModelBadge)
        badge.set_tokens(self._total_input_tokens, self._total_output_tokens)
        # No generation is active for a freshly loaded chat.
        badge.set_live_generated_tokens(None)

    def _reset_conversation_state(self) -> None:
        """Start an empty conversation for a fresh chat."""
        self.conversation = [
            {
                "role": "system",
                "content": build_system_prompt(
                    native_tools=self._native_tool_calling(),
                    tab_context=self._tab_prompt_context(),
                ),
            }
        ]
        self._transcript = []
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        badge = self._widget(ModelBadge)
        badge.set_tokens(0, 0)
        # No generation is active in a fresh chat; drop any stale counter.
        badge.set_live_generated_tokens(None)
        self._prompt_history = []
        self._history_index = None
        self._history_draft = ""

    def action_new_chat(self) -> None:
        """Create and activate a tab without stopping background agents."""
        if not self._agent_running:
            self._save_current_chat()
        chat = create_chat()
        tab = new_tab(chat["id"])
        self._tab_layout["tabs"].append(tab)
        runtime = self._add_runtime(tab["id"], chat)
        self._active_tab_id = tab["id"]
        runtime.conversation = [
            {
                "role": "system",
                "content": build_system_prompt(
                    native_tools=self._native_tool_calling(),
                    tab_context=self._tab_prompt_context(),
                ),
            }
        ]
        runtime.cached_conv_tokens = estimate_conversation_tokens(runtime.conversation)
        self.sub_title = chat["name"]
        self._save_current_chat()
        self._persist_tab_layout()
        self.call_later(self._show_runtime, tab["id"])
        self._refresh_tabs()

    def _load_chat_into_ui(self, chat_id: str) -> bool:
        """Open history in a retained runtime, leaving all workers untouched."""
        tab = next(
            (tab for tab in self._tab_layout["tabs"] if tab["chat_id"] == chat_id),
            None,
        )
        if tab is not None and tab["id"] in self._runtimes:
            return self.switch_tab(tab["id"])
        chat = load_chat(chat_id)
        if chat is None:
            self.notify("Could not load that chat", severity="warning")
            return False
        self._save_current_chat()
        if tab is None:
            tab = new_tab(chat_id)
            self._tab_layout["tabs"].append(tab)
        runtime = self._add_runtime(tab["id"], chat)
        self._active_tab_id = tab["id"]
        runtime.cached_conv_tokens = estimate_conversation_tokens(runtime.conversation)
        self.sub_title = chat.get("name", "")
        self._persist_tab_layout()
        self.call_later(self._show_runtime, tab["id"])
        self._refresh_tabs()
        return True

    async def _confirm_close_tab(self, tab_id: str) -> None:
        answer = await self.push_screen_wait(
            self._make_ask_screen(
                "This tab is still running. Stop its agent and close the tab?",
                ["Stop and close", "Cancel"],
            )
        )
        if answer == "Stop and close":
            runtime = self._runtimes.get(tab_id)
            if runtime:
                runtime.stop_requested = True
                if runtime.agent_task:
                    runtime.agent_task.cancel()
            self._close_tab_now(tab_id)

    def close_tab(self, tab_id: str | None) -> bool:
        """Close an idle tab; ask before cancelling a running one."""
        if not tab_id:
            return False
        runtime = self._runtimes.get(tab_id)
        if runtime and runtime.agent_running:
            self.call_later(self._confirm_close_tab, tab_id)
            return False
        return self._close_tab_now(tab_id)

    def _close_tab_now(self, tab_id: str) -> bool:
        tabs = self._tab_layout["tabs"]
        index = next((i for i, tab in enumerate(tabs) if tab["id"] == tab_id), None)
        if index is None:
            return False
        self._save_runtime(tab_id)
        was_active = tab_id == self._active_tab_id
        tabs.pop(index)
        runtime = self._runtimes.pop(tab_id, None)
        if runtime and runtime.pane:
            runtime.pane.remove()
        if not was_active:
            self._persist_tab_layout()
            self._refresh_tabs()
            return True
        if tabs:
            target = tabs[min(index, len(tabs) - 1)]
            return self.switch_tab(target["id"])
        self._active_tab_id = None
        self.action_new_chat()
        return True

    def switch_tab(self, tab_id: str) -> bool:
        """Change presentation only; the previous tab's worker keeps running."""
        if tab_id == self._active_tab_id:
            return True
        if tab_id not in self._runtimes:
            return False
        self._active_tab_id = tab_id
        self.sub_title = self._tab_title(tab_id)
        self._persist_tab_layout()
        self.call_later(self._show_runtime, tab_id)
        self._refresh_tabs()
        return True
