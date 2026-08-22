"""The Remie Textual application."""

import asyncio
import base64
import io
import json
import os
import time
from typing import Any, ClassVar

import httpx
from PIL import Image as PILImage
from rich.markup import escape
from rich.panel import Panel
from textual import work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.keys import format_key
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog

from remie.agent import (
    LLMRequestError,
    UnsupportedModelError,
    estimate_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens,
    estimate_tokens_from_counts,
    extract_thinking,
    extract_tool_invocations,
    get_config,
    get_connection_error_message,
    get_full_system_prompt,
    get_model_context_limit,
    load_status_animation_enabled,
    render_assistant_panel,
    render_user_message,
    save_status_animation_enabled,
    strip_protocol_lines,
    summarize_messages,
)
from remie.tools import (
    DEFAULT_CHAT_NAME,
    MEMORY_NAME_MAX_CHARS,
    create_chat,
    ensure_active_memory,
    find_chat_by_id,
    get_tool_summary,
    load_chat,
    load_latest_chat,
    rename_chat,
    save_chat,
)
from remie.tui.constants import (
    COMPACTION_CONTEXT_RATIO,
    COMPACTION_KEEP_MESSAGES,
    MAX_AUTO_CONTINUATIONS,
    MAX_EMPTY_RESPONSE_RETRIES,
    PROMPT_HISTORY_LIMIT,
)
from remie.tui.css import CSS
from remie.tui.helpers import (
    _fallback_memory_name,
    _has_tool_call,
    _preview_window,
    _safe_reasoning_markdown,
    _safe_stream_markdown,
    _should_update_stream,
)
from remie.tui.render import _render_diff, _render_tool_result
from remie.tui.screens.ask_user import AskUserScreen
from remie.tui.screens.chats import ChatScreen
from remie.tui.screens.connection import ConnectionScreen
from remie.tui.screens.memory import MemoryScreen
from remie.tui.widgets import (
    InputRow,
    ModelBadge,
    PromptSubmitted,
    PromptTextArea,
    StatusIndicator,
    StreamingRichLog,
    ThinkingIndicator,
)

# Names resolved through the package namespace at call time so tests can
# monkeypatch them on ``remie.tui`` (mirrors the old single-module globals).
import remie.tui as _tui_pkg

class AgentScreen(Screen):
    """Default screen. Overrides ctrl+c copy to confirm the selection copy."""

    def action_copy_text(self) -> None:
        selection = self.get_selected_text()
        if selection is None:
            raise SkipAction()
        self.app.copy_to_clipboard(selection)
        self.app.notify("Copied to clipboard", title="Selection")


class AgentApp(App):
    """Textual TUI for the Remie coding assistant."""

    TITLE = "Remie"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c,super+c", "copy_or_quit", "Copy/Quit"),
        ("ctrl+l", "new_chat", "New chat"),
        ("ctrl+o", "open_memory", "Memories"),
        ("ctrl+r", "open_chats", "Chats"),
        ("ctrl+p", "open_connection", "Connect"),
        ("ctrl+g", "toggle_status_image", "Toggle status image"),
        ("ctrl+t", "toggle_theme", "Toggle theme"),
        ("escape", "stop_agent", "Stop agent"),
    ]

    def get_key_display(self, binding: Binding) -> str:
        """Render keys like `Ctrl+p` in the footer instead of Textual's `^p`."""
        modifiers, key = binding.parse_key()
        key = format_key(key)
        display_mods = [modifier.title() for modifier in modifiers]
        return "+".join([*display_mods, key])

    def get_default_screen(self) -> Screen:
        return AgentScreen(id="_default")

    def __init__(self) -> None:
        super().__init__()
        self.conversation: list[dict[str, Any]] = []
        self._cached_conv_tokens = 0
        self.theme = "ansi-dark"
        self._agent_running = False
        self._agent_task: asyncio.Task[Any] | None = None
        self._stop_requested = False
        self._input_queue: asyncio.Queue[str | list | None] = asyncio.Queue()
        self._pending_image: PILImage.Image | None = None
        self._prompt_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._status_animation_enabled = load_status_animation_enabled()
        self._chat_id: str | None = None
        self._transcript: list[dict[str, Any]] = []
        self.debug_mode = os.environ.get("REMIE_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def compose(self) -> ComposeResult:
        yield Header()
        yield StreamingRichLog(id="log", markup=True, wrap=True)
        yield InputRow(id="input-row")
        yield Footer()

    def _code_theme(self) -> str:
        return "ansi_light" if self.theme == "ansi-light" else "ansi_dark"

    def _set_status(self, status: str) -> None:
        self.query_one(StatusIndicator).set_status(status)
        self.query_one(ThinkingIndicator).set_status(status)

    def on_mount(self) -> None:
        self.sub_title = ""
        prompt = self.query_one("#prompt", PromptTextArea)
        self.query_one(ModelBadge).update_config(get_config())
        self.query_one(StatusIndicator).set_animation_enabled(
            self._status_animation_enabled
        )
        ensure_active_memory()
        chat = load_latest_chat()
        if chat is not None:
            self._chat_id = chat["id"]
            self.conversation = list(chat.get("context_messages") or [])
            self._transcript = list(chat.get("transcript") or [])
            self._refresh_system_prompt()
            self.sub_title = chat.get("name", "")
            log = self.query_one("#log", StreamingRichLog)
            log.write("[dim]Resumed chat:[/] " + escape(chat.get("name", "")))
            self._replay_transcript()
        else:
            chat = create_chat()
            self._chat_id = chat["id"]
            self.conversation = [
                {
                    "role": "system",
                    "content": get_full_system_prompt(
                        native_tools=self._native_tool_calling()
                    ),
                }
            ]
            self._transcript = []
            self.sub_title = chat["name"]
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)
        self._rebuild_prompt_history()
        prompt.focus()
        self._prefetch_model_context()

    def action_toggle_status_image(self) -> None:
        """Toggle and persist the status image visibility."""
        self._status_animation_enabled = not self._status_animation_enabled
        self.query_one(StatusIndicator).set_animation_enabled(
            self._status_animation_enabled
        )
        save_status_animation_enabled(self._status_animation_enabled)
        state = "shown" if self._status_animation_enabled else "hidden"
        self.notify(f"Status image {state}", title="Status image")

    def _native_tool_calling(self) -> bool:
        """Native function calling is used for the Codex and OpenRouter
        providers; other providers rely on the text protocol."""
        return get_config().provider in {"codex", "openrouter"}

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system message from the current prompt (incl. memory) and
        keep the conversation token cache in sync."""
        new_system = {
            "role": "system",
            "content": get_full_system_prompt(
                native_tools=self._native_tool_calling()
            ),
        }
        if self.conversation and self.conversation[0]["role"] == "system":
            self._cached_conv_tokens += estimate_message_tokens(
                new_system
            ) - estimate_message_tokens(self.conversation[0])
            self.conversation[0] = new_system
        else:
            self.conversation.insert(0, new_system)
            self._cached_conv_tokens += estimate_message_tokens(new_system)

    async def _name_current_chat(
        self, user_content: str | list, assistant_content: str
    ) -> None:
        """Name a still-default chat after its first completed task."""
        if not self._chat_id:
            return
        chat = find_chat_by_id(self._chat_id)
        if chat is None or not chat["name"].startswith(DEFAULT_CHAT_NAME):
            return
        title = await _tui_pkg.generate_chat_title(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )
        name = title[:MEMORY_NAME_MAX_CHARS].rstrip() or _fallback_memory_name(
            user_content
        )
        renamed = rename_chat(self._chat_id, name)
        if renamed is not None:
            self.sub_title = renamed["name"]

    def _save_current_chat(self) -> None:
        if self._chat_id:
            save_chat(self._chat_id, self.conversation, self._transcript)

    def on_unmount(self) -> None:
        """Persist the current chat so a later launch can resume it."""
        try:
            self._save_current_chat()
        except Exception:
            pass

    def _replay_transcript(self) -> None:
        """Replay a loaded chat's visible history into the log."""
        log = self.query_one("#log", StreamingRichLog)
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

    @work(exclusive=False)
    async def _prefetch_model_context(self) -> None:
        """Populate the live context-window cache when connected to OpenCode Go,
        so compaction uses the actual model window without opening the picker."""
        config = get_config()
        if config.provider != "opencode-go" or not config.api_key:
            return
        try:
            await _tui_pkg.fetch_opencode_go_models(config.api_key)
        except Exception:
            pass

    def set_pending_image(self, image: PILImage.Image) -> None:
        self._pending_image = image

    def _record_prompt_history(self, text: str) -> None:
        if not text:
            return
        if self._prompt_history and self._prompt_history[-1] == text:
            return
        self._prompt_history.append(text)
        if len(self._prompt_history) > PROMPT_HISTORY_LIMIT:
            del self._prompt_history[:-PROMPT_HISTORY_LIMIT]
        self._history_index = None
        self._history_draft = ""

    def recall_prompt_history(self, direction: int, current_text: str) -> str | None:
        """Recall prompt history; returns the text to show or None to abort."""
        if not self._prompt_history:
            return None
        if self._history_index is None:
            if direction < 0:
                self._history_draft = current_text
                self._history_index = len(self._prompt_history) - 1
                return self._prompt_history[self._history_index]
            return None
        target = self._history_index + direction
        if target < 0:
            self._history_index = 0
            return self._prompt_history[0]
        if target >= len(self._prompt_history):
            self._history_index = None
            return self._history_draft
        self._history_index = target
        return self._prompt_history[target]

    def _image_to_content(self, image: PILImage.Image) -> list[dict[str, Any]]:
        buffer = io.BytesIO()
        image.convert("RGBA").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        user_input = event.text.strip()
        if not user_input:
            return
        if user_input.lower() in {"exit", "quit", "keluar"}:
            self.exit()
            return
        self._record_prompt_history(user_input)
        log = self.query_one("#log", StreamingRichLog)
        content: str | list = user_input
        if self._pending_image is not None:
            content = [
                {"type": "text", "text": user_input},
                self._image_to_content(self._pending_image),
            ]
            self._pending_image = None
            log.write(render_user_message(user_input))
            log.write("[dim]📷 image attached[/]")
        else:
            log.write(render_user_message(user_input))
        log.write("")
        self._input_queue.put_nowait(content)
        if not self._agent_running:
            _ = self.message_worker()

    @work(exclusive=True)
    async def message_worker(self) -> None:
        """Consume queued user messages, processing them one at a time."""
        try:
            while True:
                user_content = await self._input_queue.get()
                self._input_queue.task_done()
                if user_content is None or self._stop_requested:
                    self._drain_queue()
                    break
                self._set_status("working")
                await self.run_agent_turn(user_content)
                if self._stop_requested:
                    self._drain_queue()
                    break
        finally:
            self._stop_requested = False
            try:
                self.query_one("#prompt", PromptTextArea).focus()
            except Exception:
                pass

    def _drain_queue(self) -> None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
                self._input_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def _context_limit(self) -> int | None:
        config = get_config()
        return get_model_context_limit(config.model, config.provider)

    def _push_message(self, role: str, content: Any, extra: dict | None = None) -> None:
        """Append a message to the conversation and keep the running token
        estimate current so compaction checks stay O(1). Non-system messages
        are also appended to the visible transcript."""
        message = {"role": role, "content": content}
        if extra:
            message.update(extra)
        self.conversation.append(message)
        if role != "system":
            self._transcript.append(message)
        self._cached_conv_tokens += estimate_message_tokens(message)

    def _conversation_too_large(self, limit: int | None) -> bool:
        if not limit:
            return False
        tokens = self._cached_conv_tokens or estimate_conversation_tokens(
            self.conversation
        )
        return tokens >= limit * COMPACTION_CONTEXT_RATIO

    async def _compact_conversation(self) -> None:
        """Trim old context when the window is nearly full so long tasks continue.

        The messages being dropped are summarized into a compact "session
        memory" note that stays in the conversation; the terse omitted-note
        fallback is used when the summary call fails or yields nothing.
        """
        if len(self.conversation) <= 2:
            return
        dropped = self.conversation[1:-COMPACTION_KEEP_MESSAGES]
        summary = await summarize_messages(dropped) if dropped else ""
        note = summary or (
            "(Earlier conversation was omitted because the context window was "
            "nearly full. Continue based on the most recent messages below.)"
        )
        tail = self.conversation[1:][-COMPACTION_KEEP_MESSAGES:]
        self.conversation = self.conversation[:1] + [
            {"role": "system", "content": note}
        ] + tail
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)

    def _drain_live_reasoning(self) -> None:
        """Timer callback: render newly-arrived reasoning while the stream is
        otherwise silent.

        Providers append reasoning deltas to reasoning_box without yielding,
        so during a long think the async-for body never runs and the inline
        update path never fires. This drains the box every 100 ms and
        re-renders the Reasoning panel so reasoning streams live.
        """
        state = getattr(self, "_live_stream", None)
        if not state or not state.get("active"):
            return
        box: list[str] = state["reasoning_box"]
        consumed: int = state["consumed"]
        if len(box) <= consumed:
            return
        now = time.monotonic()
        if now - state["last_render"] < 0.1:
            return
        state["last_render"] = now
        new_text = "".join(box[consumed:])
        state["consumed"] = len(box)
        log = state["log"]
        # Reconstruct the full accumulated reasoning from what has been
        # consumed so far plus this batch.
        state.setdefault("text", "")
        state["text"] += new_text
        shown = state["text"]
        if self._stop_requested:
            return
        log.update_stream(
            _safe_reasoning_markdown(_preview_window(shown), self._code_theme()),
            title="Reasoning",
            border_style="dim",
        )

    def _stop_live_stream_timer(self, timer) -> None:
        """Stop the live-reasoning timer and mark its state inactive."""
        timer.stop()
        state = getattr(self, "_live_stream", None)
        if state:
            state["active"] = False

    async def run_agent_turn(self, user_content: str | list) -> None:
        log = self.query_one("#log", StreamingRichLog)
        completed = False
        current_task = asyncio.current_task()
        self._agent_task = current_task
        try:
            self._agent_running = True
            # Keep the system prompt in sync with the active provider (text
            # protocol vs native function calling) before every turn.
            self._refresh_system_prompt()
            native_tool_calling = self._native_tool_calling()
            self._push_message("user", user_content)
            continuations = 0
            empty_retries = 0
            while True:
                if self._stop_requested:
                    log.write("[dim]Stopped by user[/]")
                    break
                if self._conversation_too_large(self._context_limit()):
                    await self._compact_conversation()
                    log.write(
                        "[dim]Context window nearly full — older messages compacted.[/]"
                    )
                full_text = ""
                full_chars = 0
                full_newlines = 0
                log.begin_stream()
                tool_detected = False
                tool_rendered = False
                usage_box: dict[str, int] = {}
                reasoning_box: list[str] = []
                finish_box: dict[str, Any] = {}
                tool_calls_box: list[dict[str, str]] = []
                badge = self.query_one(ModelBadge)
                stream_started = time.monotonic()
                last_preview_update = stream_started
                reasoning_text = ""
                # Shared state for the live-stream timer below. Providers only
                # append reasoning to reasoning_box without yielding, so the
                # async-for body stays silent during a long think; a timer
                # drains it so the Reasoning panel still streams live.
                self._live_stream = {
                    "log": log,
                    "reasoning_box": reasoning_box,
                    "consumed": 0,
                    "last_render": stream_started,
                    "active": True,
                }
                reasoning_timer = self.set_interval(0.1, self._drain_live_reasoning)
                async for delta in _tui_pkg.stream_llm_call(
                    self.conversation,
                    usage_box,
                    reasoning_box,
                    finish_box,
                    tool_calls_box=tool_calls_box,
                ):
                    if self._stop_requested:
                        break
                    full_text += delta
                    # Keep O(1) counters for the token-speed estimate so the
                    # badge does not re-scan the whole accumulation per update.
                    full_chars += len(delta)
                    full_newlines += delta.count("\n")
                    # A complete tool-call line only exists once its trailing
                    # newline has arrived, so only rescan at line boundaries
                    # instead of re-scanning the whole accumulating text on
                    # every token.
                    if not tool_detected:
                        if "\n" in delta or _has_tool_call(delta):
                            tool_detected = _has_tool_call(full_text)
                    now = time.monotonic()
                    # Re-rendering the entire accumulated Markdown (parse +
                    # Pygments + layout) per token is the main CPU cost of
                    # streaming. Throttle the preview with an interval that
                    # grows with the text size, but force an immediate render
                    # when a tool call is first detected so the preview mode
                    # switches without delay.
                    should_render = _should_update_stream(
                        len(full_text), last_preview_update, now
                    ) or (tool_detected and not tool_rendered)
                    if tool_detected:
                        tool_rendered = True
                    if not should_render:
                        continue
                    last_preview_update = now
                    # Consume any reasoning the live-stream timer has not
                    # drained yet (shared counter avoids double rendering).
                    if len(reasoning_box) > self._live_stream["consumed"]:
                        new_reasoning = "".join(
                            reasoning_box[self._live_stream["consumed"]:]
                        )
                        reasoning_text += new_reasoning
                        self._live_stream["consumed"] = len(reasoning_box)
                        self._live_stream["last_render"] = now
                    if tool_detected:
                        shown = reasoning_text or extract_thinking(full_text)
                    elif reasoning_text:
                        shown = reasoning_text
                    else:
                        shown = ""
                    elapsed = now - stream_started
                    if elapsed > 0:
                        badge.set_speed(
                            estimate_tokens_from_counts(full_chars, full_newlines)
                            / elapsed
                        )
                    preview = _preview_window(full_text)
                    if shown:
                        preview_shown = _preview_window(shown)
                        log.update_stream(
                            _safe_reasoning_markdown(preview_shown, self._code_theme()),
                            title="Reasoning",
                            border_style="dim",
                        )
                    elif preview:
                        log.update_stream(
                            _safe_stream_markdown(preview, self._code_theme()),
                        )
                if self._stop_requested:
                    self._stop_live_stream_timer(reasoning_timer)
                    log.replace_stream()
                    log.write("[dim]Stopped by user[/]")
                    break
                self._stop_live_stream_timer(reasoning_timer)
                self.query_one(ModelBadge).set_speed(None)
                reasoning_text = "".join(reasoning_box) or extract_thinking(full_text)
                input_tokens = usage_box.get("prompt_tokens") or self._cached_conv_tokens
                output_tokens = usage_box.get("completion_tokens") or estimate_tokens(
                    full_text
                )
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self.query_one(ModelBadge).set_tokens(
                    self._total_input_tokens, self._total_output_tokens
                )
                # Native function calls (Codex provider) arrive as structured
                # items; other providers parse the text protocol.
                pending_calls: list[dict[str, Any]] = []
                if native_tool_calling:
                    for call in tool_calls_box:
                        try:
                            args = json.loads(call.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        if not isinstance(args, dict):
                            args = {}
                        pending_calls.append(
                            {
                                "id": call.get("id") or "",
                                "name": call.get("name") or "",
                                "args": args,
                            }
                        )
                    tool_invocations = [
                        (call["name"], call["args"]) for call in pending_calls
                    ]
                else:
                    tool_invocations = extract_tool_invocations(full_text)
                    pending_calls = [
                        {"id": None, "name": name, "args": args}
                        for name, args in tool_invocations
                    ]
                content = strip_protocol_lines(full_text).strip()
                import sys as _sys
                print("DBG native:", native_tool_calling, "| full_text:", repr(full_text[:120]), "| invocations:", tool_invocations, file=_sys.stderr)
                if not tool_invocations and not content:
                    # The model produced no usable output (e.g. only reasoning,
                    # or the stream ended prematurely). Don't silently mark the
                    # turn done: retry a bounded number of times.
                    if empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
                        empty_retries += 1
                        log.replace_stream()
                        log.write(
                            "[dim]Agent produced no output — retrying…[/]"
                        )
                        self._push_message(
                            "assistant", reasoning_text or "(no output)"
                        )
                        continue
                    log.replace_stream()
                    if reasoning_text:
                        log.write(
                            Panel(
                                _safe_reasoning_markdown(
                                    reasoning_text, self._code_theme()
                                ),
                                title="Reasoning",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
                    log.write("[bold red]Agent stopped: empty response[/]")
                    break
                if (
                    finish_box.get("truncated")
                    and continuations < MAX_AUTO_CONTINUATIONS
                    and not tool_invocations
                ):
                    self._push_message("assistant", full_text)
                    partial = strip_protocol_lines(full_text).strip()
                    if partial:
                        log.replace_stream(
                            _tui_pkg.render_assistant_panel(
                                partial, self._code_theme()
                            )
                        )
                    else:
                        log.replace_stream()
                    continuations += 1
                    continue
                if not tool_invocations:
                    content = strip_protocol_lines(full_text).strip()
                    renderables = []
                    if reasoning_text:
                        renderables.append(
                            Panel(
                                _safe_reasoning_markdown(
                                    reasoning_text, self._code_theme()
                                ),
                                title="Reasoning",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
                    if content:
                        renderables.append(
                            _tui_pkg.render_assistant_panel(content, self._code_theme())
                        )
                    log.replace_stream(*renderables)
                    self._push_message("assistant", full_text)
                    completed = True
                    await self._name_current_chat(user_content, full_text)
                    self._set_status("done")
                    return
                replacements: list[RenderableType] = []
                if reasoning_text:
                    replacements.append(
                        Panel(
                            _safe_reasoning_markdown(
                                reasoning_text, self._code_theme()
                            ),
                            title="Reasoning",
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )
                for name, args in tool_invocations:
                    if self.debug_mode:
                        tool_line = (
                            f"[bold cyan]Agent {escape(name)}"
                            f"({escape(json.dumps(args))})[/]"
                        )
                    else:
                        tool_line = (
                            "[bold cyan]Agent "
                            f"{escape(get_tool_summary(name))}[/]"
                        )
                    replacements.append(tool_line)
                log.replace_stream(*replacements)
                if native_tool_calling:
                    self._push_message(
                        "assistant",
                        full_text,
                        extra={
                            "tool_calls": [
                                {
                                    "id": call["id"],
                                    "name": call["name"],
                                    "arguments": json.dumps(call["args"]),
                                }
                                for call in pending_calls
                            ]
                        },
                    )
                else:
                    self._push_message("assistant", full_text)
                for call in pending_calls:
                    name = call["name"]
                    args = call["args"]
                    if self._stop_requested:
                        log.write("[dim]Stopped by user[/]")
                        break
                    if name == "ask_user":
                        question = str(args.get("question", ""))
                        options = args.get("options") or []
                        log.write(
                            f"[bold cyan]Agent asking you:[/] {escape(question)}"
                        )
                        answer = await self.push_screen_wait(
                            AskUserScreen(question, options)
                        )
                        if answer is None:
                            result = {"answer": None, "cancelled": True}
                        else:
                            result = {"answer": answer}
                    else:
                        result = await asyncio.to_thread(
                            _tui_pkg.run_tool, name, args
                        )
                    result_json = json.dumps(result, default=str)
                    if isinstance(result, dict) and result.get("diff"):
                        log.write(_tui_pkg._render_diff(result["diff"]))
                    if isinstance(result, dict) and result.get("blocked"):
                        log.write(
                            "[bold red]Blocked command:[/] "
                            f"{escape(result.get('command', ''))} "
                            f"\u2014 {escape(str(result.get('reason', 'unsafe command')))}"
                        )
                    if isinstance(result, dict):
                        result_renderable = _tui_pkg._render_tool_result(
                            name, result, self._code_theme()
                        )
                        if result_renderable is not None:
                            log.write(result_renderable)
                    if self.debug_mode:
                        log.write(
                            f"[bold magenta]tool_result:[/] {escape(result_json)}"
                        )
                    if name == "memory" and isinstance(result, dict) and (
                        result.get("action") in {"add", "clear"}
                    ):
                        self._refresh_system_prompt()
                    call_id = call.get("id")
                    if call_id:
                        # Codex native tool calling: results replay as
                        # function_call_output items.
                        self._push_message(
                            "tool",
                            result_json,
                            extra={"tool_call_id": call_id, "name": name},
                        )
                    else:
                        self._push_message("user", f"tool_result({result_json})")
        except asyncio.CancelledError:
            log.replace_stream()
            log.write("[dim]Stopped by user[/]")
        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as error:
            log.replace_stream()
            message = get_connection_error_message(error)
            if message is not None:
                self.notify(message, title="LLM connection error", severity="error")
        except UnsupportedModelError as error:
            log.replace_stream()
            self.notify(str(error), title="Unsupported model", severity="error")
        except LLMRequestError as error:
            log.replace_stream()
            message = str(error)
            if "context" in message.lower() or "maximum context length" in message.lower():
                self.notify(
                    "The conversation exceeded the model's context window. "
                    "Start a new chat (Ctrl+L).",
                    title="Context window full",
                    severity="error",
                )
            else:
                self.notify(message, title="Request error", severity="error")
        except Exception as error:
            log.replace_stream()
            self.notify(
                f"{type(error).__name__}: {error}",
                title="Agent error",
                severity="error",
            )
        finally:
            self._agent_running = False
            if self._agent_task is current_task:
                self._agent_task = None
            if completed:
                self._save_current_chat()
            else:
                self._set_status("ready")

    def _reset_conversation_state(self) -> None:
        """Start an empty conversation for a fresh chat."""
        self.conversation = [{"role": "system", "content": get_full_system_prompt()}]
        self._transcript = []
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.query_one(ModelBadge).set_tokens(0, 0)
        self._prompt_history = []
        self._history_index = None
        self._history_draft = ""

    def action_new_chat(self) -> None:
        """Start a new chat; the previous one stays in the chat history."""
        if self._agent_running:
            return
        self._save_current_chat()
        self.query_one("#log", RichLog).clear()
        chat = create_chat()
        self._chat_id = chat["id"]
        self.sub_title = chat["name"]
        self._reset_conversation_state()

    def _load_chat_into_ui(self, chat_id: str) -> bool:
        """Switch to another saved chat, keeping the current one on disk."""
        if self._agent_running:
            return False
        chat = load_chat(chat_id)
        if chat is None:
            self.notify("Could not load that chat", severity="warning")
            return False
        self._save_current_chat()
        self.query_one("#log", RichLog).clear()
        self._chat_id = chat["id"]
        self.conversation = list(chat.get("context_messages") or [])
        self._transcript = list(chat.get("transcript") or [])
        self._refresh_system_prompt()
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.query_one(ModelBadge).set_tokens(0, 0)
        self._rebuild_prompt_history()
        self.sub_title = chat.get("name", "")
        log = self.query_one("#log", StreamingRichLog)
        log.write("[dim]Switched to chat:[/] " + escape(chat.get("name", "")))
        self._replay_transcript()
        return True

    def action_copy_or_quit(self) -> None:
        """Copy selected text, or quit when nothing is selected."""
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.notify("Copied to clipboard", title="Selection")
            return
        self.exit()

    def action_toggle_theme(self) -> None:
        self.theme = "ansi-dark" if self.theme == "ansi-light" else "ansi-light"

    def action_stop_agent(self) -> None:
        """Stop the active agent turn and clear pending messages."""
        if self._agent_running:
            self._stop_requested = True
            self._input_queue.put_nowait(None)
            if self._agent_task is not None:
                self._agent_task.cancel()

    async def action_open_connection(self) -> None:
        """Open the connection/model picker. Ignored while the agent is busy."""
        if self._agent_running:
            return
        await self.push_screen(ConnectionScreen())

    async def action_open_memory(self) -> None:
        """Open the memory picker. Ignored while the agent is busy."""
        if self._agent_running:
            return
        await self.push_screen(MemoryScreen())

    async def action_open_chats(self) -> None:
        """Open the chat history picker. Ignored while the agent is busy."""
        if self._agent_running:
            return
        await self.push_screen(ChatScreen())


def run_tui() -> None:
    app = AgentApp()
    app.run()


def main() -> None:
    run_tui()
