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
from rich.console import RenderableType
from rich.markup import escape
from rich.panel import Panel
from textual import events, work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.keys import format_key
from textual.screen import Screen
from textual.widgets import Footer, Header

from remie.agent import (
    fetch_opencode_go_models,
    generate_chat_title,
    get_config,
    get_connection_error_message,
    get_model_context_limit,
    load_status_animation_enabled,
    save_status_animation_enabled,
    stream_llm_call,
    summarize_messages,
)
from remie.core.runner import AgentRunner
from remie.errors import LLMRequestError, UnsupportedModelError
from remie.prompts import build_system_prompt
from remie.protocol import extract_thinking, strip_protocol_lines
from remie.rendering import render_assistant_panel, render_user_message
from remie.tokens import (
    estimate_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens,
)
from remie.storage.chats import (
    DEFAULT_CHAT_NAME,
    create_chat,
    find_chat_by_id,
    load_latest_chat,
    rename_chat,
)
from remie.storage.memories import MEMORY_NAME_MAX_CHARS, ensure_active_memory
from remie.tools.executor import ToolExecutor, execute_tool_call
from remie.tools.registry import get_tool_summary
from remie.tui.chat_session import ChatSessionMixin
from remie.tui.constants import (
    COMPACTION_CONTEXT_RATIO,
    COMPACTION_KEEP_MESSAGES,
    LIVE_REASONING_TICK,
    MAX_AUTO_CONTINUATIONS,
    MAX_EMPTY_RESPONSE_RETRIES,
    PROMPT_HISTORY_LIMIT,
    STREAM_RENDER_COALESCE_WINDOW,
)
from remie.tui.css import CSS
from remie.tui.helpers import (
    _detect_terminal_background,
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
from remie.tui.streaming import StreamingPresentationMixin
from remie.tui.widgets import (
    InputRow,
    ModelBadge,
    PromptSelectionCompleted,
    PromptSubmitted,
    PromptTextArea,
    StatusIndicator,
    StreamingRichLog,
    ThinkingIndicator,
)


class CopyableFooter(Footer):
    """Footer whose rendered key labels participate in screen selections."""

    ALLOW_SELECT = True

    def on_mount(self) -> None:
        # FooterKey disables selection by default so clicks can invoke bindings.
        # Selection is still safe: Textual only emits a click when the pointer
        # is released at the press location, while a drag completes selection.
        for child in self.query("*"):
            child.ALLOW_SELECT = True


class AgentScreen(Screen):
    """Default screen with copy-on-release text selection."""

    def copy_selection(self, selection: str | None) -> bool:
        """Copy a non-empty selection and report whether anything was copied."""
        if selection is None or selection == "":
            return False
        self.app.copy_to_clipboard(selection)
        self.app.notify("Copied to clipboard", title="Selection")
        return True

    def on_text_selected(self, _event: events.TextSelected) -> None:
        """Copy screen-level text and remove its highlight after release."""
        if self.copy_selection(self.get_selected_text()):
            self.clear_selection()

    def on_prompt_selection_completed(self, event: PromptSelectionCompleted) -> None:
        """Copy selections made by TextArea, which owns its selection state."""
        self.copy_selection(event.text)

    def action_copy_text(self) -> None:
        if not self.copy_selection(self.get_selected_text()):
            raise SkipAction()


class AgentApp(ChatSessionMixin, StreamingPresentationMixin, App):
    """Textual TUI for the Remie coding assistant."""

    TITLE = "Remie"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    IS_REMIE_AGENT_APP = True
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
        # Match the terminal's background at startup. OSC 11 detection may be
        # unavailable (non-interactive terminals, unsupported emulators), in
        # which case dark mode is the safe default. Textual's full palettes
        # visibly theme the entire UI rather than only ANSI accents.
        self._system_theme = _detect_terminal_background() or "dark"
        self._theme_mode = "system"
        # ANSI themes inherit the terminal's background, preserving terminal
        # transparency. Explicit light/dark overrides use opaque full palettes.
        self.theme = f"ansi-{self._system_theme}"
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
        self._tool_executor = ToolExecutor(
            ask_user=self._ask_user_for_tool,
            run=execute_tool_call,
        )
        self._agent_runner = AgentRunner(self._tool_executor)

    def compose(self) -> ComposeResult:
        yield Header()
        yield StreamingRichLog(id="log", markup=True, wrap=True)
        yield InputRow(id="input-row")
        yield CopyableFooter()

    def _code_theme(self) -> str:
        theme = self.available_themes.get(self.theme)
        return "ansi_light" if theme is not None and not theme.dark else "ansi_dark"

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
            # A chat saved by an older version (or mid-crash) may contain
            # unanswered tool calls; heal them before anything is replayed.
            self._close_dangling_tool_calls()
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
                    "content": build_system_prompt(
                        native_tools=self._native_tool_calling()
                    ),
                }
            ]
            self._transcript = []
            self.sub_title = chat["name"]
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)
        if "token_usage" in chat:
            self._restore_chat_token_usage(chat)
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
            "content": build_system_prompt(native_tools=self._native_tool_calling()),
        }
        if self.conversation and self.conversation[0]["role"] == "system":
            self._cached_conv_tokens += estimate_message_tokens(
                new_system
            ) - estimate_message_tokens(self.conversation[0])
            self.conversation[0] = new_system
        else:
            self.conversation.insert(0, new_system)
            self._cached_conv_tokens += estimate_message_tokens(new_system)

    async def _update_current_chat_title(self, user_content: str | list) -> None:
        """Adapt an automatically managed title after a completed agent turn.

        The title model sees recent completed conversation plus the current
        title and is instructed to retain it unless the chat's central topic
        changed. A manual rename permanently opts the chat out.
        """
        if not self._chat_id:
            return
        chat = find_chat_by_id(self._chat_id)
        if chat is None or chat.get("title_source") == "manual":
            return
        current_name = str(chat.get("name") or DEFAULT_CHAT_NAME)
        recent_messages = self._transcript[-20:]
        title = await generate_chat_title(
            [
                {
                    "role": "system",
                    "content": f"Current chat title: {current_name}",
                },
                *recent_messages,
            ]
        )
        if title:
            name = title[:MEMORY_NAME_MAX_CHARS].rstrip()
        elif current_name.startswith(DEFAULT_CHAT_NAME):
            name = _fallback_memory_name(user_content)
        else:
            return
        if not name or name.casefold() == current_name.casefold():
            return
        renamed = rename_chat(self._chat_id, name, title_source="auto")
        if renamed is not None:
            self.sub_title = renamed["name"]

    @work(exclusive=False)
    async def _prefetch_model_context(self) -> None:
        """Populate the live context-window cache when connected to OpenCode Go,
        so compaction uses the actual model window without opening the picker."""
        config = get_config()
        if config.provider != "opencode-go" or not config.api_key:
            return
        try:
            await fetch_opencode_go_models(config.api_key)
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
        self.conversation = (
            self.conversation[:1] + [{"role": "system", "content": note}] + tail
        )
        self._cached_conv_tokens = estimate_conversation_tokens(self.conversation)

    async def _ask_user_for_tool(self, question: str, options: list[str]) -> str | None:
        """Textual implementation of the core tool executor's user callback."""
        return await self.push_screen_wait(AskUserScreen(question, options))

    async def run_agent_turn(self, user_content: str | list) -> None:
        log = self.query_one("#log", StreamingRichLog)
        completed = False
        current_task = asyncio.current_task()
        self._agent_task = current_task
        try:
            self._agent_running = True
            # Heal any tool calls left unanswered by a previous interrupt or
            # crash before they are replayed to the backend.
            self._close_dangling_tool_calls()
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
                codex_reasoning_items: list[dict[str, str]] = []
                badge = self.query_one(ModelBadge)
                badge.set_live_generated_tokens(0)
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
                    "text": "",
                    "badge": badge,
                    "started": stream_started,
                    "content_chars": 0,
                    "content_newlines": 0,
                    "reasoning_chars": 0,
                    "reasoning_newlines": 0,
                    "last_badge_update": 0.0,
                    "active": True,
                }
                reasoning_timer = self.set_interval(
                    LIVE_REASONING_TICK, self._drain_live_reasoning
                )
                self._live_reasoning_timer = reasoning_timer
                async for delta in stream_llm_call(
                    self.conversation,
                    usage_box,
                    reasoning_box,
                    finish_box,
                    tool_calls_box=tool_calls_box,
                    reasoning_items_box=(
                        codex_reasoning_items if native_tool_calling else None
                    ),
                ):
                    if self._stop_requested:
                        break
                    full_text += delta
                    # Keep O(1) counters for the token-speed estimate so the
                    # badge does not re-scan the whole accumulation per update.
                    full_chars += len(delta)
                    full_newlines += delta.count("\n")
                    self._live_stream["content_chars"] = full_chars
                    self._live_stream["content_newlines"] = full_newlines
                    # Throttled to ~1 repaint per timer tick so a fast stream
                    # does not relayout the badge row on every delta.
                    self._update_live_generation_badge(self._live_stream)
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
                            reasoning_box[self._live_stream["consumed"] :]
                        )
                        self._live_stream["text"] += new_reasoning
                        self._live_stream["reasoning_chars"] += len(new_reasoning)
                        self._live_stream["reasoning_newlines"] += new_reasoning.count(
                            "\n"
                        )
                        self._live_stream["consumed"] = len(reasoning_box)
                        self._live_stream["last_render"] = now
                        self._update_live_generation_badge(self._live_stream)
                    # The timer and this content-driven path share the same
                    # accumulator. Otherwise a timer-rendered prefix vanishes
                    # as soon as a content delta renders the next suffix.
                    reasoning_text = self._live_stream["text"]
                    # Coalesce: when the timer painted this same panel a few
                    # ms ago and no fresh reasoning arrived with this delta,
                    # skip the duplicate paint (the panel content is already
                    # up to date on screen).
                    if (
                        reasoning_text
                        and len(reasoning_box) <= self._live_stream["consumed"]
                        and now - self._live_stream["last_render"]
                        < STREAM_RENDER_COALESCE_WINDOW
                        and not (tool_detected and not tool_rendered)
                    ):
                        continue
                    if tool_detected:
                        shown = reasoning_text or extract_thinking(full_text)
                    elif reasoning_text:
                        shown = reasoning_text
                    else:
                        shown = ""
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
                badge.set_speed(None)
                reasoning_text = "".join(reasoning_box) or extract_thinking(full_text)
                input_tokens = (
                    usage_box.get("prompt_tokens") or self._cached_conv_tokens
                )
                output_tokens = usage_box.get("completion_tokens") or estimate_tokens(
                    full_text
                )
                # Replace the live estimate with the provider's exact count
                # when available, and keep the final generated count visible.
                badge.set_live_generated_tokens(output_tokens)
                # Keep the throttle window consistent so a continuation
                # iteration doesn't inherit a stale timestamp.
                self._live_stream["last_badge_update"] = time.monotonic()
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self.query_one(ModelBadge).set_tokens(
                    self._total_input_tokens, self._total_output_tokens
                )
                prepared = self._agent_runner.prepare_response(
                    full_text,
                    native_tool_calling=native_tool_calling,
                    native_calls=tool_calls_box,
                )
                pending_calls = prepared.tool_calls
                tool_invocations = prepared.tool_invocations
                content = prepared.content
                if not tool_invocations and not content:
                    # The model produced no usable output (e.g. only reasoning,
                    # or the stream ended prematurely). Don't silently mark the
                    # turn done: retry a bounded number of times.
                    if empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
                        empty_retries += 1
                        log.replace_stream()
                        log.write("[dim]Agent produced no output — retrying…[/]")
                        self._push_message("assistant", reasoning_text or "(no output)")
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
                            render_assistant_panel(partial, self._code_theme())
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
                            render_assistant_panel(content, self._code_theme())
                        )
                    log.replace_stream(*renderables)
                    self._push_message("assistant", full_text)
                    completed = True
                    await self._update_current_chat_title(user_content)
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
                            f"[bold cyan]Agent {escape(get_tool_summary(name))}[/]"
                        )
                    replacements.append(tool_line)
                log.replace_stream(*replacements)
                if native_tool_calling:
                    extra = self._agent_runner.assistant_metadata(
                        prepared, codex_reasoning_items
                    )
                    self._push_message("assistant", full_text, extra=extra)
                else:
                    self._push_message("assistant", full_text)
                for call in pending_calls:
                    name = call.name
                    args = call.args
                    if self._stop_requested:
                        log.write("[dim]Stopped by user[/]")
                        break
                    if name == "ask_user":
                        log.write(
                            "[bold cyan]Agent asking you:[/] "
                            f"{escape(str(args.get('question', '')))}"
                        )
                    result = await self._agent_runner.execute_tool(call)
                    result_json = json.dumps(result, default=str)
                    if isinstance(result, dict) and result.get("diff"):
                        log.write(_render_diff(result["diff"]))
                    if isinstance(result, dict) and result.get("blocked"):
                        log.write(
                            "[bold red]Blocked command:[/] "
                            f"{escape(result.get('command', ''))} "
                            f"\u2014 {escape(str(result.get('reason', 'unsafe command')))}"
                        )
                    if isinstance(result, dict):
                        result_renderable = _render_tool_result(
                            name, result, self._code_theme()
                        )
                        if result_renderable is not None:
                            log.write(result_renderable)
                    if self.debug_mode:
                        log.write(
                            f"[bold magenta]tool_result:[/] {escape(result_json)}"
                        )
                    if (
                        name == "memory"
                        and isinstance(result, dict)
                        and (result.get("action") in {"add", "clear"})
                    ):
                        self._refresh_system_prompt()
                    call_id = call.id
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
            if (
                "context" in message.lower()
                or "maximum context length" in message.lower()
            ):
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
            self._stop_active_reasoning_timer()
            self.query_one(ModelBadge).set_speed(None)
            self._agent_running = False
            if self._agent_task is current_task:
                self._agent_task = None
            if completed:
                self._save_current_chat()
            else:
                self._set_status("ready")

    def action_copy_or_quit(self) -> None:
        """Copy selected text, or quit when nothing is selected."""
        selected = self.screen.get_selected_text()
        if isinstance(self.screen, AgentScreen) and self.screen.copy_selection(
            selected
        ):
            return
        if selected:
            self.copy_to_clipboard(selected)
            self.notify("Copied to clipboard", title="Selection")
            return
        self.exit()

    def action_toggle_theme(self) -> None:
        modes = ("system", "light", "dark")
        self._theme_mode = modes[(modes.index(self._theme_mode) + 1) % len(modes)]
        resolved = (
            self._system_theme if self._theme_mode == "system" else self._theme_mode
        )
        self.theme = (
            f"ansi-{resolved}"
            if self._theme_mode == "system"
            else f"textual-{resolved}"
        )
        label = (
            f"system theme ({resolved})"
            if self._theme_mode == "system"
            else f"{resolved} theme"
        )
        self.notify(f"Switched to {label}", title="Theme")

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


def main() -> None:
    """Launch the terminal application."""
    AgentApp().run()


# Public compatibility alias used by main.py and older integrations.
run_tui = main
