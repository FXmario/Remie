"""Reusable widgets for the Remie TUI."""

# import os
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageGrab
from PIL import ImageSequence
from rich.markup import escape
from rich.panel import Panel
from rich.segment import Segment
from textual.strip import Strip
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Label, RichLog, TextArea
from textual_image.widget import SixelImage as TerminalImage

from remie.agent import (
    OPENCODE_GO_BASE_URL,
    get_config,
    get_model_info,
)

# from remie.agent import strip_protocol_lines
from remie.tui.contracts import is_agent_app
from remie.tui.helpers import _format_tokens, _is_tmux


class StreamingRichLog(RichLog):
    """A RichLog that can stream text in place at the bottom of the log."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_start: int | None = None

    def get_selection(self, selection) -> tuple[str, str] | None:
        """Extract selected text from the rendered log lines."""
        text = "\n".join(strip.text for strip in self.lines)
        return selection.extract(text), "\n"

    def render_line(self, y: int) -> Strip:
        """Render a line with source offsets required for precise selection."""
        scroll_x, scroll_y = self.scroll_offset
        content_y = scroll_y + y
        line = self._render_line(
            content_y, scroll_x, self.scrollable_content_region.width
        )
        strip = line.apply_style(self.rich_style)
        selection = self.text_selection
        if (
            selection is not None
            and (span := selection.get_span(content_y)) is not None
        ):
            start, end = span
            visible_start = max(0, min(strip.cell_length, start - scroll_x))
            visible_end = (
                strip.cell_length
                if end == -1
                else max(0, min(strip.cell_length, end - scroll_x))
            )
            if visible_start < visible_end:
                selection_style = self.screen.get_component_rich_style(
                    "screen--selection"
                )
                selected = strip.crop(visible_start, visible_end)
                # Selection must be a post-style so it wins over Rich markup
                # (panels and syntax spans often set explicit foreground and
                # background colors of their own).
                highlighted = Strip(
                    Segment.apply_style(selected, post_style=selection_style),
                    selected.cell_length,
                )
                strip = Strip.join(
                    (
                        strip.crop(0, visible_start),
                        highlighted,
                        strip.crop(visible_end),
                    )
                )
        # RichLog doesn't attach Textual's source-offset metadata to rendered
        # segments. Without it, a drag inside the log is interpreted as a
        # whole-widget selection and get_selected_text() returns the transcript.
        return strip.apply_offsets(scroll_x, content_y)

    def begin_stream(self) -> None:
        self._stream_start = len(self.lines)

    def update_stream(
        self,
        content: str,
        *,
        title: str = "Assistant",
        border_style: str = "yellow",
    ) -> None:
        if self._stream_start is None:
            self.begin_stream()
        inner = (
            content
            if not isinstance(content, str)
            else Text.from_markup(escape(content))
        )
        renderable = Panel(
            inner,
            title=title,
            border_style=border_style,
            padding=(0, 1),
        )
        console = self.app.console
        width = max(self.scrollable_content_region.width, 1)
        segments = console.render(renderable, console.options.update_width(width))
        lines = list(Segment.split_lines(segments))
        strips = Strip.from_lines(lines)
        for strip in strips:
            strip.adjust_cell_length(width)
        del self.lines[self._stream_start :]
        self.lines.extend(strips)
        self._widest_line_width = max(
            self._widest_line_width,
            max(
                (sum(segment.cell_length for segment in strip) for strip in strips),
                default=0,
            ),
        )
        self._line_cache.clear()
        self.virtual_size = Size(self._widest_line_width, len(self.lines))
        self.scroll_end(animate=False, immediate=False, x_axis=False)
        self.refresh()

    def end_stream(self) -> None:
        self._stream_start = None

    def replace_stream(self, *renderables: object) -> None:
        if self._stream_start is not None:
            del self.lines[self._stream_start :]
            self._stream_start = None
        for renderable in renderables:
            self.write(renderable)
        self._line_cache.clear()
        self.refresh()


def _load_status_gif(name: str) -> tuple[list[PILImage.Image], list[float]]:
    """Load a status GIF, returning empty data when no usable asset exists.

    Installations are not required to include the optional animation assets.
    Also treat unreadable or malformed images as unavailable so they can never
    prevent the rest of the TUI from starting.
    """
    candidates = (
        # Bundled with the package: remie/assets/, resolved from
        # remie/tui/widgets.py via parent.parent (see package-data in
        # pyproject.toml). Also covers running from a source checkout.
        Path(__file__).resolve().parent.parent / "assets" / name,
        # Dev fallback when launching from the repo root.
        Path.cwd() / "assets" / name,
    )
    asset = next((path for path in candidates if path.is_file()), None)
    if asset is None:
        return [], []

    try:
        with PILImage.open(asset) as image:
            frames = []
            durations = []
            for frame in ImageSequence.Iterator(image):
                frames.append(frame.convert("RGBA").copy())
                durations.append(max(0.1, frame.info.get("duration", 100) / 1000))
    except Exception:
        # Any decode failure (corrupt file, PIL limits, unexpected errors)
        # degrades to "no animation" instead of preventing the TUI launch.
        return [], []
    return frames, durations


class StatusIndicator(Vertical):
    """Animated Sixel/unicode status indicator with a text fallback."""

    STATUSES = ("ready", "working", "done")

    def __init__(self) -> None:
        super().__init__(id="status")
        self._frames: dict[str, tuple[list[PILImage.Image], list[float]]] = {}
        self._state = "ready"
        self._frame_index = 0
        self._timer = None
        self._animation_enabled = True

    def _ensure_loaded(self, status: str) -> tuple[list[PILImage.Image], list[float]]:
        """Return the frames/durations for a status, loading them on first use.

        GIFs are loaded lazily per state so startup only decodes the initial
        "ready" frames instead of all three animations up front.
        """
        if status not in self._frames:
            try:
                self._frames[status] = _load_status_gif(f"{status}.gif")
            except Exception:
                # Asset loading must never take the TUI down (missing or
                # unreadable GIFs simply mean no status image is shown).
                self._frames[status] = ([], [])
        return self._frames[status]

    def compose(self):
        frames, _ = self._ensure_loaded(self._state)
        if frames:
            yield TerminalImage(frames[0], id="status-gif")
        else:
            # An empty optional asset set must not reserve a blank status box.
            self.display = False

    def on_mount(self) -> None:
        if self._animation_enabled and self.display and not _is_tmux():
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        _, durations = self._ensure_loaded(self._state)
        if not durations:
            self.display = False
            return
        self._timer = self.set_timer(durations[self._frame_index], self._advance)

    def _advance(self) -> None:
        if not self._animation_enabled:
            return
        frames, _ = self._ensure_loaded(self._state)
        if not frames:
            self.display = False
            return
        self._frame_index = (self._frame_index + 1) % len(frames)
        try:
            self.query_one("#status-gif", TerminalImage).image = frames[
                self._frame_index
            ]
        except NoMatches:
            # The initial GIF was unavailable, so compose did not create an
            # image widget. Status changes should remain harmless.
            self.display = False
            return
        if self._animation_enabled and not _is_tmux():
            self._schedule_next_frame()

    def set_animation_enabled(self, enabled: bool) -> None:
        """Show or hide the GIF and pause its timer without changing status."""
        self._animation_enabled = enabled
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        frames, _ = self._ensure_loaded(self._state)
        self.display = enabled and bool(frames)
        if self.is_attached:
            # Sixel pixels are drawn outside Textual's normal cell buffer.
            # Force the surrounding row to repaint when hiding/showing so tmux
            # clears stale image pixels instead of leaving a fragment behind.
            self.parent.refresh(layout=True)
            self.screen.refresh(layout=True)
        if self.display and self.is_attached and not _is_tmux():
            self._schedule_next_frame()

    def set_status(self, status: str) -> None:
        if status not in self.STATUSES:
            raise ValueError(f"Unknown status: {status}")
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._state = status
        self._frame_index = 0
        frames, _ = self._ensure_loaded(status)
        if not frames:
            self.display = False
            return
        try:
            self.query_one("#status-gif", TerminalImage).image = frames[0]
        except NoMatches:
            self.display = False
            return
        self.display = self._animation_enabled
        if self._animation_enabled and not _is_tmux():
            self._schedule_next_frame()


class ThinkingIndicator(Label):
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        super().__init__(id="tmux-spinner")
        self._frame_index = 0
        self._working = False

    def on_mount(self) -> None:
        if _is_tmux():
            self.set_interval(0.1, self._advance)

    def _advance(self) -> None:
        if self._working:
            self._frame_index = (self._frame_index + 1) % len(self.FRAMES)
            self.update(f"{self.FRAMES[self._frame_index]}")

    def set_status(self, status: str) -> None:
        self._working = status == "working"
        self.display = self._working and _is_tmux()
        if self._working:
            self._frame_index = 0
            self.update(self.FRAMES[0])


class ModelBadge(Label):
    """Clickable label showing the model and generated-output token usage."""

    def __init__(self) -> None:
        super().__init__(id="model-badge")
        self._model_text = ""
        self._vendor_text = ""
        self._reasoning_effort = "off"
        self._input_tokens = 0
        self._output_tokens = 0
        # Estimated tokens generated by the response currently streaming.
        # ``None`` means no generation is active; zero is intentionally visible
        # at the beginning of a request so the user sees the counter start.
        self._live_generated_tokens: int | None = None
        self._speed: float | None = None
        self.update_config(get_config())

    def update_config(self, config) -> None:
        provider = config.provider
        if provider == "local" and config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL:
            provider = "opencode-go"
        vendor = {
            "opencode-go": "OpenCode Go",
            "codex": "Codex (ChatGPT)",
            "openrouter": "OpenRouter",
        }.get(provider, "Local")
        self._model_text = get_model_info(config.model).resolved_display()
        self._vendor_text = vendor
        self._reasoning_effort = config.reasoning_effort
        self._show(self._input_tokens, self._output_tokens)

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._show(input_tokens, output_tokens)

    def set_live_generated_tokens(self, tokens: int | None) -> None:
        """Show the number of output tokens generated by the active response."""
        self._live_generated_tokens = tokens
        self._show(self._input_tokens, self._output_tokens)

    def set_speed(self, speed: float | None) -> None:
        self._speed = speed
        self._show(self._input_tokens, self._output_tokens)

    def _show(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        text = (
            f"{self._model_text}  {self._vendor_text}"
            if self._model_text
            else self._vendor_text
        )
        if self._reasoning_effort != "off":
            text += f" · effort {self._reasoning_effort}"
        if input_tokens or output_tokens:
            total = input_tokens + output_tokens
            text += f" · {_format_tokens(total)} tok"
        if self._live_generated_tokens is not None:
            text += f" · {_format_tokens(self._live_generated_tokens)} generated"
        if self._speed is not None:
            text += f" · {self._speed:.1f} tok/s"
        self.update(text)

    async def on_click(self) -> None:
        app = self.app
        if is_agent_app(app):
            await app.action_open_connection()


class PromptSubmitted(Message):
    """Posted when the user submits the prompt box."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PromptSelectionCompleted(Message):
    """Posted after a mouse selection is completed in the prompt editor."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PromptTextArea(TextArea):
    """Multiline prompt: Enter submits, Shift+Enter / Ctrl+J insert newlines."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            id="prompt",
            placeholder="Type a prompt here...",
            soft_wrap=True,
            show_line_numbers=False,
            **kwargs,
        )

    def on_mouse_up(self) -> None:
        """Report and then visually clear a completed prompt selection."""
        if selected_text := self.selected_text:
            selection_end = self.selection.end
            self.post_message(PromptSelectionCompleted(selected_text))
            self.move_cursor(selection_end, select=False)

    async def on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self._submit()
        elif event.key in {"shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif event.key == "ctrl+v":
            if self._paste_clipboard_image():
                event.stop()
                event.prevent_default()
        elif event.key in {"up", "down"}:
            if self._maybe_history_navigate(event.key):
                event.stop()
                event.prevent_default()

    def _maybe_history_navigate(self, key: str) -> bool:
        """Navigate prompt history when at a line boundary, like a shell."""
        app = self.app
        if not is_agent_app(app):
            return False
        at_boundary = (
            self.cursor_at_first_line if key == "up" else self.cursor_at_last_line
        )
        if not at_boundary:
            return False
        direction = -1 if key == "up" else 1
        text = app.recall_prompt_history(direction, self.text)
        if text is None:
            return False
        self.load_text(text)
        self.move_cursor(self.document.end, center=False)
        return True

    def _submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.post_message(PromptSubmitted(text))
        self.load_text("")

    def _paste_clipboard_image(self) -> bool:
        """Attach an image from the clipboard; return True if one was attached."""
        try:
            grabbed = ImageGrab.grabclipboard()
        except Exception:
            grabbed = None
        if not isinstance(grabbed, PILImage.Image):
            return False
        app = self.app
        if is_agent_app(app):
            app.set_pending_image(grabbed)
            app.notify("Image attached — press Enter to send", title="Clipboard")
        return True


class InputRow(Horizontal):
    def compose(self):
        yield StatusIndicator()
        yield PromptBox()


class ModelRow(Horizontal):
    """Top-right row holding the TMUX spinner next to the model badge."""

    def compose(self):
        yield ThinkingIndicator()
        yield ModelBadge()


class PromptBox(Vertical):
    def __init__(self) -> None:
        super().__init__(id="prompt-box")

    def compose(self):
        yield ModelRow(id="model-row")
        yield PromptTextArea()
