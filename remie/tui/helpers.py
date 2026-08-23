"""Small stateless helpers for the Remie TUI."""

import os
import re
import select
import sys
import time

from rich.markup import escape
from rich.markdown import Markdown
from rich.text import Text
from rich.console import RenderableType

from remie.tui.constants import (
    STREAM_PREVIEW_MAX_CHARS,
    STREAM_UPDATE_CHARS_PER_SECOND,
    STREAM_UPDATE_LARGE_PREVIEW_CHARS,
    STREAM_UPDATE_MIN_INTERVAL,
    STREAM_UPDATE_MIN_INTERVAL_LARGE,
)
from remie.model_names import ModelInfo, prettify_model_id
from remie.tools import MEMORY_NAME_MAX_CHARS


def _detect_terminal_background() -> str | None:
    """
    Query the terminal background color via OSC 11 and return 'light' or 'dark'.

    Returns None when the terminal does not respond or is not interactive.
    The active query is skipped inside tmux: tmux may deliver the reply after
    our timeout, causing the raw ``rgb:…`` control response to leak into the
    shell after Remie starts.
    """
    if os.environ.get("TMUX"):
        return None
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    sys.stdout.write("\x1b]11;?\x1b\\")
    sys.stdout.flush()
    response = b""
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 0.02)
        if ready:
            try:
                data = os.read(sys.stdin.fileno(), 1024)
            except OSError, ValueError:
                break
            if not data:
                break
            response += data
            if b"\x1b\\" in response:
                break
    match = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", response)
    if not match:
        return None
    channels = []
    for component in match.groups():
        value = int(component[:2], 16)
        channels.append(value / 255.0)
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "light" if luminance >= 0.5 else "dark"


def _is_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def _safe_stream_markdown(
    text: str, code_theme: str, style: str = ""
) -> RenderableType:
    """Render partial Markdown safely during streaming.

    Auto-closes incomplete code fences so Pygments highlighting engages
    as soon as the opening fence and language arrive.  Falls back to
    plain escaped text if the Markdown parser rejects the content.

    When the text contains no code fence, the full Markdown + Pygments path is
    skipped entirely (it is the dominant cost of streaming re-renders) and the
    text is shown as plain styled text instead.
    """
    fence_count = text.count("\n```") + (1 if text.startswith("```") else 0)
    if fence_count == 0:
        return Text.from_markup(escape(text), style=style or None)
    if fence_count % 2 == 1:
        text = text + "\n```"
    try:
        return Markdown(
            text, code_theme=code_theme, hyperlinks=True, style=style or "none"
        )
    except Exception:
        return Text.from_markup(escape(text), style=style or None)


def _safe_reasoning_markdown(text: str, code_theme: str) -> RenderableType:
    """Render reasoning content gray so it reads as secondary text."""
    return _safe_stream_markdown(text, code_theme, style="grey62")


def _has_tool_call(text: str) -> bool:
    """Return True if any complete line is a tool invocation (or DSML markup)."""
    if "<|DSML|>" in text and "invoke name=" in text:
        return True
    return any(
        line.strip().startswith(("tool:", "<tool:"))
        for line in text.splitlines()
        if line.strip()
    )


def _stream_update_interval(text_len: int) -> float:
    """Minimum seconds between streaming preview re-renders for a given size.

    The length is clamped to the preview window because that bounds the actual
    render cost: `_preview_window` never renders more than
    `STREAM_PREVIEW_MAX_CHARS`, so the throttle interval must not keep growing
    with the (unbounded) accumulated text.

    Short previews render in well under a millisecond, so they use a ~30 fps
    floor for a smooth feel; large previews fall back to the slower floor to
    keep render work bounded.
    """
    floor = (
        STREAM_UPDATE_MIN_INTERVAL_LARGE
        if text_len > STREAM_UPDATE_LARGE_PREVIEW_CHARS
        else STREAM_UPDATE_MIN_INTERVAL
    )
    return max(
        floor,
        min(text_len, STREAM_PREVIEW_MAX_CHARS) / STREAM_UPDATE_CHARS_PER_SECOND,
    )


def _should_update_stream(accumulated_len: int, last_update: float, now: float) -> bool:
    """Whether the streaming preview should re-render now (throttled)."""
    return now - last_update >= _stream_update_interval(accumulated_len)


def _preview_window(text: str, limit: int = STREAM_PREVIEW_MAX_CHARS) -> str:
    """Return a bounded tail window of text for the live preview.

    Keeps per-update rendering cost roughly constant regardless of how long
    the generated answer gets. The cut is moved forward to the next newline so
    the preview never starts mid-line; empty and short inputs are unchanged.
    """
    if len(text) <= limit:
        return text
    start = text.rfind("\n", 0, len(text) - limit)
    if start == -1:
        start = len(text) - limit
    else:
        start += 1
    return text[start:]


def _format_tokens(count: int) -> str:
    """Format a token count compactly, e.g. 1234 -> '1.2k'."""
    if count >= 1000:
        value = count / 1000.0
        if value == int(value):
            return f"{int(value)}k"
        return f"{value:.1f}k"
    return str(count)


def _coerce_model_info(model: "str | ModelInfo") -> ModelInfo:
    """Accept raw ids or ModelInfo rows; ids get heuristic display names."""
    if isinstance(model, ModelInfo):
        return model
    return prettify_model_id(str(model))


def _model_option(model: "str | ModelInfo") -> tuple[Text, str]:
    """Build a dropdown option: pretty label + raw id as the stored value."""
    info = _coerce_model_info(model)
    label = Text(info.resolved_display())
    if info.vendor:
        label.append(f"  {info.vendor}", style="dim")
    if info.free:
        label.append("  Free", style="green")
    return label, info.id


def _fallback_memory_name(task: str | list) -> str:
    """Create a readable fallback title when model title generation fails."""
    if isinstance(task, str):
        text = task
    else:
        text = " ".join(
            part.get("text", "")
            for part in task
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    text = " ".join(text.split())
    if not text:
        return "general"
    cut = text[:MEMORY_NAME_MAX_CHARS]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip()
