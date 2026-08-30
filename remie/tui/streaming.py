"""Reusable live-stream presentation behavior for the Textual frontend."""

import time
from typing import Any

from remie.tokens import estimate_tokens_from_counts
from remie.tui.constants import LIVE_REASONING_TICK
from remie.tui.helpers import (
    _preview_window,
    _safe_reasoning_markdown,
    _stream_update_interval,
)
from remie.tui.widgets import ModelBadge


class StreamingPresentationMixin:
    def _update_live_generation_badge(
        self, state: dict[str, Any], force: bool = False
    ) -> None:
        """Update generated-token count and speed from O(1) stream counters.

        Repaints are throttled to roughly one per timer tick so a fast stream
        emitting hundreds of deltas per second does not relayout the badge row
        on every chunk; pass force=True for the final, exact update.
        """
        now = time.monotonic()
        if not force and now - state.get("last_badge_update", 0.0) < (
            LIVE_REASONING_TICK
        ):
            return
        state["last_badge_update"] = now
        generated_tokens = estimate_tokens_from_counts(
            state["content_chars"] + state["reasoning_chars"],
            state["content_newlines"] + state["reasoning_newlines"],
        )
        badge: ModelBadge = state["badge"]
        elapsed = now - state["started"]
        speed = generated_tokens / elapsed if elapsed > 0 else None
        # Updating both values together avoids two Textual widget repaints for
        # every streaming tick.
        badge.set_live_metrics(generated_tokens, speed)

    def _drain_live_reasoning(self) -> None:
        """Render and count newly-arrived reasoning while content is silent.

        Providers append reasoning deltas to reasoning_box without yielding,
        so during a long think the async-for body never runs and the inline
        update path never fires. This timer drains the box and updates both the
        Reasoning panel and generated-token counter.
        """
        state = getattr(self, "_live_stream", None)
        if not state or not state.get("active"):
            return
        box: list[str] = state["reasoning_box"]
        consumed: int = state["consumed"]
        if len(box) <= consumed:
            return
        now = time.monotonic()
        if now - state["last_render"] < _stream_update_interval(
            len(state["text"]) or 1
        ):
            return
        state["last_render"] = now
        new_text = "".join(box[consumed:])
        state["consumed"] = len(box)
        log = state["log"]
        # Reconstruct the full accumulated reasoning from what has been
        # consumed so far plus this batch.
        state["text"] += new_text
        state["reasoning_chars"] += len(new_text)
        state["reasoning_newlines"] += new_text.count("\n")
        self._update_live_generation_badge(state)
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
        if getattr(self, "_live_reasoning_timer", None) is timer:
            self._live_reasoning_timer = None
        state = getattr(self, "_live_stream", None)
        if state:
            state["active"] = False

    def _stop_active_reasoning_timer(self) -> None:
        """Clean up a timer when a stream exits through an exception/cancel."""
        timer = getattr(self, "_live_reasoning_timer", None)
        if timer is not None:
            self._stop_live_stream_timer(timer)
