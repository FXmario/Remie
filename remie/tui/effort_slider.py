"""Discrete reasoning-effort slider with keyboard and pointer controls."""

from rich.text import Text
from textual import events
from textual.reactive import reactive
from textual.widgets import Static

from remie.tui.constants import REASONING_EFFORTS


class EffortSlider(Static, can_focus=True):
    DEFAULT_CSS = """
    EffortSlider {
        height: 3;
        width: 100%;
        padding: 0 1;
    }
    EffortSlider:focus { background: $primary 20%; }
    EffortSlider:disabled { opacity: 50%; }
    """
    ALLOW_SELECT = False
    value = reactive("medium")

    def __init__(self, *, value="medium", id=None):
        super().__init__(id=id)
        self.value = value

    def validate_value(self, value):
        return value if value in REASONING_EFFORTS else "medium"

    def render(self):
        width = max(5, self.content_size.width)
        selected = REASONING_EFFORTS.index(self.value)
        track = Text()
        positions = [round(i * (width - 1) / 4) for i in range(5)]
        for x in range(width):
            track.append("●" if x == positions[selected] else "─",
                         style="bold cyan" if x <= positions[selected] else "dim")
        track.append("\n" + self.value.title() + "  (←/→)")
        return track

    def on_key(self, event: events.Key):
        if self.disabled:
            return
        index = REASONING_EFFORTS.index(self.value)
        if event.key in {"left", "down", "right", "up", "home", "end"}:
            event.stop()
            event.prevent_default()
            if event.key == "home":
                index = 0
            elif event.key == "end":
                index = 4
            else:
                index += -1 if event.key in {"left", "down"} else 1
            self.value = REASONING_EFFORTS[max(0, min(4, index))]

    def _pick(self, event):
        width = max(1, self.content_size.width - 1)
        x = event.x - self.gutter.left
        self.value = REASONING_EFFORTS[max(0, min(4, round(x * 4 / width)))]

    def on_mouse_down(self, event: events.MouseDown):
        if not self.disabled and event.button == 1:
            event.stop()
            self.focus()
            self.capture_mouse()
            self._pick(event)

    def on_mouse_move(self, event: events.MouseMove):
        if not self.disabled and self.app.mouse_captured is self:
            event.stop()
            self._pick(event)

    def on_mouse_up(self, event: events.MouseUp):
        if self.app.mouse_captured is self:
            event.stop()
            self.release_mouse()
