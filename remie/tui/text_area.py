"""TextArea compatibility with Textual's terminal-default ANSI themes."""

from rich.style import Style
from textual.widgets import TextArea


class TerminalTextArea(TextArea):
    """Keep terminal-default backgrounds valid when Textual builds editor styles."""

    @property
    def rich_style(self) -> Style:
        style = super().rich_style
        # TextAreaTheme.apply_css falls back to current_theme.surface when no
        # background is present. ANSI themes use "ansi_default", a Textual
        # color name which Rich cannot parse. Supply Rich's equivalent before
        # that fallback, without changing concrete backgrounds or other themes.
        if style.bgcolor is None and self.app.current_theme.surface == "ansi_default":
            return style + Style(bgcolor="default")
        return style
