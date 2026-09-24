"""Keyboard and image behavior in text-only terminal environments."""

import asyncio

import pytest
from textual.widgets import Button

from remie.tui.app import AgentApp
from remie.tui.helpers import _supports_terminal_graphics
from remie.tui.widgets import ModelBadge, PromptTextArea, StatusIndicator, ThinkingIndicator


@pytest.mark.parametrize(
    ("term", "tmux", "expected"),
    [
        ("foot", "", True),
        ("xterm-256color", "", True),
        ("foot", "/tmp/tmux/default,1,0", False),
        ("linux", "", False),
        ("vt100", "", False),
        ("dumb", "", False),
        ("screen-256color", "", False),
        ("", "", False),
    ],
)
def test_terminal_graphics_capability(monkeypatch, term, tmux, expected):
    monkeypatch.setenv("TERM", term)
    monkeypatch.setenv("TMUX", tmux)
    assert _supports_terminal_graphics() is expected


def test_tty_never_loads_graphics_and_keeps_text_indicator(monkeypatch, tmp_path):
    import remie.agent as agent

    monkeypatch.setenv("TERM", "linux")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(agent, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(agent, "CONFIG_FILE", tmp_path / "config.json")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            indicator = app.query_one(StatusIndicator)
            assert not indicator.display
            assert not indicator.query("#status-gif")
            assert indicator._frames == {}
            app._set_status("working")
            assert app.query_one(ThinkingIndicator).display
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert not indicator.display
            assert indicator._frames == {}
            assert agent.load_status_animation_enabled() is True

    asyncio.run(exercise())


def test_tty_prompt_tab_navigates_to_buttons_and_activates(monkeypatch, tmp_path):
    import remie.agent as agent

    monkeypatch.setenv("TERM", "linux")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(agent, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(agent, "CONFIG_FILE", tmp_path / "config.json")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptTextArea)
            assert app.focused is prompt
            await pilot.press("tab")
            await pilot.pause()
            assert isinstance(app.focused, Button)
            assert prompt.text == ""
            # Shift+Tab moves back to the prompt; a focused button responds
            # to Enter just as it does to a mouse click.
            await pilot.press("shift+tab")
            assert app.focused is prompt
            await pilot.press("tab")
            assert isinstance(app.focused, Button)
            hide = app.query_one("#tab-hide", Button)
            hide.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#tabs-show", Button).display

    asyncio.run(exercise())


def test_tty_model_badge_is_keyboard_accessible(monkeypatch, tmp_path):
    import remie.agent as agent

    monkeypatch.setenv("TERM", "linux")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(agent, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(agent, "CONFIG_FILE", tmp_path / "config.json")

    async def exercise():
        app = AgentApp()
        opened = []

        async def open_connection():
            opened.append(True)

        app.action_open_connection = open_connection
        async with app.run_test() as pilot:
            badge = app.query_one(ModelBadge)
            assert badge.can_focus
            badge.focus()
            await pilot.press("enter")
            assert opened == [True]

    asyncio.run(exercise())
