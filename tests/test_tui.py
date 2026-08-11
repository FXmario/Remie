import asyncio

import httpx

import chaldea.tui as tui
from chaldea.tui import (
    AgentApp,
    StatusIndicator,
    ThinkingIndicator,
    _is_tmux,
    _load_status_gif,
)


def test_tmux_detection(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    assert not _is_tmux()

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    assert _is_tmux()


def test_tmux_thinking_indicator_visibility(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    indicator = ThinkingIndicator()

    indicator.set_status("working")
    assert indicator._working
    assert indicator.display

    indicator.set_status("ready")
    assert not indicator.display


def test_status_gifs_load_with_frame_timing():
    frames, durations = _load_status_gif("ready.gif")

    assert len(frames) > 1
    assert len(frames) == len(durations)
    assert all(frame.mode == "RGBA" for frame in frames)
    assert all(duration > 0 for duration in durations)


def test_status_indicator_starts_ready():
    indicator = StatusIndicator()

    assert indicator._state == "ready"
    assert len(indicator._frames["working"][0]) > 1


def test_connection_error_shows_toast_and_keeps_app_running(monkeypatch):
    async def exercise():
        async def failed_stream(_conversation):
            raise httpx.ConnectError("connection refused")
            yield ""

        monkeypatch.setattr(tui, "stream_llm_call", failed_stream)

        app = AgentApp()
        notifications = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda *args, **kwargs: notifications.append((args, kwargs)),
        )
        async with app.run_test() as pilot:
            worker = app.run_agent_turn()
            await worker.wait()
            await pilot.pause()

            assert app.is_running
            assert app.query_one("#prompt").disabled is False
            assert len(notifications) == 1
            assert notifications[0][1]["severity"] == "error"

    asyncio.run(exercise())
