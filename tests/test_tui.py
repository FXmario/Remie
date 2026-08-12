import asyncio

import httpx

import remie.tui as tui
from remie.agent import ConnectionConfig, OPENCODE_GO_BASE_URL
from remie.tui import (
    AgentApp,
    ConnectionScreen,
    ModelBadge,
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


def test_model_badge_includes_vendor():
    local_badge = ModelBadge()
    local_badge.update_config(ConnectionConfig("http://localhost/v1", "key", "local"))
    assert local_badge.render().plain == "local  Local"

    remote_badge = ModelBadge()
    remote_badge.update_config(ConnectionConfig(OPENCODE_GO_BASE_URL, "key", "kimi-k3"))
    assert remote_badge.render().plain == "kimi-k3  OpenCode Go"


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


def test_open_connection_ignored_while_agent_running(monkeypatch):
    async def exercise():
        monkeypatch.setattr(tui, "stream_llm_call", _empty_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._agent_running = True
            await app.action_open_connection()
            await pilot.pause()
            assert len(app.screen_stack) == 1

    asyncio.run(exercise())


def test_open_connection_opens_modal_when_idle():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert len(app.screen_stack) == 2
            assert app.screen.query_one("#submit-button")
            assert isinstance(app.screen, ConnectionScreen)

    asyncio.run(exercise())


def test_escape_stops_running_agent(monkeypatch):
    async def exercise():
        async def endless_stream(_conversation):
            while True:
                yield "chunk"
                await asyncio.sleep(0)

        monkeypatch.setattr(tui, "stream_llm_call", endless_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._agent_running = True
            worker = app.run_agent_turn()
            await pilot.pause()
            app.action_stop_agent()
            await pilot.pause()
            assert app._stop_requested
            await worker.wait()
            await pilot.pause()
            assert app._agent_running is False

    asyncio.run(exercise())


async def _empty_stream(_conversation):
    return
    yield  # pragma: no cover
