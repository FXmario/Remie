import asyncio

import httpx
from rich.panel import Panel

import fuica.tui as tui
from fuica.agent import ConnectionConfig, OPENCODE_GO_BASE_URL
from fuica.tui import (
    AgentApp,
    ConnectionScreen,
    ModelBadge,
    StatusIndicator,
    ThinkingIndicator,
    _format_tokens,
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
        async def failed_stream(_conversation, usage_box=None, reasoning_box=None):
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
            await app.run_agent_turn("hello")
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
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None):
            while True:
                yield "chunk"
                await asyncio.sleep(0)

        monkeypatch.setattr(tui, "stream_llm_call", endless_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._agent_running = True
            task = asyncio.create_task(app.run_agent_turn("hello"))
            await pilot.pause()
            app.action_stop_agent()
            await pilot.pause()
            assert app._stop_requested
            await task
            await pilot.pause()
            assert app._agent_running is False

    asyncio.run(exercise())


def test_messages_are_queued_and_processed_in_order(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None):
            yield "reply"

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._input_queue.put_nowait("hello")
            app._input_queue.put_nowait("world")
            worker = app.message_worker()
            app._input_queue.put_nowait(None)
            await worker.wait()
            await pilot.pause()

            assert app._agent_running is False
            user_contents = [
                m["content"] for m in app.conversation if m["role"] == "user"
            ]
            assert user_contents == ["hello", "world"]

    asyncio.run(exercise())


def test_stop_drains_pending_messages(monkeypatch):
    async def exercise():
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None):
            while True:
                yield "chunk"
                await asyncio.sleep(0)

        monkeypatch.setattr(tui, "stream_llm_call", endless_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._input_queue.put_nowait("first")
            app._input_queue.put_nowait("queued")
            worker = app.message_worker()
            await pilot.pause()
            app.action_stop_agent()
            await worker.wait()
            await pilot.pause()

            assert app._input_queue.empty()
            user_contents = [
                m["content"] for m in app.conversation if m["role"] == "user"
            ]
            assert user_contents == ["first"]

    asyncio.run(exercise())


def test_edit_tool_writes_diff_panel_to_log(monkeypatch, tmp_path):
    async def exercise():
        file = tmp_path / "doc.txt"
        file.write_text("hello\n", encoding="utf-8")

        calls = 0

        async def tool_stream(_conversation, usage_box=None, reasoning_box=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield (
                    f'tool: edit_file({{"path": "{file}", '
                    '"old_str": "hello", "new_str": "hi"})'
                )
            else:
                yield "done"

        monkeypatch.setattr(tui, "stream_llm_call", tool_stream)
        rendered = []
        monkeypatch.setattr(
            tui, "_render_diff", lambda diff: rendered.append(diff) or Panel("")
        )

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("edit it")
            await pilot.pause()

            assert file.read_text(encoding="utf-8") == "hi\n"
            assert len(rendered) == 1
            assert "-hello" in rendered[0]
            assert "+hi" in rendered[0]

    asyncio.run(exercise())


async def _empty_stream(_conversation, usage_box=None, reasoning_box=None):
    return
    yield  # pragma: no cover


def test_format_tokens():
    assert _format_tokens(0) == "0"
    assert _format_tokens(999) == "999"
    assert _format_tokens(1234) == "1.2k"
    assert _format_tokens(25000) == "25k"


def test_model_badge_shows_token_usage():
    badge = ModelBadge()
    badge.update_config(ConnectionConfig(OPENCODE_GO_BASE_URL, "key", "kimi-k3"))
    badge.set_tokens(1234, 5678)
    assert badge.render().plain == "kimi-k3  OpenCode Go · 6.9k tok"


def test_model_badge_hides_usage_when_zero():
    badge = ModelBadge()
    badge.update_config(ConnectionConfig("http://localhost/v1", "key", "local"))
    assert badge.render().plain == "local  Local"


def test_turn_updates_badge_tokens(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None
        ):
            if reasoning_box is not None:
                reasoning_box.append("reasoning text")
            usage_box["prompt_tokens"] = 100
            usage_box["completion_tokens"] = 50
            yield "reply"

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            badge = app.query_one(ModelBadge)
            assert "tok" in badge.render().plain
            assert app._total_input_tokens == 100
            assert app._total_output_tokens == 50

    asyncio.run(exercise())


def test_final_answer_strips_protocol_lines(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None
        ):
            yield "thinking: let me think"
            yield "\n"
            yield "Here is the answer."

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)
        captured = {}
        monkeypatch.setattr(
            tui,
            "render_assistant_panel",
            lambda text, code_theme="ansi_dark": captured.update(text=text)
            or Panel(""),
        )

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert app._agent_running is False
            assert app.conversation[-1]["content"] == (
                "thinking: let me think\nHere is the answer."
            )
            assert captured["text"] == "Here is the answer."

    asyncio.run(exercise())
