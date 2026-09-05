import asyncio
import base64
import httpx
import json
import pytest
from PIL import Image
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.widgets import Button, Input, Label, OptionList, Select
from textual.widgets import RadioButton, RadioSet

from remie.tui.constants import REASONING_EFFORTS
from remie.tui.effort_slider import EffortSlider

import remie.tui as tui
import remie.tui.app as tui_app
import remie.tui.screens.connection as connection_screen
from remie.agent import (
    LLMRequestError,
    ConnectionConfig,
    OPENCODE_GO_BASE_URL,
    configure_openai,
    get_config,
)
from remie.tui import (
    MAX_AUTO_CONTINUATIONS,
    AgentApp,
    AskUserScreen,
    ChatScreen,
    ConnectionScreen,
    MemoryScreen,
    ModelBadge,
    PromptSubmitted,
    PromptTextArea,
    StatusIndicator,
    StreamingRichLog,
    ThinkingIndicator,
    _format_tokens,
    _has_tool_call,
    _is_tmux,
    _load_status_gif,
)
from remie.tools import (
    create_chat,
    delete_chat,
    find_chat_by_id,
    find_memory_by_id,
    get_active_memory_id,
    list_chats,
    load_chat,
    load_chat_index,
    load_latest_chat,
    memory_tool,
    save_chat,
    save_chat_index,
    set_active_memory_id,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point Remie's saved connection config at a temp dir so tests neither
    read nor overwrite the user's real ~/.config/remie/config.json, and reset
    the in-memory active connection (loaded at import time from the real file)
    back to the local default so provider-specific behavior is deterministic."""
    import remie.agent as agent

    config_dir = tmp_path / "remie-config"
    monkeypatch.setattr(agent, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(agent, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(agent, "_config", agent._default_config())
    yield


@pytest.fixture(autouse=True)
def _no_network_on_mount(monkeypatch):
    """Keep the app's mount-time model prefetch offline during tests, since the
    active config may be a saved OpenCode Go connection."""

    async def _noop_fetch(api_key):
        return ["kimi-k3"]

    monkeypatch.setattr(tui_app, "fetch_opencode_go_models", _noop_fetch)


def test_preview_window_short_text_unchanged():
    assert tui._preview_window("hello") == "hello"


def test_preview_window_bounds_tail():
    text = "x" * 5000
    window = tui._preview_window(text)
    assert len(window) <= tui.STREAM_PREVIEW_MAX_CHARS
    assert window == text[-tui.STREAM_PREVIEW_MAX_CHARS:]


def test_preview_window_starts_at_line_boundary():
    chunk = "\n".join(f"line-{i}" for i in range(100))
    window = tui._preview_window(chunk, limit=200)
    assert len(window) <= 200 + max(
        len(line) + 1 for line in chunk.splitlines()
    )
    assert window.startswith("line-") or window == chunk
    assert window == chunk[-len(window):]


def test_safe_stream_markdown_plain_without_fence():
    result = tui._safe_stream_markdown("just some **text**", "ansi_dark")
    from rich.text import Text
    assert isinstance(result, Text)


def test_safe_stream_markdown_uses_markdown_with_fence():
    result = tui._safe_stream_markdown("```python\nx = 1\n```", "ansi_dark")
    from rich.markdown import Markdown
    assert isinstance(result, Markdown)


def test_safe_stream_markdown_reasoning_keeps_grey_style():
    result = tui._safe_reasoning_markdown("plain reasoning", "ansi_dark")
    from rich.text import Text
    assert isinstance(result, Text)
    assert result.style is not None


def test_tmux_detection(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    assert not _is_tmux()

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    assert _is_tmux()


def test_background_detection_skips_osc_query_in_tmux(monkeypatch):
    import remie.tui.helpers as helpers

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    writes = []
    monkeypatch.setattr(helpers.sys.stdout, "write", writes.append)

    assert helpers._detect_terminal_background() is None
    assert writes == []


def test_tmux_thinking_indicator_visibility(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    indicator = ThinkingIndicator()

    indicator.set_status("working")
    assert indicator._working
    assert indicator.display

    indicator.set_status("ready")
    assert not indicator.display


def test_tmux_spinner_lives_next_to_model_badge(monkeypatch):
    async def exercise():
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
        app = AgentApp()
        async with app.run_test() as pilot:
            row = app.query_one("#model-row")
            assert row.query_one(ThinkingIndicator) is not None
            assert row.query_one(ModelBadge) is not None
            assert row.query_one("#tmux-spinner", ThinkingIndicator) is not None

    asyncio.run(exercise())


def test_tmux_spinner_hidden_when_idle(monkeypatch):
    async def exercise():
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
        app = AgentApp()
        async with app.run_test() as pilot:
            indicator = app.query_one(ThinkingIndicator)
            assert indicator.display is False

    asyncio.run(exercise())


def test_sidebar_toggle_buttons_have_matching_labels_and_positions():
    async def exercise():
        app = AgentApp()
        app._tab_layout["sidebar_visible"] = True
        async with app.run_test() as pilot:
            sidebar = app.query_one("#tab-sidebar")
            hide = app.query_one("#tab-hide", Button)
            show = app.query_one("#tabs-show", Button)

            header = app.query_one("#tabs-header")
            title = app.query_one("#tabs-title")
            assert sidebar.children[0] is header
            assert list(header.children) == [title, hide]
            assert hide.region.x > title.region.x
            assert str(hide.label) == "< Hide"
            assert str(show.label) == "> Tabs"
            assert hide.size.height == 1

            await pilot.click("#tab-hide")
            assert sidebar.display is False
            assert show.display is True

            await pilot.click("#tabs-show")
            assert sidebar.display is True
            assert show.display is False

    asyncio.run(exercise())


def test_status_gifs_load_with_frame_timing():
    frames, durations = _load_status_gif("ready.gif")

    assert len(frames) > 1
    assert len(frames) == len(durations)
    assert all(frame.mode == "RGBA" for frame in frames)
    assert all(duration > 0 for duration in durations)
    assert all(duration >= 1 / tui.STATUS_ANIMATION_MAX_FPS for duration in durations)


def test_status_indicator_starts_ready():
    indicator = StatusIndicator()

    assert indicator._state == "ready"
    assert len(indicator._ensure_loaded("working")[0]) > 1


def test_status_gifs_load_lazily():
    indicator = StatusIndicator()
    # Nothing loads at construction; each state loads on first use and caches.
    assert indicator._frames == {}
    frames, durations = indicator._ensure_loaded("working")
    assert len(frames) > 1
    assert indicator._ensure_loaded("working") is indicator._frames["working"]
    assert "done" not in indicator._frames


def test_missing_status_gif_returns_empty_animation():
    assert _load_status_gif("does-not-exist.gif") == ([], [])


def test_tui_starts_without_status_gif_assets(monkeypatch):
    """Optional animation files must never be required to mount the app."""
    import remie.tui.widgets as widgets

    monkeypatch.setattr(widgets, "_load_status_gif", lambda _name: ([], []))

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            indicator = app.query_one(StatusIndicator)
            assert indicator.display is False
            assert not indicator.query("#status-gif")
            # Status transitions and preference toggles remain safe too.
            app._set_status("working")
            indicator.set_animation_enabled(False)
            indicator.set_animation_enabled(True)
            assert indicator.display is False

    asyncio.run(exercise())


def test_missing_later_status_gif_disables_animation(monkeypatch):
    """A missing working/done asset must not crash a mounted indicator."""
    import remie.tui.widgets as widgets

    original_loader = widgets._load_status_gif

    def load_except_working(name):
        return ([], []) if name == "working.gif" else original_loader(name)

    monkeypatch.setattr(widgets, "_load_status_gif", load_except_working)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            indicator = app.query_one(StatusIndicator)
            indicator.set_status("working")
            await pilot.pause()
            assert indicator.display is False

    asyncio.run(exercise())


def test_load_status_gif_missing_file_returns_empty(tmp_path, monkeypatch):
    """The real loader degrades to empty data when no GIF exists anywhere."""
    import remie.tui.widgets as widgets

    # Point the cwd fallback at an empty dir; the package-relative candidate
    # is bypassed by asking for a name that never ships.
    monkeypatch.chdir(tmp_path)
    assert widgets._load_status_gif("does-not-exist.gif") == ([], [])


def test_tui_launches_without_status_gif_assets(monkeypatch, tmp_path):
    """A directory without assets/*.gif must not crash the TUI at startup.

    Regression test: the installed CLI once called PILImage.open() on a
    non-existent asset path, raising FileNotFoundError during compose().
    """
    import remie.tui.widgets as widgets

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        widgets,
        "_load_status_gif",
        lambda name: ([], []),
    )

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            indicator = app.query_one(StatusIndicator)
            assert indicator.display is False
            # Status transitions stay harmless without any frames.
            for status in ("working", "done", "ready"):
                indicator.set_status(status)
                await pilot.pause()
                assert indicator.display is False

    asyncio.run(exercise())


def test_ctrl_g_toggles_and_persists_status_animation(monkeypatch, tmp_path):
    import remie.agent as agent

    monkeypatch.setattr(agent, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(agent, "CONFIG_DIR", tmp_path)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            indicator = app.query_one(StatusIndicator)
            assert indicator.display is True

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert indicator.display is False
            assert agent.load_status_animation_enabled() is False

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert indicator.display is True
            assert agent.load_status_animation_enabled() is True

    asyncio.run(exercise())


def test_startup_theme_matches_terminal_background(monkeypatch):
    monkeypatch.setattr(tui_app, "_detect_terminal_background", lambda: "light")
    assert AgentApp().theme == "ansi-light"

    monkeypatch.setattr(tui_app, "_detect_terminal_background", lambda: "dark")
    assert AgentApp().theme == "ansi-dark"

    monkeypatch.setattr(tui_app, "_detect_terminal_background", lambda: None)
    assert AgentApp().theme == "ansi-dark"


def test_ctrl_t_toggles_full_light_and_dark_themes(monkeypatch):
    monkeypatch.setattr(tui_app, "_detect_terminal_background", lambda: "dark")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            assert app._theme_mode == "system"
            assert app.theme == "ansi-dark"
            assert app._code_theme() == "ansi_dark"

            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app._theme_mode == "light"
            assert app.theme == "textual-light"
            assert app._code_theme() == "ansi_light"

            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app._theme_mode == "dark"
            assert app.theme == "textual-dark"
            assert app._code_theme() == "ansi_dark"

            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app._theme_mode == "system"
            assert app.theme == "ansi-dark"
            assert app._code_theme() == "ansi_dark"

    asyncio.run(exercise())


def test_ctrl_g_toggles_static_status_image_in_tmux(monkeypatch, tmp_path):
    import remie.agent as agent

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    monkeypatch.setattr(agent, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(agent, "CONFIG_DIR", tmp_path)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            indicator = app.query_one(StatusIndicator)
            assert indicator.display is True
            assert indicator._timer is None
            assert indicator.query_one("#status-gif")

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert indicator.display is False

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert indicator.display is True
            assert indicator._timer is None

    asyncio.run(exercise())


def test_stream_update_interval_is_bounded_by_preview_window():
    # The interval never grows past the preview cap: for text longer than the
    # preview window the cost is constant, so the throttle stays at a floor
    # instead of scaling with the (unbounded) accumulated text.
    assert tui._stream_update_interval(0) == tui.STREAM_UPDATE_MIN_INTERVAL
    assert tui._stream_update_interval(100) == tui.STREAM_UPDATE_MIN_INTERVAL
    huge = tui.STREAM_PREVIEW_MAX_CHARS * 100
    assert tui._stream_update_interval(huge) == (
        tui.STREAM_UPDATE_MIN_INTERVAL_LARGE
    )
    assert tui._stream_update_interval(huge) < (
        huge / tui.STREAM_UPDATE_CHARS_PER_SECOND
    )


def test_stream_update_interval_adaptive_floor():
    # Short previews use the fast ~30 fps floor; once the text passes the
    # large-preview threshold the floor rises to keep render work bounded.
    fast = tui._stream_update_interval(tui.STREAM_UPDATE_LARGE_PREVIEW_CHARS)
    slow = tui._stream_update_interval(
        tui.STREAM_UPDATE_LARGE_PREVIEW_CHARS + 1
    )
    assert fast == tui.STREAM_UPDATE_MIN_INTERVAL
    assert slow >= tui.STREAM_UPDATE_MIN_INTERVAL_LARGE


def test_push_message_updates_conversation_token_cache(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app.conversation = [{"role": "system", "content": "sys"}]
            app._cached_conv_tokens = tui.estimate_conversation_tokens(
                app.conversation
            )
            before = app._cached_conv_tokens

            app._push_message("user", "hello world")
            assert app.conversation[-1] == {"role": "user", "content": "hello world"}
            assert app._cached_conv_tokens == before + tui.estimate_message_tokens(
                {"role": "user", "content": "hello world"}
            )

            app._push_message("user", "a" * 400 + "\n" * 300)
            assert app._cached_conv_tokens == before + tui.estimate_tokens(
                "hello world"
            ) + tui.estimate_tokens("a" * 400 + "\n" * 300)

    asyncio.run(exercise())


def test_conversation_too_large_uses_cached_tokens(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app.conversation = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x" * 200},
            ]
            app._cached_conv_tokens = tui.estimate_conversation_tokens(
                app.conversation
            )
            limit = app._cached_conv_tokens // tui.COMPACTION_CONTEXT_RATIO
            assert app._conversation_too_large(limit) is True
            assert app._conversation_too_large(None) is False
            assert app._conversation_too_large(10**9) is False

    asyncio.run(exercise())


def test_compaction_recomputes_token_cache(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app.conversation = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "old" * 200},
                {"role": "assistant", "content": "old" * 200},
                {"role": "user", "content": "recent"},
            ]
            app._cached_conv_tokens = 999_999
            await app._compact_conversation()
            assert app._cached_conv_tokens == tui.estimate_conversation_tokens(
                app.conversation
            )

    asyncio.run(exercise())


def test_model_badge_includes_vendor():
    local_badge = ModelBadge()
    local_badge.update_config(
        ConnectionConfig("http://localhost/v1", "key", "local", reasoning_effort="off")
    )
    assert local_badge.render().plain == "Local  Local"

    remote_badge = ModelBadge()
    remote_badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL, "key", "kimi-k3", reasoning_effort="off"
        )
    )
    assert remote_badge.render().plain == "Kimi K3  OpenCode Go"

def test_connection_error_shows_toast_and_keeps_app_running(monkeypatch):
    async def exercise():
        async def failed_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            raise httpx.ConnectError("connection refused")
            yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", failed_stream)

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


def test_management_shortcut_replaces_individual_picker_shortcuts():
    keys = {
        binding[0] if isinstance(binding, tuple) else binding.key
        for binding in AgentApp.BINDINGS
    }
    assert "ctrl+p" in keys
    assert keys.isdisjoint({"ctrl+r", "ctrl+o"})


def test_open_connection_ignored_while_agent_running(monkeypatch):
    async def exercise():
        monkeypatch.setattr(tui_app, "stream_llm_call", _empty_stream)

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
            await app.action_open_connection()
            await pilot.pause()
            assert len(app.screen_stack) == 2
            assert app.screen.query_one("#submit-button")
            assert isinstance(app.screen, ConnectionScreen)

    asyncio.run(exercise())


def test_escape_stops_running_agent(monkeypatch):
    async def exercise():
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            while True:
                yield "chunk"
                await asyncio.sleep(0)

        monkeypatch.setattr(tui_app, "stream_llm_call", endless_stream)

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


def test_escape_key_stops_stalled_agent(monkeypatch):
    async def exercise():
        stalled = asyncio.Event()

        async def stalled_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            await stalled.wait()
            yield "never"

        monkeypatch.setattr(tui_app, "stream_llm_call", stalled_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            task = asyncio.create_task(app.run_agent_turn("hello"))
            await pilot.pause()
            await pilot.press("escape")
            await task

            assert app._agent_running is False
            assert app._stop_requested is True

    asyncio.run(exercise())


def test_escape_in_agent_question_stops_agent():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app._agent_running = True
            await app.push_screen(AskUserScreen("continue?", ["yes", "no"]))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            assert len(app.screen_stack) == 1
            assert app._stop_requested is True

    asyncio.run(exercise())


def test_messages_are_queued_and_processed_in_order(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            yield "reply"

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

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
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            while True:
                yield "chunk"
                await asyncio.sleep(0)

        monkeypatch.setattr(tui_app, "stream_llm_call", endless_stream)

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


def test_blocked_command_shows_blocked_log_line(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield 'tool: run_command({"command": "rm -rf /"})'
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("run it")
            await pilot.pause()

            log_lines = [
                strip.text
                for strip in app.query_one("#log").lines
            ]
            joined = "\n".join(log_lines)
            assert "Blocked command:" in joined
            assert "recursive forced delete" in joined
            assert "rm -rf /" in joined

    asyncio.run(exercise())


def test_edit_tool_writes_diff_panel_to_log(monkeypatch, tmp_path):
    async def exercise():
        file = tmp_path / "doc.txt"
        file.write_text("hello\n", encoding="utf-8")

        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield (
                    f'tool: edit_file({{"path": "{file}", '
                    '"old_str": "hello", "new_str": "hi"})'
                )
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)
        rendered = []
        monkeypatch.setattr(
            tui_app, "_render_diff", lambda diff: rendered.append(diff) or Panel("")
        )

        app = AgentApp()
        app._tool_executor.project_root = tmp_path
        async with app.run_test() as pilot:
            await app.run_agent_turn("edit it")
            await pilot.pause()

            assert file.read_text(encoding="utf-8") == "hi\n"
            assert len(rendered) == 1
            assert "-hello" in rendered[0]
            assert "+hi" in rendered[0]

    asyncio.run(exercise())


def test_truncated_response_continues_automatically(monkeypatch):
    async def exercise():
        calls = 0

        async def truncating_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                finish_box["finish_reason"] = "length"
                finish_box["truncated"] = True
                yield "The answer is cut "
            else:
                yield "off here, now complete."

        monkeypatch.setattr(tui_app, "stream_llm_call", truncating_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert calls == 2
            assistant_messages = [
                m["content"]
                for m in app.conversation
                if m["role"] == "assistant"
            ]
            assert assistant_messages == [
                "The answer is cut ",
                "off here, now complete.",
            ]

    asyncio.run(exercise())


def test_truncated_tool_call_continues_before_executing(monkeypatch, tmp_path):
    async def exercise():
        file = tmp_path / "doc.txt"
        file.write_text("hello\n", encoding="utf-8")

        calls = 0

        async def truncating_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                finish_box["finish_reason"] = "length"
                finish_box["truncated"] = True
                yield (
                    f'tool: edit_file({{"path": "{file}", '
                    '"old_str": "hello", "new_str": "hi"})'
                )
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", truncating_stream)

        app = AgentApp()
        app._tool_executor.project_root = tmp_path
        async with app.run_test() as pilot:
            await app.run_agent_turn("edit it")
            await pilot.pause()

            # Complete tool call still executes; no bogus continuation.
            assert file.read_text(encoding="utf-8") == "hi\n"
            assert calls == 2

    asyncio.run(exercise())


def test_truncation_continuation_limit(monkeypatch):
    async def exercise():
        calls = 0

        async def always_truncating(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            finish_box["finish_reason"] = "length"
            finish_box["truncated"] = True
            yield "x"

        monkeypatch.setattr(tui_app, "stream_llm_call", always_truncating)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert calls == MAX_AUTO_CONTINUATIONS + 1
            assistant_messages = [
                m["content"]
                for m in app.conversation
                if m["role"] == "assistant"
            ]
            assert len(assistant_messages) == MAX_AUTO_CONTINUATIONS + 1

    asyncio.run(exercise())


def test_ask_user_modal_renders_question_and_options():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.push_screen(AskUserScreen("pick one", ["a", "b", "c"]))
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#ask-question").render().plain == "pick one"
            options = screen.query_one("#ask-options", OptionList)
            assert [option.prompt for option in options.options] == [
                "1. a",
                "2. b",
                "3. c",
                "4. Write your answer",
            ]
            assert screen.query_one("#ask-input", Input).display is False
            assert screen.query_one("#ask-submit", Button).display is False
            assert screen.query_one("#ask-close", Button)
            assert not screen.query("#ask-cancel")
            dialog = screen.query_one("#ask-dialog")
            footer = screen.query_one("#ask-footer")
            assert options.region.height == len(options.options) + 2
            assert options.region.bottom < footer.region.y
            assert footer.region.bottom <= dialog.content_region.bottom

    asyncio.run(exercise())


def test_ask_user_write_answer_reveals_input_and_submit():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.push_screen(AskUserScreen("pick one", ["a", "b"]))
            await pilot.pause()
            screen = app.screen
            option_list = screen.query_one("#ask-options", OptionList)
            custom_option = option_list.options[2]
            screen.on_option_list_option_selected(
                OptionList.OptionSelected(option_list, custom_option, 2)
            )
            await pilot.pause()

            answer_input = screen.query_one("#ask-input", Input)
            assert option_list.display is False
            assert answer_input.display is True
            assert screen.query_one("#ask-submit", Button).display is True
            assert answer_input.has_focus

    asyncio.run(exercise())


def test_ask_user_selecting_option_dismisses():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.push_screen(AskUserScreen("pick one", ["a", "b"]))
            await pilot.pause()
            screen = app.screen
            option_list = screen.query_one("#ask-options", OptionList)
            dismissed = []
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))
            try:
                option = option_list.options[1]
                screen.on_option_list_option_selected(
                    OptionList.OptionSelected(option_list, option, 1)
                )
            finally:
                monkeypatch.undo()
            # A real option dismisses with its value; an empty selection does not.
            assert dismissed == ["b"]

    asyncio.run(exercise())


def test_ask_user_tool_feeds_answer_back(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield (
                    'tool: ask_user({"question": "pick one", '
                    '"options": ["a", "b"]})'
                )
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            async def fake_push(screen):
                assert isinstance(screen, AskUserScreen)
                return "a"

            monkeypatch.setattr(app, "push_screen_wait", fake_push)
            await app.run_agent_turn("hello")
            await pilot.pause()

            user_messages = [
                m["content"] for m in app.conversation if m["role"] == "user"
            ]
            assert any("answer" in m and "a" in m for m in user_messages)
            assert any('"answer": "a"' in m for m in user_messages)
            assert app.conversation[-1]["role"] == "assistant"

    asyncio.run(exercise())


def test_ask_user_cancel_marks_cancelled(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield 'tool: ask_user({"question": "q"})'
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            async def fake_cancel(screen):
                return None

            monkeypatch.setattr(app, "push_screen_wait", fake_cancel)
            await app.run_agent_turn("hello")
            await pilot.pause()

            user_messages = [
                m["content"] for m in app.conversation if m["role"] == "user"
            ]
            assert any('"cancelled": true' in m for m in user_messages)

    asyncio.run(exercise())


async def _empty_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
    return
    yield  # pragma: no cover


def test_format_tokens():
    assert _format_tokens(0) == "0"
    assert _format_tokens(999) == "999"
    assert _format_tokens(1234) == "1.2k"
    assert _format_tokens(25000) == "25k"


def test_has_tool_call_detects_dsml():
    assert _has_tool_call(
        '<|DSML|>invoke name="list-files">\n<|DSML|>parameter path="." />'
    )
    assert _has_tool_call('tool: read_file({"filename": "a.py"})')
    assert _has_tool_call("<tool: list_files(path='.')>")
    assert not _has_tool_call("just a normal reply")


def test_format_tool_result_read_file():
    text = tui._format_tool_result(
        "read_file", {"file_path": "main.py", "content": "abc"}
    )
    assert text == "Read main.py (3 chars)"


def test_format_tool_result_run_command():
    text = tui._format_tool_result(
        "run_command",
        {"exit_code": 0, "stdout": "hello\nworld\n", "stderr": "", "timed_out": False},
    )
    assert "exit 0" in text
    assert "hello\nworld" in text


def test_format_tool_result_error_and_blocked():
    assert tui._format_tool_result("read_file", {"error": "boom"}) == "Error: boom"
    assert tui._format_tool_result("run_command", {"blocked": True, "reason": "no"}) == ""


def test_format_tool_result_ask_user():
    assert tui._format_tool_result("ask_user", {"answer": "yes"}) == "Answer: yes"
    assert tui._format_tool_result("ask_user", {"cancelled": True}) == "Cancelled"


def test_render_tool_result_panel_and_truncation():
    panel = tui._render_tool_result("read_file", {"file_path": "a.py", "content": "x"})
    assert panel is not None
    assert panel.title == "Tool result · read_file"

    big = tui._render_tool_result(
        "run_command",
        {"exit_code": 0, "stdout": "z" * 5000, "stderr": "", "timed_out": False},
    )
    assert "result truncated" in big.renderable.plain

    assert tui._render_tool_result("run_command", {"blocked": True}) is None


def test_tool_results_rendered_in_turn(monkeypatch, tmp_path):
    async def exercise():
        path = tmp_path / "x.txt"
        path.write_text("hello", encoding="utf-8")

        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield f'tool: read_file({{"filename": "{path}"}})'
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)
        rendered = []
        monkeypatch.setattr(
            tui_app,
            "_render_tool_result",
            lambda name, result, code_theme="ansi_dark": rendered.append(
                (name, result)
            )
            or Panel(""),
        )

        app = AgentApp()
        app._tool_executor.project_root = tmp_path
        async with app.run_test() as pilot:
            await app.run_agent_turn("read it")
            await pilot.pause()

            assert len(rendered) == 1
            assert rendered[0][0] == "read_file"
            assert rendered[0][1]["content"] == "hello"

    asyncio.run(exercise())


def test_panel_decoration_is_removed_from_log_selection():
    assert StreamingRichLog._without_panel_border("╭──── Assistant ────╮") == "Assistant"
    assert StreamingRichLog._without_panel_border("│  answer text       │") == " answer text"
    assert StreamingRichLog._without_panel_border("╰────────────────────╯") is None
    # Ordinary content that happens to contain a box character is unchanged.
    assert StreamingRichLog._without_panel_border("value │ value") == "value │ value"
    assert StreamingRichLog._panel_content_bounds("│ answer │") == (2, 8)
    assert StreamingRichLog._panel_content_bounds("╰────────╯") is None


def test_live_output_does_not_force_scrolled_log_to_bottom():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(100, 30)) as pilot:
            log = app.query_one("#log", StreamingRichLog)
            for index in range(60):
                log.write(f"history {index}")
            log.begin_stream()
            log.update_stream("initial response")
            await pilot.pause()

            log.scroll_to(y=0, animate=False, immediate=True)
            assert not log.is_vertical_scroll_end

            # Both ordinary agent events and in-place streaming updates must
            # preserve a user's position while they read earlier content.
            log.write("tool event")
            log.update_stream("updated response\n" * 20)
            await pilot.pause()

            assert log.scroll_y == 0
            assert not log.is_vertical_scroll_end

    asyncio.run(exercise())


def test_mouse_release_copies_screen_selection_immediately(monkeypatch):
    async def exercise():
        app = AgentApp()
        copied = []
        notifications = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        monkeypatch.setattr(
            app, "notify", lambda *args, **kwargs: notifications.append(kwargs)
        )

        async with app.run_test() as pilot:
            log = app.query_one("#log")
            log.write(Panel("before How it works after"))
            await pilot.pause()
            line_y, line = next(
                (line_y, strip.text)
                for line_y, strip in enumerate(log.lines)
                if "How it works" in strip.text
            )
            phrase_x = line.index("How it works")
            start = (log.gutter.left + phrase_x, log.gutter.top + line_y)
            end = (start[0] + len("How it works") - 1, start[1])
            await pilot.mouse_down(log, offset=start)
            await pilot.hover(log, offset=end)
            await pilot.pause()

            selected_style = app.screen.get_component_rich_style(
                "screen--selection"
            )
            rendered_line = log.render_line(line_y - int(log.scroll_y))
            highlighted = next(
                segment for segment in rendered_line if segment.text == "How it works"
            )
            assert highlighted.style.bgcolor == selected_style.bgcolor
            assert highlighted.style.color == selected_style.color
            assert copied == []

            await pilot.mouse_up(log, offset=end)
            await pilot.pause()

            assert copied == ["How it works"]
            assert notifications[-1].get("title") == "Selection"
            assert app.screen.selections == {}
            # A later mouse-up that reports the same stale selection (for
            # example, over the scrollbar) must not copy it a second time.
            app.screen.post_message(events.TextSelected())
            await pilot.pause()
            assert copied == ["How it works"]

    asyncio.run(exercise())


def test_mouse_release_without_selection_does_not_copy(monkeypatch):
    async def exercise():
        app = AgentApp()
        copied = []
        notifications = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        monkeypatch.setattr(
            app, "notify", lambda *args, **kwargs: notifications.append(kwargs)
        )

        async with app.run_test() as pilot:
            app.screen.post_message(events.TextSelected())
            await pilot.pause()
            assert copied == []
            assert notifications == []

    asyncio.run(exercise())


def test_mouse_release_copies_prompt_selection(monkeypatch):
    async def exercise():
        app = AgentApp()
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        monkeypatch.setattr(app, "notify", lambda *args, **kwargs: None)

        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.load_text("copy this")
            await pilot.pause()
            await pilot.mouse_down(prompt, offset=(1, 1))
            await pilot.hover(prompt, offset=(6, 1))
            await pilot.mouse_up(prompt, offset=(6, 1))
            await pilot.pause()

            assert prompt.selected_text == ""
            assert prompt.selection.start == prompt.selection.end
            assert copied == ["copy"]

    asyncio.run(exercise())


def test_ctrl_c_copies_selection(monkeypatch):
    async def exercise():
        app = AgentApp()
        copied = []
        notifications = []
        monkeypatch.setattr(
            app, "copy_to_clipboard", lambda text: copied.append(text)
        )
        monkeypatch.setattr(
            app, "notify", lambda *a, **k: notifications.append(k)
        )
        async with app.run_test() as pilot:
            log = app.query_one("#log")
            log.write("selectable text")
            await pilot.pause()
            app.screen._select_all_in_widget(log)
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()

            assert copied == ["selectable text"]
            assert app.is_running
            assert notifications and notifications[-1].get("title") == "Selection"

    asyncio.run(exercise())


def test_ctrl_c_quits_without_selection():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.is_running is False

    asyncio.run(exercise())


def test_conversation_compaction_keeps_system_and_recent(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        app._prompt_history = []
        async with app.run_test() as pilot:
            app.conversation = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "old" * 200},
                {"role": "assistant", "content": "old" * 200},
                {"role": "user", "content": "tool_result(" + "x" * 500 + ")"},
                {"role": "user", "content": "recent"},
            ]
            assert app._conversation_too_large(500) is True
            await app._compact_conversation()
            assert app.conversation[0]["role"] == "system"
            assert "omitted" in app.conversation[1]["content"]
            assert app.conversation[-1]["content"] == "recent"
            assert len(app.conversation) == 6

    asyncio.run(exercise())


def test_compaction_summarizes_dropped_messages(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            yield "SUMMARY: user wanted type hints and a ruff config"

        monkeypatch.setattr("remie.agent.stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app.conversation = [
                {"role": "system", "content": "sys"},
                *[
                    {"role": "user", "content": f"old message {i} " * 30}
                    for i in range(14)
                ],
            ]
            await app._compact_conversation()
            assert "SUMMARY: user wanted" in app.conversation[1]["content"]
            assert app.conversation[1]["role"] == "system"

    asyncio.run(exercise())


def test_compaction_falls_back_when_summary_fails(monkeypatch):
    async def exercise():
        async def failing_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            raise RuntimeError("boom")
            yield

        monkeypatch.setattr("remie.agent.stream_llm_call", failing_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app.conversation = [
                {"role": "system", "content": "sys"},
                *[
                    {"role": "user", "content": f"old message {i} " * 30}
                    for i in range(14)
                ],
            ]
            await app._compact_conversation()
            assert "omitted" in app.conversation[1]["content"]

    asyncio.run(exercise())


def test_on_mount_resumes_latest_chat_and_preserves_memory(monkeypatch):
    async def exercise():
        old_memory = memory_tool("add", "keep this", name="old work")
        chat = create_chat("earlier work")
        save_chat(
            chat["id"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "hi"},
            ],
            [
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        app = AgentApp()
        async with app.run_test() as pilot:
            assert app._chat_id == chat["id"]
            assert len(app.conversation) == 3
            assert app.conversation[0]["role"] == "system"
            # The saved system prompt is replaced with the current one.
            assert "Agent memory" in app.conversation[0]["content"]
            assert app._cached_conv_tokens == tui.estimate_conversation_tokens(
                app.conversation
            )
            log_lines = [strip.text for strip in app.query_one("#log").lines]
            assert any("Resumed chat" in line for line in log_lines)
            assert any("hello there" in line for line in log_lines)
            active = find_memory_by_id(get_active_memory_id())
            assert active is not None
            assert active["name"] == "old work"

    asyncio.run(exercise())


def test_on_mount_fresh_when_no_chats(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            assert len(app.conversation) == 1
            assert app.conversation[0]["role"] == "system"
            assert app._chat_id is not None
            assert find_chat_by_id(app._chat_id) is not None
            active = find_memory_by_id(get_active_memory_id())
            assert active is not None
            assert active["name"] == "general"

    asyncio.run(exercise())


def test_action_new_chat_keeps_previous_chat(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            first_id = app._chat_id
            app.conversation.append({"role": "user", "content": "hello"})
            app._transcript.append({"role": "user", "content": "hello"})
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert app._chat_id != first_id
            assert len(app.conversation) == 1
            assert app.conversation[0]["role"] == "system"
            # The previous chat was kept, not deleted.
            assert find_chat_by_id(first_id) is not None
            assert load_chat_index().get(app._chat_id) is not None

    asyncio.run(exercise())


def test_chat_saved_after_turn(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            yield "final reply"

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()
            data = load_latest_chat()
            assert data is not None
            joined = " ".join(
                str(m.get("content")) for m in data["context_messages"]
            )
            assert "final reply" in joined
            transcript_joined = " ".join(
                str(m.get("content")) for m in data["transcript"]
            )
            assert "final reply" in transcript_joined

    asyncio.run(exercise())


def test_memory_tool_add_refreshes_system_prompt(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield 'tool: memory({"action": "add", "text": "remember X"})'
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("save it")
            await pilot.pause()
            assert "remember X" in app.conversation[0]["content"]

    asyncio.run(exercise())


def test_memory_add_without_name_uses_active_memory(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield 'tool: memory({"action": "add", "text": "remember Y"})'
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            active_memory_id = get_active_memory_id()
            await app.run_agent_turn("refactor the parser module")
            await pilot.pause()

            # The note goes to the still-active durable memory...
            active = get_active_memory_id()
            assert active == active_memory_id
            assert find_memory_by_id(active)["name"] == "general"
            assert "remember Y" in memory_tool("read")["content"]
            assert "remember Y" in app.conversation[0]["content"]
            # ...while the chat itself is auto-named after the task.
            chat = find_chat_by_id(app._chat_id)
            assert chat is not None
            assert chat["name"] == "refactor the parser"
            log_lines = [strip.text for strip in app.query_one("#log").lines]
            assert not any("auto-named" in line for line in log_lines)

    asyncio.run(exercise())


def test_named_memory_add_is_not_renamed(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield (
                    'tool: memory({"action": "add", "text": "design fact", '
                    '"name": "design"})'
                )
            else:
                yield "done"

        monkeypatch.setattr(tui_app, "stream_llm_call", tool_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("refactor the parser module")
            await pilot.pause()

            # Explicitly named notes remain isolated from the active memory,
            # which keeps its default name now that chats are named instead.
            assert find_memory_by_id(get_active_memory_id())["name"] == "general"
            assert "design fact" in memory_tool("read", name="design")["content"]

    asyncio.run(exercise())


def test_open_memory_action_opens_picker(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_memory()
            await pilot.pause()
            assert len(app.screen_stack) == 2
            assert isinstance(app.screen, MemoryScreen)
            # Memories are picked from a dropdown, not an option list.
            assert app.screen.query_one("#memory-select", Select)
            assert not app.screen.query("#memory-list")
            assert not app.screen.query("#memory-switch")
            assert not app.screen.query("#new-memory-input")
            # Closing is done via the ✕ button in the dialog header.
            assert not app.screen.query("#memory-cancel")
            assert app.screen.query_one("#memory-close", Button)
            assert app.screen.query_one("#memory-delete")

    asyncio.run(exercise())


def test_memory_picker_switch_updates_active(monkeypatch):
    async def exercise():
        memory_tool("add", "base note")  # activates 'general'
        design = memory_tool("add", "design fact", name="design")

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_memory()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MemoryScreen)
            assert get_active_memory_id() != design["id"]
            memory_select = screen.query_one("#memory-select", Select)
            memory_select.value = design["id"]
            await pilot.pause()
            assert get_active_memory_id() == design["id"]
            assert "design fact" in app.conversation[0]["content"]

    asyncio.run(exercise())


def test_memory_picker_selects_active_memory(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_memory()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MemoryScreen)
            active_memory = find_memory_by_id(get_active_memory_id())
            assert active_memory is not None
            assert active_memory["name"] == "general"
            memory_select = screen.query_one("#memory-select", Select)
            assert memory_select.value == active_memory["id"]

    asyncio.run(exercise())

def test_memory_picker_switch_to_existing_memory(monkeypatch):
    async def exercise():
        notes = memory_tool("add", "standalone notes", name="notes")

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_memory()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MemoryScreen)
            screen._switch(notes["id"])
            await pilot.pause()
            assert get_active_memory_id() == notes["id"]
            assert find_memory_by_id(notes["id"])["name"] == "notes"
            assert "standalone notes" in memory_tool("read", name="notes")["content"]
            assert len(app.screen_stack) == 1

    asyncio.run(exercise())


def test_memory_picker_deletes_memory(monkeypatch):
    async def exercise():
        design = memory_tool("add", "to be deleted", name="design")

        app = AgentApp()
        async with app.run_test() as pilot:
            set_active_memory_id(design["id"])
            app._refresh_system_prompt()
            await app.action_open_memory()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MemoryScreen)

            # Delete is immediate and does not open a confirmation modal.
            screen.query_one("#memory-delete", Button).press()
            await pilot.pause()
            await pilot.pause()

            assert find_memory_by_id(design["id"]) is None
            memory_select = screen.query_one("#memory-select", Select)
            assert memory_select.value != design["id"]
            # Deleting the active memory falls back to 'general'.
            assert find_memory_by_id(get_active_memory_id())["name"] == "general"

    asyncio.run(exercise())


def test_memory_picker_delete_is_immediate(monkeypatch):
    async def exercise():
        design = memory_tool("add", "keep me", name="design")

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_memory()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MemoryScreen)

            memory_select = screen.query_one("#memory-select", Select)
            memory_select.value = design["id"]
            await pilot.pause()
            await screen._delete_current()
            await pilot.pause()

            assert find_memory_by_id(design["id"]) is None

    asyncio.run(exercise())


def test_memory_picker_blocked_while_agent_running(monkeypatch):
    async def exercise():
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            while True:
                yield "chunk"
                await asyncio.sleep(0)

        monkeypatch.setattr(tui_app, "stream_llm_call", endless_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._agent_running = True
            await app.action_open_memory()
            await pilot.pause()
            assert len(app.screen_stack) == 1

    asyncio.run(exercise())


def test_open_chats_action_opens_picker(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_chats()
            await pilot.pause()
            assert len(app.screen_stack) == 2
            assert isinstance(app.screen, ChatScreen)
            assert app.screen.query_one("#chat-list", OptionList)
            assert app.screen.query_one("#chat-new", Button)
            assert app.screen.query_one("#chat-switch", Button)
            assert app.screen.query_one("#chat-delete", Button)
            assert app.screen.query_one("#chat-cancel", Button)

    asyncio.run(exercise())


def test_chat_picker_blocked_while_agent_running(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app._agent_running = True
            await app.action_open_chats()
            await pilot.pause()
            assert len(app.screen_stack) == 1

    asyncio.run(exercise())


def test_chat_picker_switch_loads_chat(monkeypatch):
    async def exercise():
        first = create_chat("first chat")
        save_chat(
            first["id"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
        )
        second = create_chat("second chat")
        save_chat(
            second["id"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "second question"},
            ],
            [{"role": "user", "content": "second question"}],
        )
        chats = load_chat_index()
        chats[first["id"]]["updated_at"] = "2026-01-01T10:00:00"
        chats[second["id"]]["updated_at"] = "2026-01-02T10:00:00"
        save_chat_index(chats)

        app = AgentApp()
        async with app.run_test() as pilot:
            # Latest chat resumes on launch.
            assert app._chat_id == second["id"]
            await app.action_open_chats()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._switch(first["id"])
            await pilot.pause()
            assert len(app.screen_stack) == 1
            assert app._chat_id == first["id"]
            assert len(app.conversation) == 3
            assert app._transcript[0]["content"] == "first question"
            log_lines = [strip.text for strip in app.query_one("#log").lines]
            assert any("Switched to chat" in line for line in log_lines)
            assert any("first answer" in line for line in log_lines)

    asyncio.run(exercise())


def test_chat_picker_new_starts_fresh_chat(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            original_id = app._chat_id
            app.conversation.append({"role": "user", "content": "hi"})
            app._transcript.append({"role": "user", "content": "hi"})
            await app.action_open_chats()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen.query_one("#chat-new", Button).press()
            await pilot.pause()
            assert len(app.screen_stack) == 1
            assert app._chat_id != original_id
            assert len(app.conversation) == 1
            assert find_chat_by_id(original_id) is not None

    asyncio.run(exercise())


def test_chat_picker_delete_requires_confirmation(monkeypatch):
    async def exercise():
        kept = create_chat("kept chat")
        doomed = create_chat("doomed chat")

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_chats()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            chat_list = screen.query_one("#chat-list", OptionList)
            chat_list.highlighted = chat_list.get_option_index(doomed["id"])

            delete_button = screen.query_one("#chat-delete", Button)
            delete_button.press()
            await pilot.pause()
            # First press only arms the button.
            assert find_chat_by_id(doomed["id"]) is not None
            assert delete_button.label == "Really delete?"

            delete_button.press()
            await pilot.pause()
            assert find_chat_by_id(doomed["id"]) is None
            remaining_ids = [option.id for option in chat_list.options]
            assert doomed["id"] not in remaining_ids
            assert kept["id"] in remaining_ids

    asyncio.run(exercise())


def test_deleting_active_chat_loads_next_latest(monkeypatch):
    async def exercise():
        older = create_chat("older chat")
        save_chat(
            older["id"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "older"},
            ],
            [{"role": "user", "content": "older"}],
        )
        newer = create_chat("newer chat")
        save_chat(
            newer["id"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "newer"},
            ],
            [{"role": "user", "content": "newer"}],
        )
        chats = load_chat_index()
        chats[older["id"]]["updated_at"] = "2026-01-01T10:00:00"
        chats[newer["id"]]["updated_at"] = "2026-01-02T10:00:00"
        save_chat_index(chats)

        app = AgentApp()
        async with app.run_test() as pilot:
            # The newest chat resumes on launch.
            assert app._chat_id == newer["id"]
            await app.action_open_chats()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            chat_list = screen.query_one("#chat-list", OptionList)
            chat_list.highlighted = chat_list.get_option_index(newer["id"])
            delete_button = screen.query_one("#chat-delete", Button)
            delete_button.press()
            await pilot.pause()
            delete_button.press()
            await pilot.pause()
            assert find_chat_by_id(newer["id"]) is None
            assert app._chat_id == older["id"]
            assert app.conversation[-1]["content"] == "older"

    asyncio.run(exercise())


def test_context_full_error_notifies_clearly(monkeypatch):
    async def exercise():
        async def failing_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            raise LLMRequestError(
                400,
                "maximum context length is 131072 tokens",
            )
            yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", failing_stream)
        notifications = []

        app = AgentApp()
        monkeypatch.setattr(app, "notify", lambda *a, **k: notifications.append(k))
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert notifications
            assert notifications[0].get("title") == "Context window full"

    asyncio.run(exercise())


def test_prompt_ctrl_a_selects_all_text():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.load_text("select all of this")
            prompt.focus()
            await pilot.press("ctrl+a")
            assert prompt.selected_text == "select all of this"

    asyncio.run(exercise())


def test_prompt_grows_and_shrinks_with_multiline_input():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()
            initial_height = prompt.outer_size.height
            prompt.load_text("one\ntwo\nthree")
            await pilot.pause()
            assert prompt.outer_size.height == initial_height + 2
            prompt.load_text("")
            await pilot.pause()
            assert prompt.outer_size.height == initial_height

    asyncio.run(exercise())


def test_prompt_enter_submits_and_ctrl_j_inserts_newline(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()
            await pilot.press("a")
            await pilot.press("ctrl+j")
            await pilot.press("b")
            assert prompt.text == "a\nb"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            user_contents = [
                m["content"] for m in app.conversation if m["role"] == "user"
            ]
            assert user_contents == ["a\nb"]
            assert prompt.text == ""

    asyncio.run(exercise())


def test_prompt_shift_enter_inserts_newline(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()
            await pilot.press("x")
            await pilot.press("shift+enter")
            await pilot.press("y")
            assert prompt.text == "x\ny"

    asyncio.run(exercise())


def test_prompt_history_up_and_down(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()

            await pilot.press("a")
            await pilot.press("a")
            await pilot.press("a")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.press("b")
            await pilot.press("b")
            await pilot.press("b")
            await pilot.press("enter")
            await pilot.pause()

            assert app._prompt_history == ["aaa", "bbb"]

            await pilot.press("up")
            assert prompt.text == "bbb"
            await pilot.press("up")
            assert prompt.text == "aaa"
            await pilot.press("down")
            assert prompt.text == "bbb"
            await pilot.press("down")
            assert prompt.text == ""

    asyncio.run(exercise())


def test_prompt_history_down_past_end_restores_draft(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            app._prompt_history = ["hello", "world"]
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()
            await pilot.press("d")
            await pilot.press("r")
            await pilot.press("a")
            await pilot.press("f")
            await pilot.press("t")

            await pilot.press("up")
            assert prompt.text == "world"
            await pilot.press("up")
            assert prompt.text == "hello"
            await pilot.press("down")
            assert prompt.text == "world"
            await pilot.press("down")
            assert prompt.text == "draft"

    asyncio.run(exercise())


def test_prompt_history_skips_consecutive_duplicates(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()
            await pilot.press("a")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("a")
            await pilot.press("enter")
            await pilot.pause()
            assert app._prompt_history == ["a"]

    asyncio.run(exercise())


def test_up_arrow_moves_lines_before_history(monkeypatch):
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app._prompt_history = ["saved"]
            prompt = app.query_one("#prompt", PromptTextArea)
            prompt.focus()
            await pilot.press("l")
            await pilot.press("i")
            await pilot.press("n")
            await pilot.press("e")
            await pilot.press("1")
            await pilot.press("ctrl+j")
            await pilot.press("l")
            await pilot.press("i")
            await pilot.press("n")
            await pilot.press("e")
            await pilot.press("2")

            assert prompt.cursor_location[0] == 1
            await pilot.press("up")
            assert prompt.cursor_location[0] == 0
            assert prompt.text == "line1\nline2"

            await pilot.press("up")
            assert prompt.text == "saved"

    asyncio.run(exercise())


def test_paste_clipboard_image_attaches(monkeypatch):
    async def exercise():
        image = Image.new("RGB", (8, 8), "blue")
        monkeypatch.setattr(tui.ImageGrab, "grabclipboard", lambda: image)

        app = AgentApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", PromptTextArea)
            assert prompt._paste_clipboard_image() is True
            assert app._pending_images == [image]

    asyncio.run(exercise())


def test_image_attachment_builds_multimodal_message(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs):
            if False:
                yield ""

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        images = [
            Image.new("RGB", (8, 8), "red"),
            Image.new("RGB", (8, 8), "blue"),
        ]
        for image in images:
            app.add_pending_image(image)
        async with app.run_test() as pilot:
            app.on_prompt_submitted(PromptSubmitted("what is this"))
            await pilot.pause()
            await pilot.pause()

            user_content = [
                m["content"]
                for m in app.conversation
                if m["role"] == "user"
            ][-1]
            assert isinstance(user_content, list)
            assert [part["type"] for part in user_content] == [
                "text",
                "image_url",
                "image_url",
            ]
            assert user_content[0]["text"] == "what is this"
            assert all(
                part["image_url"]["url"].startswith("data:image/png;base64,")
                for part in user_content[1:]
            )
            assert app._pending_images == []

    asyncio.run(exercise())


def test_model_badge_shows_token_usage():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL, "key", "kimi-k3", reasoning_effort="off"
        )
    )
    badge.set_tokens(1234, 5678)
    assert badge.render().plain == "Kimi K3  OpenCode Go · 6.9k tok"


def test_model_badge_hides_usage_when_zero():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig("http://localhost/v1", "key", "local", reasoning_effort="off")
    )
    assert badge.render().plain == "Local  Local"


def test_model_badge_shows_reasoning_effort():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL,
            "key",
            "kimi-k3",
            reasoning_effort="max",
        )
    )
    assert badge.render().plain == "Kimi K3  OpenCode Go · effort max"


def test_live_generated_tokens_shown_and_cleared():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL,
            "key",
            "kimi-k3",
            reasoning_effort="off",
        )
    )
    badge.set_live_generated_tokens(0)
    assert "0 generated" in badge.render().plain
    badge.set_live_generated_tokens(1234)
    assert "1.2k generated" in badge.render().plain
    badge.set_live_generated_tokens(None)
    assert "generated" not in badge.render().plain


def test_token_speed_shown_and_cleared():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL,
            "key",
            "kimi-k3",
            reasoning_effort="off",
        )
    )
    badge.set_speed(12.5)
    assert "12.5 tok/s" in badge.render().plain
    badge.set_speed(None)
    assert "tok/s" not in badge.render().plain


def test_connection_screen_has_submit_and_cancel_only():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_connection()
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#submit-button")
            assert screen.query_one("#cancel-button")
            assert not screen.query("#refresh-button")

    asyncio.run(exercise())


def test_reasoning_effort_fades_for_unsupported_model(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["minimax-m3", "glm-5.2"]

        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)
        previous = get_config()
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "key",
            "minimax-m3",
            provider="opencode-go",
            reasoning_effort="high",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                effort = screen.query_one("#reasoning-effort-select", EffortSlider)
                label = screen.query_one("#reasoning-effort-label")
                assert effort.value == "off"
                assert effort.disabled is True
                assert label.disabled is True
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_reasoning_effort_restores_on_switch_back(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["minimax-m3", "glm-5.2"]

        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)
        previous = get_config()
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "key",
            "minimax-m3",
            provider="opencode-go",
            reasoning_effort="medium",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                effort = screen.query_one("#reasoning-effort-select", EffortSlider)
                # Initial state: unsupported model -> faded at "off".
                assert effort.value == "off"
                assert effort.disabled is True

                assert screen.query_one("#model-select").display is True
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_local_keeps_reasoning_effort_enabled(monkeypatch):
    async def exercise():
        previous = get_config()
        configure_openai(
            "http://localhost:7070/v1",
            "key",
            "minimax-m3",
            provider="local",
            reasoning_effort="off",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                effort = screen.query_one("#reasoning-effort-select", EffortSlider)
                assert effort.disabled is False
                assert screen.query_one("#reasoning-effort-label").disabled is False
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_connect_clamps_effort_for_unsupported_model(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["grok-4.5", "glm-5.2"]

        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)
        previous = get_config()
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "key",
            "grok-4.5",
            provider="opencode-go",
            reasoning_effort="high",
        )
        saved = []
        monkeypatch.setattr(
            connection_screen,
            "save_provider_configs",
            lambda profiles, active: saved.append(profiles[active]),
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                effort = screen.query_one("#reasoning-effort-select", EffortSlider)
                assert effort.value == "off"
                screen._connect()
                await pilot.pause()
                assert saved
                assert saved[-1].reasoning_effort == "off"
                assert saved[-1].model == "grok-4.5"
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_connection_screen_restores_provider_and_effort(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["kimi-k3"]

        # The app prefetches the live model list on mount; keep it offline.
        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)

        previous = get_config()
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "key",
            "kimi-k3",
            provider="opencode-go",
            reasoning_effort="high",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                provider_select = screen.query_one("#provider-select", Select)
                assert provider_select.value == "opencode-go"
                assert screen.query_one("#reasoning-effort-select").value == "high"
                assert screen.query_one("#base-url-input").disabled is True
                assert screen.query_one("#base-url-input").display is False
                assert screen.query_one("#base-url-label").display is False
                assert screen.query_one("#model-select").disabled is False
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_connection_screen_preserves_saved_model_not_in_bundled_list(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["kimi-k3", "live-only-model"]

        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)

        previous = get_config()
        # Empty API key so no live refresh fires; the picker must still show
        # the saved model even though it is not in the bundled fallback list.
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "",
            "saved-custom-model",
            provider="opencode-go",
            reasoning_effort="medium",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                select = screen.query_one("#model-select")
                options = [value for _, value in select._options]
                assert "saved-custom-model" in options
                assert select.value == "saved-custom-model"
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_connection_screen_refresh_keeps_selected_live_model(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["kimi-k3", "live-only-model"]

        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)

        previous = get_config()
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "key",
            "kimi-k3",
            provider="opencode-go",
            reasoning_effort="medium",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                select = screen.query_one("#model-select")
                options = [value for _, value in select._options]
                # The live refresh must not clobber the user's selected model
                # when it is still offered by the live list.
                assert "kimi-k3" in options
                assert "live-only-model" in options
                assert select.value == "kimi-k3"
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_startup_prefetch_populates_context_cache(monkeypatch):
    async def exercise():
        import remie.agent as agent

        fetched = []

        async def fake_fetch(api_key):
            fetched.append(api_key)
            return ["kimi-k3"]

        monkeypatch.setattr(tui_app, "fetch_opencode_go_models", fake_fetch)
        monkeypatch.setattr(agent, "_opencode_go_model_context", {"kimi-k3": 256_000})

        previous = get_config()
        configure_openai(
            OPENCODE_GO_BASE_URL,
            "key",
            "kimi-k3",
            provider="opencode-go",
            reasoning_effort="off",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                for _ in range(50):
                    if fetched:
                        break
                    await pilot.pause()
                assert fetched == ["key"]
                assert (
                    agent.get_model_context_limit("kimi-k3", "opencode-go") == 256_000
                )
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_startup_prefetch_skips_local_provider(monkeypatch):
    async def exercise():
        fetched = []

        async def fake_fetch(api_key):
            fetched.append(api_key)
            return ["kimi-k3"]

        monkeypatch.setattr(tui_app, "fetch_opencode_go_models", fake_fetch)

        previous = get_config()
        configure_openai(
            "http://localhost:7070/v1",
            "key",
            "local-model",
            provider="local",
            reasoning_effort="off",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.pause()
                assert fetched == []
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_connection_screen_shows_local_url_field():
    async def exercise():
        previous = get_config()
        configure_openai(
            "http://localhost:7070/v1",
            "key",
            "local-model",
            provider="local",
            reasoning_effort="off",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "local"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "local")
                )
                assert screen.query_one("#base-url-input").display is True
                assert screen.query_one("#base-url-label").display is True
                assert screen.query_one("#base-url-input").value == (
                    "http://localhost:7070/v1"
                )
                assert screen.query_one("#model-select").disabled is True
                assert screen.query_one("#local-model-input").display is True
                assert screen.query_one("#local-model-input").value == "local-model"
                assert screen.query_one("#verify-ssl-label").display is True
                assert screen.query_one("#verify-ssl-radio", RadioSet).display is True
                assert screen.query_one("#verify-ssl-true", RadioButton).value is False
                assert screen.query_one("#verify-ssl-false", RadioButton).value is True
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_connection_screen_preserves_each_provider_form(monkeypatch):
    async def exercise():
        async def fake_fetch(api_key):
            return ["gpt-test"]

        monkeypatch.setattr(connection_screen, "fetch_opencode_go_models", fake_fetch)
        previous = get_config()
        configure_openai(
            "http://localhost:7070/v1",
            "local-key",
            "local-model",
            provider="local",
            reasoning_effort="medium",
        )
        try:
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                local_input = screen.query_one("#local-model-input", Input)
                local_input.value = "custom-local-model"

                provider_select = screen.query_one("#provider-select", Select)
                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                screen.query_one("#api-key-input", Input).value = "go-key"
                provider_select.value = "local"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "local")
                )
                assert local_input.value == "custom-local-model"

                provider_select.value = "opencode-go"
                await screen.on_select_changed(
                    Select.Changed(provider_select, "opencode-go")
                )
                assert screen.query_one("#api-key-input", Input).value == "go-key"
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
                previous.verify_ssl,
            )

    asyncio.run(exercise())


def test_reasoning_stream_keeps_timer_rendered_prefix(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            assert reasoning_box is not None
            reasoning_box.append("timer prefix ")
            # No content is yielded during this pause, so the timer owns the
            # first render. The following content delta must retain that text.
            await asyncio.sleep(0.15)
            reasoning_box.append("content suffix")
            yield "reply"

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            joined = "\n".join(strip.text for strip in app.query_one("#log").lines)
            assert "timer prefix content suffix" in joined

    asyncio.run(exercise())


def test_reasoning_stream_coalesces_same_tick_renders(monkeypatch):
    """When the reasoning timer paints the panel and a content delta arrives
    within the coalesce window (with no new reasoning), the inline path must
    skip its duplicate update_stream call."""
    async def exercise():
        from remie.tui.widgets import StreamingRichLog

        calls = {"n": 0}
        real_update = StreamingRichLog.update_stream

        def counting_update(self, *args, **kwargs):
            calls["n"] += 1
            return real_update(self, *args, **kwargs)

        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            assert reasoning_box is not None
            reasoning_box.append("timer renders this ")
            await asyncio.sleep(0.08)  # timer paints at least once
            # Content deltas arrive with no fresh reasoning; each would
            # trigger an inline render without coalescing.
            for _ in range(5):
                reasoning_box.append("x")
                yield "a"
                await asyncio.sleep(0.005)

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)
        monkeypatch.setattr(StreamingRichLog, "update_stream", counting_update)
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()
            # Without coalescing every content delta adds a render on top of
            # the timer's. With coalescing the total stays modest.
            assert calls["n"] < 8
            joined = "\n".join(strip.text for strip in app.query_one("#log").lines)
            assert "timer renders this" in joined

    asyncio.run(exercise())


def test_reasoning_timer_stops_when_stream_errors(monkeypatch):
    async def exercise():
        async def failing_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            assert reasoning_box is not None
            reasoning_box.append("thinking")
            await asyncio.sleep(0.15)
            raise RuntimeError("stream failed")
            yield

        monkeypatch.setattr(tui_app, "stream_llm_call", failing_stream)
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()
            assert app._live_stream["active"] is False
            assert app._live_reasoning_timer is None

    asyncio.run(exercise())


def test_reasoning_only_stream_updates_live_generated_counter(monkeypatch):
    async def exercise():
        async def reasoning_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            reasoning_box.append("r" * 400)
            await asyncio.sleep(0.15)
            yield "reply"

        monkeypatch.setattr(tui_app, "stream_llm_call", reasoning_stream)
        app = AgentApp()
        async with app.run_test() as pilot:
            task = asyncio.create_task(app.run_agent_turn("hello"))
            await asyncio.sleep(0.1)
            assert app.query_one(ModelBadge)._live_generated_tokens >= 100
            await task
            await pilot.pause()

    asyncio.run(exercise())


def test_token_usage_persists_and_restores_with_chat(monkeypatch):
    """Per-chat cumulative usage survives save/reload and a fresh launch."""

    async def fake_stream(
        _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
    ):
        usage_box["prompt_tokens"] = 100
        usage_box["completion_tokens"] = 50
        yield "reply"

    async def exercise():
        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            saved = load_chat(app._chat_id)
            assert saved["token_usage"] == {
                "input_tokens": 100,
                "output_tokens": 50,
            }

            # Switching away and back keeps this chat's own totals.
            assert app._load_chat_into_ui(app._chat_id) is True
            await pilot.pause()
            assert app._total_input_tokens == 100
            assert app._total_output_tokens == 50

        # A later launch resumes the latest chat with the saved totals.
        resumed = AgentApp()
        async with resumed.run_test() as pilot:
            await pilot.pause()
            assert resumed._total_input_tokens == 100
            assert resumed._total_output_tokens == 50

    asyncio.run(exercise())


def test_load_chat_without_token_usage_defaults_to_zero():
    """Chats saved before usage tracking load with zeroed totals."""
    chat = create_chat()
    try:
        save_chat(
            chat["id"],
            [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
            [{"role": "user", "content": "hi"}],
        )
        loaded = load_chat(chat["id"])
        assert loaded["token_usage"] == {"input_tokens": 0, "output_tokens": 0}
    finally:
        delete_chat(chat["id"])


def test_turn_updates_badge_tokens(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            if reasoning_box is not None:
                reasoning_box.append("reasoning text")
            usage_box["prompt_tokens"] = 100
            usage_box["completion_tokens"] = 50
            yield "reply"

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            badge = app.query_one(ModelBadge)
            assert "tok" in badge.render().plain
            assert "50 generated" in badge.render().plain
            assert app._total_input_tokens == 100
            assert app._total_output_tokens == 50

    asyncio.run(exercise())


def test_empty_response_is_retried_and_completes(monkeypatch):
    async def exercise():
        calls = 0

        async def empty_then_reply(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                return
            yield "reply"

        monkeypatch.setattr(tui_app, "stream_llm_call", empty_then_reply)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert calls == 2
            assert app._agent_running is False
            assistant_msgs = [
                m["content"] for m in app.conversation if m["role"] == "assistant"
            ]
            assert assistant_msgs == ["(no output)", "reply"]

    asyncio.run(exercise())


def test_empty_response_gives_up_with_notice(monkeypatch):
    async def exercise():
        async def always_empty(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            if reasoning_box is not None:
                reasoning_box.append("reasoning only")
            return
            yield

        monkeypatch.setattr(tui_app, "stream_llm_call", always_empty)

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert app._agent_running is False
            log_lines = [
                strip.text for strip in app.query_one("#log").lines
            ]
            joined = "\n".join(log_lines)
            assert "Agent stopped: empty response" in joined
            assistant_msgs = [
                m["content"] for m in app.conversation if m["role"] == "assistant"
            ]
            assert assistant_msgs == ["reasoning only", "reasoning only"]

    asyncio.run(exercise())


def test_empty_response_stops_when_user_stops(monkeypatch):
    async def exercise():
        async def slow_empty(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            await asyncio.sleep(0.1)
            return
            yield

        monkeypatch.setattr(tui_app, "stream_llm_call", slow_empty)

        app = AgentApp()
        async with app.run_test() as pilot:
            task = asyncio.create_task(app.run_agent_turn("hello"))
            await asyncio.sleep(0.15)
            app.action_stop_agent()
            await task
            await pilot.pause()

            assert app._agent_running is False
            log_lines = [
                strip.text for strip in app.query_one("#log").lines
            ]
            joined = "\n".join(log_lines)
            assert "Stopped by user" in joined

    asyncio.run(exercise())


def test_plain_write_exposes_plain_text():
    inner = tui._make_syntax("x = 1", "python", "ansi_dark")
    wrapped = tui._PlainWrite("x = 1", inner)
    assert wrapped.plain == "x = 1"


def test_make_syntax_highlights_with_lexer():
    syntax = tui._make_syntax('def f():\n    return 1', "python", "ansi_dark")
    assert isinstance(syntax, Syntax)
    assert syntax.lexer.name == "Python"


def test_make_syntax_falls_back_to_plain_text():
    renderable = tui._make_syntax("x", "no-such-lexer", "ansi_dark")
    assert isinstance(renderable, Text)
    assert renderable.plain == "x"


def test_guess_lexer_name_known_and_unknown():
    assert tui._guess_lexer_name("main.py") == "Python"
    assert tui._guess_lexer_name("script.js") == "JavaScript"
    assert tui._guess_lexer_name("README.md") == "Markdown"
    assert tui._guess_lexer_name("file.xyzunknown") is None


def test_command_output_lexer_detection():
    assert tui._command_output_lexer('{"a": 1, "b": [2, 3]}') == "json"
    assert tui._command_output_lexer('[1, 2, 3]') == "json"
    assert tui._command_output_lexer(
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b"
    ) == "diff"
    assert (
        tui._command_output_lexer(
            'Traceback (most recent call last):\n  File "main.py", line 1\n    x'
        )
        == "pytb"
    )
    assert tui._command_output_lexer("just some plain text") is None
    assert tui._command_output_lexer("") is None
    assert tui._command_output_lexer(
        'not json {"unterminated'
    ) is None


def test_read_file_result_is_highlighted_python(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def f():\n    return 'x'\n", encoding="utf-8")
    result = {"file_path": str(path), "content": path.read_text(encoding="utf-8")}
    panel = tui._render_read_file_result(result, "ansi_dark")
    assert panel.title == "Tool result · read_file"
    renderable = panel.renderable
    assert isinstance(renderable, Group)
    # The body is wrapped by _PlainWrite and keeps plain text extractable.
    body = list(renderable.renderables)[-1]
    assert isinstance(body, tui._PlainWrite)
    assert "def f():" in body.plain


def test_run_command_json_output_is_highlighted():
    result = {
        "exit_code": 0,
        "stdout": '{"status": "ok", "count": 3}',
        "stderr": "",
        "timed_out": False,
    }
    panel = tui._render_run_command_result(result, "ansi_dark")
    rendered = panel.renderable
    assert isinstance(rendered, Group)
    body = list(rendered.renderables)[-1]
    assert isinstance(body, tui._PlainWrite)
    assert body.plain == '{\n  "status": "ok",\n  "count": 3\n}'


def test_run_command_plain_output_is_plain_text():
    result = {
        "exit_code": 0,
        "stdout": "hello\nworld\n",
        "stderr": "",
        "timed_out": False,
    }
    panel = tui._render_run_command_result(result, "ansi_dark")
    assert isinstance(panel.renderable, Text)
    assert "hello\nworld" in panel.renderable.plain


def test_render_tool_result_dispatch_highlighted():
    panel = tui._render_tool_result(
        "read_file", {"file_path": "a.py", "content": "x = 1\n"}
    )
    assert panel is not None
    assert panel.title == "Tool result · read_file"

    command_panel = tui._render_tool_result(
        "run_command",
        {"exit_code": 0, "stdout": "[1, 2]", "stderr": "", "timed_out": False},
    )
    assert command_panel is not None


def test_final_answer_strips_protocol_lines(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None, **_kwargs
        ):
            yield "thinking: let me think"
            yield "\n"
            yield "Here is the answer."

        monkeypatch.setattr(tui_app, "stream_llm_call", fake_stream)
        captured = {}
        monkeypatch.setattr(
            tui_app,
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


def test_connection_screen_shows_codex_sign_in_when_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tui.codex_auth, "auth_json_path", lambda: tmp_path / "auth.json"
    )
    previous = get_config()
    import remie.agent as _agent

    _agent.configure_openai("http://localhost:7070/v1", "k", "m", provider="local")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_connection()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConnectionScreen)

            provider = screen.query_one("#provider-select", Select)
            values = []
            for option in provider._options:
                values.append(
                    option.value if hasattr(option, "value") else option[1]
                )
            assert "codex" in values

            signin = screen.query_one("#codex-signin-button", Button)
            api_key_input = screen.query_one("#api-key-input", Input)
            account_label = screen.query_one("#codex-account-label", Label)

            # Hidden while a different provider is active.
            assert not signin.display
            assert api_key_input.display

            provider.value = "codex"
            await pilot.pause()

            assert signin.display
            assert screen.query_one("#codex-signout-button", Button).display
            assert not api_key_input.display
            assert "Not signed in" in str(account_label.render())

    asyncio.run(exercise())
    _agent.configure_openai(
        previous.base_url,
        previous.api_key,
        previous.model,
        previous.provider,
        previous.reasoning_effort,
        previous.verify_ssl,
    )


def test_connection_screen_codex_connect_requires_sign_in(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tui.codex_auth, "auth_json_path", lambda: tmp_path / "auth.json"
    )
    previous = get_config()
    import remie.agent as _agent

    _agent.configure_openai("http://localhost:7070/v1", "k", "m", provider="local")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_connection()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConnectionScreen)
            screen.query_one("#provider-select", Select).value = "codex"
            await pilot.pause()

            notifications: list[str] = []
            monkeypatch.setattr(
                app,
                "notify",
                lambda *args, **kwargs: notifications.append(args[0]),
            )
            screen._connect()
            await pilot.pause()

            assert any("Sign in with ChatGPT" in note for note in notifications)
            # Still on the connection screen; nothing was connected.
            assert isinstance(app.screen, ConnectionScreen)
            assert get_config().provider != "codex"

    asyncio.run(exercise())
    _agent.configure_openai(
        previous.base_url,
        previous.api_key,
        previous.model,
        previous.provider,
        previous.reasoning_effort,
        previous.verify_ssl,
    )


def test_connection_screen_codex_connects_when_signed_in(monkeypatch, tmp_path):
    import time as _time

    from remie import codex_auth as _codex_auth

    auth_file = tmp_path / "auth.json"
    monkeypatch.setattr(tui.codex_auth, "auth_json_path", lambda: auth_file)

    def make_jwt(claims):
        segment = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
        return f"e.{segment.decode()}.s"

    token = make_jwt(
        {
            "exp": _time.time() + 3600,
            _codex_auth.OPENAI_AUTH_CLAIM: {
                "chatgpt_account_id": "acc_1",
                "chatgpt_plan_type": "pro",
            },
        }
    )
    auth_file.write_text(
        json.dumps(
            {"tokens": {"access_token": token, "refresh_token": "r", "id_token": ""}}
        )
    )
    previous = get_config()
    try:

        async def exercise():
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)
                screen.query_one("#provider-select", Select).value = "codex"
                await pilot.pause()
                account_label = screen.query_one("#codex-account-label", Label)
                assert "Signed in" in str(account_label.render())

                screen._connect()
                await pilot.pause()

                config = get_config()
                assert config.provider == "codex"
                assert config.model == tui.CODEX_MODELS[0]
                badge = app.query_one(ModelBadge)
                assert "Codex (ChatGPT)" in str(badge.render())

        asyncio.run(exercise())
    finally:
        from remie.agent import configure_openai as _configure

        _configure(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def test_codex_native_tool_calls_execute_and_replay(monkeypatch):
    """Codex provider: function_call items are executed and results replay as
    role='tool' messages, then the loop continues to the final answer."""
    import remie.agent as agent

    previous = get_config()
    try:
        agent.configure_openai(
            tui.CODEX_BACKEND_BASE, "", "gpt-5.5", provider="codex"
        )

        stream_rounds = {"n": 0}

        async def scripted_stream(
            _conversation,
            usage_box=None,
            reasoning_box=None,
            finish_box=None,
            **_kwargs,
        ):
            stream_rounds["n"] += 1
            if stream_rounds["n"] == 1:
                box = _kwargs.get("tool_calls_box")
                assert box is not None
                box.append(
                    {
                        "id": "call_1",
                        "name": "list_files",
                        "arguments": '{"path": "."}',
                    }
                )
                yield ""  # no prose alongside the call
            else:
                finish_box["finish_reason"] = "stop"
                yield "All done."

        monkeypatch.setattr(tui_app, "stream_llm_call", scripted_stream)
        # Title generation runs through agent.stream_llm_call directly; keep
        # the Codex-provider test offline.
        async def fake_title(_messages):
            return "a finished task"

        monkeypatch.setattr(tui_app, "generate_chat_title", fake_title)
        tool_calls_seen = []

        def fake_run_tool(name, args):
            tool_calls_seen.append((name, args))
            return {"path": ".", "files": [{"filename": "main.py", "type": "file"}]}

        monkeypatch.setattr(tui_app, "execute_tool_call", fake_run_tool)

        async def exercise():
            app = AgentApp()
            async with app.run_test() as pilot:
                task = asyncio.create_task(app.run_agent_turn("list the files"))
                await task
                await pilot.pause()

                roles = [m["role"] for m in app.conversation]
                assert roles == [
                    "system",
                    "user",
                    "assistant",
                    "tool",
                    "assistant",
                ]
                call_msg = app.conversation[2]
                assert call_msg["tool_calls"] == [
                    {
                        "id": "call_1",
                        "name": "list_files",
                        "arguments": '{"path": "."}',
                    }
                ]
                result_msg = app.conversation[3]
                assert result_msg["role"] == "tool"
                assert result_msg["tool_call_id"] == "call_1"
                assert result_msg["name"] == "list_files"
                assert '"files"' in result_msg["content"]
                assert app.conversation[4]["content"] == "All done."
                assert tool_calls_seen == [("list_files", {"path": "."})]
                assert app._agent_running is False

        asyncio.run(exercise())
    finally:
        agent.configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def test_native_tools_system_prompt_swaps_protocol(monkeypatch, tmp_path):
    import remie.agent as agent

    monkeypatch.setattr(tui.codex_auth, "auth_json_path", lambda: tmp_path / "a.json")

    def make_jwt(claims):
        segment = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
        return f"e.{segment.decode()}.s"

    import time as time_module

    token = make_jwt(
        {
            "exp": time_module.time() + 3600,
            tui.codex_auth.OPENAI_AUTH_CLAIM: {"chatgpt_plan_type": "plus"},
        }
    )

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            agent.configure_openai(
                "http://localhost:7070/v1", "k", "m", provider="local"
            )
            app._refresh_system_prompt()
            system_text = app.conversation[0]["content"]
            assert "tool: TOOL_NAME" in system_text
            assert "'thinking:'" in system_text

            agent.configure_openai(
                tui.CODEX_BACKEND_BASE, "", "gpt-5.5", provider="codex"
            )
            try:
                await app._refresh_system_prompt_async() if hasattr(
                    app, "_refresh_system_prompt_async"
                ) else None
                app._refresh_system_prompt()
                native_text = app.conversation[0]["content"]
                assert "tool: TOOL_NAME" not in native_text
                assert "function tools provided with each request" in native_text
            finally:
                pass

    asyncio.run(exercise())


def test_connection_screen_openrouter_live_model_list(monkeypatch):
    """Regression: agent.fetch_openrouter_models returns plain model id
    strings (context windows are cached inside the agent); the connection
    screen must populate the dropdown without unpacking them as tuples."""
    import remie.agent as _agent

    async def fake_fetch():
        return ["vendor/model-a", "vendor/model-b"]

    monkeypatch.setattr(connection_screen, "fetch_openrouter_models", fake_fetch)
    previous = get_config()
    _agent.configure_openai("http://localhost:7070/v1", "k", "m", provider="local")
    try:

        async def exercise():
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)

                provider = screen.query_one("#provider-select", Select)
                provider.value = "openrouter"
                await pilot.pause()

                # API key field is required/visible; base URL is fixed.
                assert screen.query_one("#api-key-input", Input).display
                assert not screen.query_one("#base-url-input", Input).display

                model_select = screen.query_one("#model-select", Select)
                for _ in range(100):
                    values = []
                    for option in model_select._options:
                        values.append(
                            option.value if hasattr(option, "value") else option[1]
                        )
                    if "vendor/model-a" in values:
                        break
                    await pilot.pause()
                assert "vendor/model-b" in values
                # The prior selection is intentionally kept as an option;
                # anything selected must still be a valid choice.
                assert model_select.value in values

        asyncio.run(exercise())
    finally:
        _agent.configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def test_connection_screen_model_search_filters_live_list(monkeypatch):
    import remie.agent as _agent
    from remie.model_names import ModelInfo

    async def fake_fetch():
        return [
            ModelInfo(id="google/gemini-x", display="Gemini X", vendor="Google"),
            ModelInfo(id="z-ai/glm-5.3", display="GLM 5.3", vendor="Z.AI"),
            ModelInfo(
                id="dots-studio/dots-3:free",
                display="Dots 3",
                vendor="Dots Studio",
                free=True,
            ),
        ]

    monkeypatch.setattr(connection_screen, "fetch_openrouter_models", fake_fetch)
    previous = get_config()
    _agent.configure_openai("http://localhost:7070/v1", "k", "m", provider="local")
    try:

        async def exercise():
            app = AgentApp()
            async with app.run_test() as pilot:
                await app.action_open_connection()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ConnectionScreen)
                screen.query_one("#provider-select", Select).value = "openrouter"
                await pilot.pause()

                model_select = screen.query_one("#model-select", Select)

                def current_values():
                    return [
                        option.value
                        if hasattr(option, "value") else option[1]
                        for option in model_select._options
                        if (option.value if hasattr(option, "value") else option[1])
                        is not Select.NULL
                    ]

                for _ in range(100):
                    if "google/gemini-x" in current_values():
                        break
                    await pilot.pause()
                assert "z-ai/glm-5.3" in current_values()

                search = screen.query_one("#model-search", Input)
                search.value = "gem"
                await pilot.pause()
                assert current_values() == ["google/gemini-x"]

                # Clearing restores the full (filtered-out) catalog.
                search.value = ""
                await pilot.pause()
                values = current_values()
                assert "z-ai/glm-5.3" in values and "dots-studio/dots-3:free" in values

        asyncio.run(exercise())
    finally:
        _agent.configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def test_chat_picker_search_filters(monkeypatch):
    from remie.tools import create_chat, rename_chat

    alpha = create_chat()
    rename_chat(alpha["id"], "alpha refactor")
    beta = create_chat()
    rename_chat(beta["id"], "beta migration")

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            app._load_chat_into_ui(alpha["id"])
            await app.action_open_chats()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            search = screen.query_one("#chat-search", Input)
            search.value = "migration"
            await pilot.pause()

            chat_list = screen.query_one("#chat-list", OptionList)
            visible_ids = [option.id for option in chat_list._options]
            assert visible_ids == [beta["id"]]

            search.value = ""
            await pilot.pause()
            visible_ids = [option.id for option in chat_list._options]
            assert alpha["id"] in visible_ids and beta["id"] in visible_ids

    asyncio.run(exercise())


def test_memory_picker_search_filters(monkeypatch):
    async def exercise():
        memory_tool("add", "base note")  # activates 'general'
        memory_tool("add", "design fact", name="design-notes")

        app = AgentApp()
        async with app.run_test() as pilot:
            await app.action_open_memory()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MemoryScreen)

            search = screen.query_one("#memory-search", Input)
            search.value = "design"
            await pilot.pause()

            memory_select = screen.query_one("#memory-select", Select)
            names = [name for name, _ in memory_select._options]
            assert names == ["design-notes"]
            from remie.tools import find_memory_by_name as _find_by_name

            assert memory_select.value == _find_by_name("design-notes")["id"]

            search.value = ""
            await pilot.pause()
            names = [name for name, _ in memory_select._options]
            assert "general" in names

    asyncio.run(exercise())


def test_adaptive_title_updates_after_completed_turn(monkeypatch):
    """A completed turn re-titles an auto-managed chat and records it as auto."""
    async def reply_stream(_c, usage_box=None, reasoning_box=None, finish_box=None, **_kw):
        yield "reply"

    monkeypatch.setattr(tui_app, "stream_llm_call", reply_stream)

    async def fake_title(messages):
        return "Fix Login Redirect Bug"

    monkeypatch.setattr(tui_app, "generate_chat_title", fake_title)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("please fix the login bug")
            await pilot.pause()

            chat = find_chat_by_id(app._chat_id)
            assert chat is not None
            assert chat["name"] == "Fix Login Redirect Bug"
            assert chat["title_source"] == "auto"
            assert app.sub_title == "Fix Login Redirect Bug"

    asyncio.run(exercise())


def test_adaptive_title_sees_transcript_and_current_name(monkeypatch):
    """The title prompt includes recent transcript plus the current title."""
    async def reply_stream(_c, usage_box=None, reasoning_box=None, finish_box=None, **_kw):
        yield "reply"

    monkeypatch.setattr(tui_app, "stream_llm_call", reply_stream)

    captured = {}

    async def fake_title(messages):
        captured["messages"] = messages
        return "whatever"

    monkeypatch.setattr(tui_app, "generate_chat_title", fake_title)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.run_agent_turn("discuss caching strategy")
            await pilot.pause()

        roles = [m["role"] for m in captured["messages"]]
        assert roles[0] == "system"
        assert "Current chat title:" in captured["messages"][0]["content"]
        contents = json.dumps(
            [
                m.get("content")
                for m in captured["messages"][1:]
                if m["role"] == "user"
            ],
            default=str,
        )
        assert "discuss caching strategy" in contents

    asyncio.run(exercise())


def test_manual_rename_blocks_adaptive_retitle(monkeypatch):
    """Once a person renames a chat, Remie never retitles it again."""
    from remie.tools import rename_chat

    async def reply_stream(_c, usage_box=None, reasoning_box=None, finish_box=None, **_kw):
        yield "reply"

    monkeypatch.setattr(tui_app, "stream_llm_call", reply_stream)

    async def unexpected_title(_messages):
        raise AssertionError("title generation should be skipped")

    monkeypatch.setattr(tui_app, "generate_chat_title", unexpected_title)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            renamed = rename_chat(app._chat_id, "My custom name")
            assert renamed is not None
            await app.run_agent_turn("fix the parser bug")
            await pilot.pause()

            chat = find_chat_by_id(app._chat_id)
            assert chat is not None
            assert chat["name"] == "My custom name"
            assert chat["title_source"] == "manual"

    asyncio.run(exercise())


def test_adaptive_title_keeps_unchanged_topic_stable(monkeypatch):
    """Returning the current title performs no rewrite at all."""
    async def reply_stream(_c, usage_box=None, reasoning_box=None, finish_box=None, **_kw):
        yield "reply"

    monkeypatch.setattr(tui_app, "stream_llm_call", reply_stream)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            from remie.tools import rename_chat

            rename_chat(app._chat_id, "Discuss Caching", title_source="auto")

            calls = {"n": 0}

            async def same_title(_messages):
                calls["n"] += 1
                return "Discuss Caching"

            monkeypatch.setattr(tui_app, "generate_chat_title", same_title)
            await app.run_agent_turn("more about caches")
            await pilot.pause()

            assert calls["n"] == 1
            chats = load_chat_index()
            entry = chats[app._chat_id]
            assert entry["updated_at"] == chats[app._chat_id]["updated_at"]
            assert entry["name"] == "Discuss Caching"

    asyncio.run(exercise())


def test_adaptive_title_falls_back_when_generation_fails(monkeypatch):
    """An empty generated title keeps the previous auto title instead of
    clobbering it with a fallback derived only from the last prompt."""
    async def reply_stream(_c, usage_box=None, reasoning_box=None, finish_box=None, **_kw):
        yield "reply"

    monkeypatch.setattr(tui_app, "stream_llm_call", reply_stream)

    async def empty_title(_messages):
        return ""

    monkeypatch.setattr(tui_app, "generate_chat_title", empty_title)

    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            from remie.tools import rename_chat

            rename_chat(app._chat_id, "Refactor Auth Flow", title_source="auto")
            await app.run_agent_turn("now what about retries?")
            await pilot.pause()

            chat = find_chat_by_id(app._chat_id)
            assert chat is not None
            assert chat["name"] == "Refactor Auth Flow"

    asyncio.run(exercise())


def test_ctrl_p_tabs_render_existing_modal_layouts():
    async def exercise():
        from remie.tui.screens.open import OpenScreen

        app = AgentApp()
        async with app.run_test(size=(110, 50)) as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, OpenScreen)
            # Inactive management layouts are mounted lazily so opening the
            # launcher does not pay the composition cost for all four tabs.
            assert len(screen.query("#open-chat-content")) == 1
            assert len(screen.query("#open-memory-content")) == 0
            assert len(screen.query("#open-provider-content")) == 0
            assert len(screen.query("#open-model-content")) == 0
            tabs = screen.query_one("#open-tabs")
            expected = {
                "open-chats": "chat-dialog",
                "open-memories": "memory-dialog",
                "open-providers": "connection-dialog",
                "open-models": "model-dialog",
            }
            for pane_id, dialog_id in expected.items():
                tabs.active = pane_id
                await pilot.pause()
                assert tabs.active == pane_id
                dialog = screen.query_one(f"#{dialog_id}")
                pane = screen.query_one(f"#{pane_id}")
                assert dialog.region == pane.region
                assert not dialog.styles.border

    asyncio.run(exercise())


def test_effort_slider_keyboard_and_pointer():
    async def exercise():
        app = AgentApp()
        async with app.run_test() as pilot:
            await app.push_screen(ConnectionScreen())
            slider = app.screen.query_one(EffortSlider)
            slider.disabled = False
            slider.focus()
            await pilot.press('home', 'right', 'right')
            assert slider.value == 'medium'
            await pilot.press('end', 'right')
            assert slider.value == 'max'
            slider.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click(slider, offset=(1, 0))
            assert slider.value == 'off'
            assert not app.screen.query('#reasoning-search')
            # Dragging across the slider must never begin a text selection.
            assert slider.ALLOW_SELECT is False
            await pilot.mouse_down(slider, offset=(1, 0))
            await pilot.mouse_up(slider, offset=(10, 0))
            await pilot.pause()
            assert slider.text_selection is None
            assert slider.value in REASONING_EFFORTS
    asyncio.run(exercise())


def test_long_question_keeps_footer_and_answer_accessible():
    async def exercise():
        app = AgentApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = AskUserScreen('Long question\n' * 60, ['Yes', 'No'])
            await app.push_screen(screen)
            await pilot.pause()
            dialog = screen.query_one('#ask-dialog')
            footer = screen.query_one('#ask-footer')
            assert footer.region.bottom <= dialog.content_region.bottom
            screen._show_custom_answer()
            await pilot.pause()
            submit = screen.query_one('#ask-submit')
            assert submit.region.bottom <= dialog.content_region.bottom
            assert screen.query_one('#ask-input').region.intersection(
                screen.query_one('#ask-body').content_region
            ).height > 0
    asyncio.run(exercise())
