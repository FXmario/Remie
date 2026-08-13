import asyncio

import httpx
from openai import BadRequestError
from PIL import Image
from rich.panel import Panel

import fuica.tui as tui
from fuica.agent import (
    ConnectionConfig,
    OPENCODE_GO_BASE_URL,
    configure_openai,
    get_config,
)
from fuica.tui import (
    MAX_AUTO_CONTINUATIONS,
    AgentApp,
    AgentScreen,
    AskUserScreen,
    ConnectionScreen,
    ModelBadge,
    PromptSubmitted,
    PromptTextArea,
    StatusIndicator,
    ThinkingIndicator,
    _format_tokens,
    _has_tool_call,
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
    local_badge.update_config(
        ConnectionConfig("http://localhost/v1", "key", "local", reasoning_effort="off")
    )
    assert local_badge.render().plain == "local  Local"

    remote_badge = ModelBadge()
    remote_badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL, "key", "kimi-k3", reasoning_effort="off"
        )
    )
    assert remote_badge.render().plain == "kimi-k3  OpenCode Go"


def test_connection_error_shows_toast_and_keeps_app_running(monkeypatch):
    async def exercise():
        async def failed_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
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
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
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
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
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
        async def endless_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
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

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
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


def test_truncated_response_continues_automatically(monkeypatch):
    async def exercise():
        calls = 0

        async def truncating_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                finish_box["finish_reason"] = "length"
                finish_box["truncated"] = True
                yield "The answer is cut "
            else:
                yield "off here, now complete."

        monkeypatch.setattr(tui, "stream_llm_call", truncating_stream)

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
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
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

        monkeypatch.setattr(tui, "stream_llm_call", truncating_stream)

        app = AgentApp()
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
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
        ):
            nonlocal calls
            calls += 1
            finish_box["finish_reason"] = "length"
            finish_box["truncated"] = True
            yield "x"

        monkeypatch.setattr(tui, "stream_llm_call", always_truncating)

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
            assert screen.query_one("#ask-option-0")
            assert screen.query_one("#ask-option-2")
            assert screen.query_one("#ask-input")

    asyncio.run(exercise())


def test_ask_user_tool_feeds_answer_back(monkeypatch):
    async def exercise():
        calls = 0

        async def tool_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
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

        monkeypatch.setattr(tui, "stream_llm_call", tool_stream)

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
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield 'tool: ask_user({"question": "q"})'
            else:
                yield "done"

        monkeypatch.setattr(tui, "stream_llm_call", tool_stream)

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


async def _empty_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
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
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

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
            app._compact_conversation()
            assert app.conversation[0]["role"] == "system"
            assert "omitted" in app.conversation[1]["content"]
            assert app.conversation[-1]["content"] == "recent"
            assert len(app.conversation) == 6

    asyncio.run(exercise())


def test_context_full_error_notifies_clearly(monkeypatch):
    async def exercise():
        class FakeRequest:
            method = "POST"
            url = "http://test/v1"

        class FakeResponse:
            request = FakeRequest()
            status_code = 400
            headers = {"x-request-id": "abc"}

        async def failing_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            raise BadRequestError(
                "maximum context length is 131072 tokens",
                response=FakeResponse(),
                body=None,
            )
            yield ""

        monkeypatch.setattr(tui, "stream_llm_call", failing_stream)
        notifications = []

        app = AgentApp()
        monkeypatch.setattr(app, "notify", lambda *a, **k: notifications.append(k))
        async with app.run_test() as pilot:
            await app.run_agent_turn("hello")
            await pilot.pause()

            assert notifications
            assert notifications[0].get("title") == "Context window full"

    asyncio.run(exercise())


def test_prompt_enter_submits_and_ctrl_j_inserts_newline(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

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
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

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
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

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
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

        app = AgentApp()
        app._prompt_history = ["hello", "world"]
        async with app.run_test() as pilot:
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
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

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
        app._prompt_history = ["saved"]
        async with app.run_test() as pilot:
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
            assert app._pending_image is not None

    asyncio.run(exercise())


def test_image_attachment_builds_multimodal_message(monkeypatch):
    async def exercise():
        async def fake_stream(_conversation, usage_box=None, reasoning_box=None, finish_box=None):
            if False:
                yield ""

        monkeypatch.setattr(tui, "stream_llm_call", fake_stream)

        app = AgentApp()
        image = Image.new("RGB", (8, 8), "red")
        app.set_pending_image(image)
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
            assert [part["type"] for part in user_content] == ["text", "image_url"]
            assert user_content[0]["text"] == "what is this"
            assert user_content[1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
            assert app._pending_image is None

    asyncio.run(exercise())


def test_model_badge_shows_token_usage():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig(
            OPENCODE_GO_BASE_URL, "key", "kimi-k3", reasoning_effort="off"
        )
    )
    badge.set_tokens(1234, 5678)
    assert badge.render().plain == "kimi-k3  OpenCode Go · 6.9k tok"


def test_model_badge_hides_usage_when_zero():
    badge = ModelBadge()
    badge.update_config(
        ConnectionConfig("http://localhost/v1", "key", "local", reasoning_effort="off")
    )
    assert badge.render().plain == "local  Local"


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
    assert badge.render().plain == "kimi-k3  OpenCode Go · effort max"


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
            await pilot.press("ctrl+p")
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#submit-button")
            assert screen.query_one("#cancel-button")
            assert not screen.query("#refresh-button")

    asyncio.run(exercise())


def test_connection_screen_restores_provider_and_effort():
    async def exercise():
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
                await pilot.press("ctrl+p")
                await pilot.pause()
                screen = app.screen
                assert screen.query_one("#provider-select").value == "opencode-go"
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
                await pilot.press("ctrl+p")
                await pilot.pause()
                screen = app.screen
                assert screen.query_one("#base-url-input").display is True
                assert screen.query_one("#base-url-label").display is True
                assert screen.query_one("#base-url-input").value == (
                    "http://localhost:7070/v1"
                )
                assert screen.query_one("#model-select").disabled is True
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    asyncio.run(exercise())


def test_turn_updates_badge_tokens(monkeypatch):
    async def exercise():
        async def fake_stream(
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
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
            _conversation, usage_box=None, reasoning_box=None, finish_box=None
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
