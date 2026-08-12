import pytest
import httpx
import subprocess
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from fuica.agent import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
    RUN_COMMAND_MAX_OUTPUT,
    RUN_COMMAND_TIMEOUT,
    configure_openai,
    edit_file_tool,
    estimate_conversation_tokens,
    estimate_tokens,
    extract_thinking,
    extract_tool_invocations,
    fetch_opencode_go_models,
    get_config,
    get_connection_error_message,
    get_tool_summary,
    glob_files_tool,
    load_config,
    list_files_tool,
    read_file_tool,
    render_assistant_message,
    render_assistant_panel,
    render_user_message,
    resolve_abs_path,
    run_command_tool,
    run_tool,
    save_config,
    stream_llm_call,
    strip_protocol_lines,
    tree_files_tool,
)


class TestResolveAbsPath:
    def test_relative_path_resolves_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = resolve_abs_path("file.py")
        assert result == (tmp_path / "file.py").resolve()

    def test_absolute_path_unchanged(self, tmp_path):
        path = tmp_path / "file.py"
        assert resolve_abs_path(str(path)) == path.resolve()

    def test_expanduser(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_abs_path("~/file.py") == tmp_path / "file.py"


class TestReadFileTool:
    def test_reads_file_content(self, tmp_path):
        path = tmp_path / "hello.txt"
        path.write_text("hello world", encoding="utf-8")
        result = read_file_tool(str(path))
        assert result == {
            "file_path": str(path.resolve()),
            "content": "hello world",
        }

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_file_tool(str(tmp_path / "missing.txt"))

    def test_run_tool_accepts_path_alias(self, tmp_path):
        path = tmp_path / "hello.txt"
        path.write_text("hello", encoding="utf-8")
        result = run_tool("read_file", {"path": str(path)})
        assert result["content"] == "hello"

    def test_run_tool_returns_directory_error(self, tmp_path):
        result = run_tool("read_file", {"path": str(tmp_path)})
        assert result["error"].startswith("IsADirectoryError:")


class TestListFilesTool:
    def test_lists_files_and_dirs(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        result = list_files_tool(str(tmp_path))
        assert result["path"] == str(tmp_path.resolve())
        files = {(entry["filename"], entry["type"]) for entry in result["files"]}
        assert files == {("a.txt", "file"), ("sub", "dir")}

    def test_empty_directory(self, tmp_path):
        result = list_files_tool(str(tmp_path))
        assert result["files"] == []

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_files_tool(str(tmp_path / "missing"))


class TestGlobFilesTool:
    def test_matches_recursively(self, tmp_path):
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("", encoding="utf-8")
        (tmp_path / "sub" / "c.txt").write_text("", encoding="utf-8")
        result = glob_files_tool("*.py", str(tmp_path))
        assert result["count"] == 2
        assert result["matches"] == ["a.py", "sub/b.py"]

    def test_matches_full_path_pattern(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
        result = glob_files_tool("src/*.py", str(tmp_path))
        assert result["matches"] == ["src/main.py"]

    def test_skips_ignored_dirs(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / "main.py").write_text("", encoding="utf-8")
        (tmp_path / ".venv" / "hidden.py").write_text("", encoding="utf-8")
        result = glob_files_tool("*.py", str(tmp_path))
        assert result["matches"] == ["main.py"]

    def test_no_matches_returns_empty(self, tmp_path):
        result = glob_files_tool("*.rs", str(tmp_path))
        assert result["matches"] == []
        assert result["count"] == 0
        assert result["truncated"] is False

    def test_truncates_at_limit(self, tmp_path):
        for i in range(300):
            (tmp_path / f"f{i}.txt").write_text("", encoding="utf-8")
        result = glob_files_tool("*.txt", str(tmp_path))
        assert result["truncated"] is True
        assert len(result["matches"]) == 200

    def test_dispatch_via_run_tool(self, tmp_path):
        (tmp_path / "x.py").write_text("", encoding="utf-8")
        result = run_tool(
            "glob_files", {"pattern": "*.py", "path": str(tmp_path)}
        )
        assert result["matches"] == ["x.py"]


class TestTreeFilesTool:
    def test_renders_nested_tree(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "root.txt").write_text("", encoding="utf-8")
        result = tree_files_tool(str(tmp_path))
        tree = result["tree"]
        assert "sub/" in tree
        assert "a.py" in tree
        assert "root.txt" in tree

    def test_skips_ignored_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "real.py").write_text("", encoding="utf-8")
        tree = tree_files_tool(str(tmp_path))["tree"]
        assert ".git" not in tree
        assert "real.py" in tree

    def test_depth_is_clamped(self, tmp_path):
        (tmp_path / "l1").mkdir()
        (tmp_path / "l1" / "l2").mkdir()
        (tmp_path / "l1" / "l2" / "l3").mkdir()
        (tmp_path / "l1" / "l2" / "l3" / "l4").mkdir()
        (tmp_path / "l1" / "l2" / "l3" / "l4" / "deep.py").write_text(
            "", encoding="utf-8"
        )
        result = tree_files_tool(str(tmp_path), max_depth=2)
        assert "deep.py" not in result["tree"]
        assert "l2/" in result["tree"]

    def test_empty_directory(self, tmp_path):
        result = tree_files_tool(str(tmp_path))
        assert result["truncated"] is False

    def test_dispatch_via_run_tool(self, tmp_path):
        result = run_tool("tree_files", {"path": str(tmp_path)})
        assert tmp_path.name in result["tree"]


class TestEditFileTool:
    def test_replaces_first_occurrence(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("aaa bbb aaa", encoding="utf-8")
        result = edit_file_tool(str(path), "aaa", "X")
        assert result["path"] == str(path.resolve())
        assert result["action"] == "edited"
        assert path.read_text(encoding="utf-8") == "X bbb aaa"

    def test_replaces_first_occurrence_returns_diff(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        result = edit_file_tool(str(path), "bbb", "CCC")
        diff = result["diff"]
        assert "-bbb" in diff
        assert "+CCC" in diff
        assert "a/" + str(path.resolve()) in diff
        assert "b/" + str(path.resolve()) in diff

    def test_old_str_not_found(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("hello", encoding="utf-8")
        result = edit_file_tool(str(path), "zzz", "X")
        assert result == {"path": str(path.resolve()), "action": "old_str not found"}
        assert path.read_text(encoding="utf-8") == "hello"

    def test_empty_old_str_creates_file(self, tmp_path):
        path = tmp_path / "new.txt"
        result = edit_file_tool(str(path), "", "created content")
        assert result["path"] == str(path.resolve())
        assert result["action"] == "created_file"
        assert "+created content" in result["diff"]
        assert path.read_text(encoding="utf-8") == "created content"

    def test_empty_old_str_overwrites_existing(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("old content", encoding="utf-8")
        result = edit_file_tool(str(path), "", "new content")
        assert result["action"] == "created_file"
        assert "-old content" in result["diff"]
        assert "+new content" in result["diff"]
        assert path.read_text(encoding="utf-8") == "new content"

    def test_large_diff_is_truncated(self, tmp_path):
        path = tmp_path / "big.txt"
        path.write_text("x" * 10_000, encoding="utf-8")
        result = edit_file_tool(str(path), "x" * 100, "y" * 100)
        assert "diff truncated" in result["diff"]
        assert len(result["diff"]) <= 4040


class TestRunCommandTool:
    def test_runs_command_and_captures_stdout(self):
        result = run_command_tool("echo hello")
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "hello"
        assert result["timed_out"] is False
        assert result["truncated"] is False

    def test_captures_non_zero_exit_and_stderr(self):
        result = run_command_tool("echo oops >&2; exit 3")
        assert result["exit_code"] == 3
        assert result["stderr"].strip() == "oops"
        assert result["stdout"] == ""

    def test_runs_in_given_cwd(self, tmp_path):
        (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
        result = run_command_tool("ls marker.txt", cwd=str(tmp_path))
        assert result["exit_code"] == 0
        assert "marker.txt" in result["stdout"]

    def test_uses_shell_pipes_and_env(self):
        result = run_command_tool("echo one | tr 'o' '0'")
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "0ne"

    def test_timeout_kills_long_command(self, monkeypatch):
        import time

        def fake_run(*args, **kwargs):
            time.sleep(1)
            cmd = kwargs.get("args", args[0] if args else "")
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        monkeypatch.setattr(
            "fuica.agent.subprocess.run", fake_run, raising=False
        )
        result = run_command_tool("sleep 60")
        assert result["timed_out"] is True
        assert result["exit_code"] == 124

    def test_output_is_truncated(self, monkeypatch):
        class FakeResult:
            returncode = 0
            stdout = "a" * (RUN_COMMAND_MAX_OUTPUT + 10_000)
            stderr = ""

        monkeypatch.setattr(
            "fuica.agent.subprocess.run",
            lambda *a, **k: FakeResult(),
            raising=False,
        )
        result = run_command_tool("cat bigfile")
        assert result["truncated"] is True
        assert "[output truncated]" in result["stdout"]
        assert len(result["stdout"]) <= RUN_COMMAND_MAX_OUTPUT

    def test_dispatch_via_run_tool(self):
        result = run_tool("run_command", {"command": "echo dispatched"})
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "dispatched"


class TestRunTool:
    def test_dispatches_read_file(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_text("print(1)", encoding="utf-8")
        result = run_tool("read_file", {"filename": str(path)})
        assert result["content"] == "print(1)"

    def test_dispatches_list_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        result = run_tool("list_files", {"path": str(tmp_path)})
        assert result["path"] == str(tmp_path.resolve())
        assert result["files"] == [{"filename": "a.txt", "type": "file"}]

    def test_dispatches_edit_file(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("foo", encoding="utf-8")
        result = run_tool(
            "edit_file", {"path": str(path), "old_str": "foo", "new_str": "bar"}
        )
        assert result["action"] == "edited"
        assert path.read_text(encoding="utf-8") == "bar"

    def test_unknown_tool(self):
        args = {"x": 1}
        assert run_tool("nonexistent", args) == {
            "action": "unknown_tool_nonexistent",
            "args": args,
        }


class TestExtractThinking:
    def test_joins_thinking_lines(self):
        text = "thinking: first\nthinking: second"
        assert extract_thinking(text) == "first\nsecond"

    def test_no_thinking_returns_empty(self):
        assert extract_thinking("just a reply") == ""
        assert extract_thinking("") == ""


class TestExtractToolInvocations:
    def test_parses_single_invocation(self):
        text = 'tool: read_file({"filename": "a.py"})'
        assert extract_tool_invocations(text) == [("read_file", {"filename": "a.py"})]

    def test_parses_multiple_invocations(self):
        text = 'tool: read_file({"filename": "a.py"})\ntool: list_files({"path": "."})'
        assert extract_tool_invocations(text) == [
            ("read_file", {"filename": "a.py"}),
            ("list_files", {"path": "."}),
        ]

    def test_parses_python_style_keyword_arguments(self):
        text = 'tool: list_files(path=".")'
        assert extract_tool_invocations(text) == [("list_files", {"path": "."})]

    def test_ignores_invalid_json(self):
        assert extract_tool_invocations("tool: read_file({bad})") == []

    def test_ignores_non_tool_lines(self):
        assert extract_tool_invocations("hello world") == []

    def test_ignores_malformed_line_without_closing_paren(self):
        assert extract_tool_invocations('tool: read_file({"filename": "a.py"') == []


class TestEstimateTokens:
    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0

    def test_plain_text(self):
        assert estimate_tokens("hello world") == 2

    def test_newlines_add_bonus(self):
        assert estimate_tokens("a" * 40 + "\n" * 30) == 27

    def test_minimum_one_for_non_empty(self):
        assert estimate_tokens("x") == 1


class TestEstimateConversationTokens:
    def test_sums_string_content(self):
        conversation = [
            {"role": "system", "content": "abcd"},
            {"role": "user", "content": "efghijkl"},
        ]
        assert estimate_conversation_tokens(conversation) == estimate_tokens(
            "abcd"
        ) + estimate_tokens("efghijkl")

    def test_sums_tool_result_messages(self):
        conversation = [
            {"role": "user", "content": "tool_result({\"x\": 1})"},
            {"role": "user", "content": "tool_result({\"y\": 2})"},
        ]
        assert estimate_conversation_tokens(conversation) == 2 * estimate_tokens(
            'tool_result({"x": 1})'
        )

    def test_ignores_non_string_content(self):
        conversation = [{"role": "user", "content": [{"image": "base64"}]}]
        assert estimate_conversation_tokens(conversation) == 0

    def test_empty_conversation(self):
        assert estimate_conversation_tokens([]) == 0


class TestStripProtocolLines:
    def test_removes_thinking_and_tool_lines(self):
        text = "thinking: why\nhello\ntool: read_file({\"filename\": \"a.py\"})\ndone"
        assert strip_protocol_lines(text) == "hello\ndone"

    def test_keeps_normal_text(self):
        assert strip_protocol_lines("just a reply") == "just a reply"

    def test_empty_input(self):
        assert strip_protocol_lines("") == ""

    def test_only_protocol_lines(self):
        assert strip_protocol_lines("thinking: x\ntool: list_files({\"path\": \".\"})") == ""


class TestStreamLlmUsageAndReasoning:
    def test_captures_usage_and_reasoning(self, monkeypatch):
        import asyncio

        class FakeUsage:
            prompt_tokens = 100
            completion_tokens = 50

        class FakeDelta:
            def __init__(self, content=None, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content

        class FakeChoice:
            def __init__(self, delta):
                self.delta = delta

        class FakeChunk:
            def __init__(self, choices, usage=None):
                self.choices = choices
                self.usage = usage

        class FakeCompletions:
            async def create(self, **kwargs):
                async def gen():
                    yield FakeChunk(
                        [FakeChoice(FakeDelta(reasoning_content="think..."))]
                    )
                    yield FakeChunk([FakeChoice(FakeDelta(content="hi"))])
                    yield FakeChunk([], usage=FakeUsage())

                return gen()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        monkeypatch.setattr("fuica.agent.openai_client", FakeClient())

        conversation = []
        usage_box = {}
        reasoning_box = []

        async def collect():
            return [d async for d in stream_llm_call(
                conversation, usage_box, reasoning_box
            )]

        chunks = asyncio.run(collect())
        assert chunks == ["hi"]
        assert usage_box == {"prompt_tokens": 100, "completion_tokens": 50}
        assert reasoning_box == ["think..."]

    def test_no_boxes_keeps_plain_stream(self, monkeypatch):
        import asyncio

        class FakeDelta:
            content = "hi"

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]

        class FakeCompletions:
            async def create(self, **kwargs):
                async def gen():
                    yield FakeChunk()

                return gen()

        class FakeClient:
            chat = type("FakeChat", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr("fuica.agent.openai_client", FakeClient())

        async def collect():
            return [d async for d in stream_llm_call([])]

        assert asyncio.run(collect()) == ["hi"]

    def test_sends_reasoning_effort_and_omits_off(self, monkeypatch):
        import asyncio
        import fuica.agent as agent

        captured: list[dict] = []

        class FakeDelta:
            content = "hi"

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]

        class FakeCompletions:
            async def create(self, **kwargs):
                captured.append(kwargs)

                async def gen():
                    yield FakeChunk()

                return gen()

        class FakeClient:
            chat = type("FakeChat", (), {"completions": FakeCompletions()})()

        previous = agent.get_config()
        try:
            agent.configure_openai(
                "http://localhost:1234/v1",
                "key",
                "model",
                reasoning_effort="high",
            )
            monkeypatch.setattr(agent, "openai_client", FakeClient())

            async def collect():
                return [d async for d in stream_llm_call([])]

            assert asyncio.run(collect()) == ["hi"]
            assert captured[-1]["reasoning_effort"] == "high"

            agent.configure_openai(
                "http://localhost:1234/v1",
                "key",
                "model",
                reasoning_effort="off",
            )
            monkeypatch.setattr(agent, "openai_client", FakeClient())
            assert asyncio.run(collect()) == ["hi"]
            assert "reasoning_effort" not in captured[-1]
        finally:
            agent.configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )


class TestGetToolSummary:
    def test_known_tools_return_summary(self):
        assert get_tool_summary("read_file") == "read a file"
        assert get_tool_summary("list_files") == "list the files in a directory"
        assert get_tool_summary("edit_file") == "edit a file"
        assert get_tool_summary("run_command") == "run a shell command"
        assert get_tool_summary("glob_files") == "find files matching a glob pattern"
        assert get_tool_summary("tree_files") == "show the directory tree"

    def test_unknown_tool_falls_back_to_name(self):
        assert get_tool_summary("nonexistent") == "nonexistent"


class TestConnectionErrorMessage:
    def test_timeout_error(self):
        error = httpx.ReadTimeout(
            "request timed out",
            request=httpx.Request("GET", "http://localhost:1234/v1"),
        )
        assert get_connection_error_message(error) == (
            "The LLM request to http://localhost:1234/v1 timed out. "
            "Check that the model server is responding."
        )

    def test_connect_error(self):
        error = httpx.ConnectError(
            "connection refused",
            request=httpx.Request("GET", "http://localhost:1234/v1"),
        )
        assert get_connection_error_message(error) == (
            "Could not connect to the LLM server at http://localhost:1234/v1. "
            "Check that it is running."
        )

    def test_non_connection_error_returns_none(self):
        assert get_connection_error_message(ValueError("bad response")) is None


class TestRenderMessages:
    def test_user_message_is_panel(self):
        panel = render_user_message("hello")
        assert isinstance(panel, Panel)
        assert panel.title == "You"

    def test_assistant_message_plain_text(self):
        renderable = render_assistant_message("just a reply")
        assert isinstance(renderable, Markdown)

    def test_assistant_message_renders_markdown(self):
        renderable = render_assistant_message(
            "# Heading\n\n**bold** and `inline code`\n\n- item"
        )
        assert isinstance(renderable, Markdown)
        assert renderable.markup == (
            "# Heading\n\n**bold** and `inline code`\n\n- item"
        )

    def test_assistant_message_supports_code_blocks(self):
        renderable = render_assistant_message("```python\na = 1\n```")
        assert isinstance(renderable, Markdown)
        assert renderable.code_theme == "ansi_dark"

    def test_assistant_message_supports_light_code_theme(self):
        renderable = render_assistant_message("```python\na = 1\n```", "ansi_light")
        assert renderable.code_theme == "ansi_light"

    def test_assistant_panel_is_panel(self):
        panel = render_assistant_panel("final answer")
        assert isinstance(panel, Panel)
        assert panel.title == "Assistant"


class TestConnectionConfig:
    def test_configure_openai_updates_config_and_client(self):
        previous = get_config()
        try:
            config = configure_openai("http://test:1234/v1", "secret", "test-model")
            assert get_config().base_url == "http://test:1234/v1"
            assert get_config().api_key == "secret"
            assert get_config().model == "test-model"
            assert config.model == "test-model"
        finally:
            configure_openai(previous.base_url, previous.api_key, previous.model)

    def test_configure_openai_updates_provider_and_effort(self):
        previous = get_config()
        try:
            config = configure_openai(
                "http://test:1234/v1",
                "secret",
                "test-model",
                provider="local",
                reasoning_effort="high",
            )
            assert config.provider == "local"
            assert config.reasoning_effort == "high"
            assert get_config().reasoning_effort == "high"
        finally:
            configure_openai(
                previous.base_url,
                previous.api_key,
                previous.model,
                previous.provider,
                previous.reasoning_effort,
            )

    def test_saved_config_round_trips_provider_and_effort(self, tmp_path, monkeypatch):
        import fuica.agent as agent

        config_file = tmp_path / "config.json"
        monkeypatch.setattr(agent, "CONFIG_FILE", config_file)
        monkeypatch.setattr(agent, "CONFIG_DIR", tmp_path)
        original = agent.ConnectionConfig(
            "https://example.test/v1",
            "secret",
            "model",
            "opencode-go",
            "max",
        )
        save_config(original)
        assert load_config() == original

    def test_load_config_derives_provider_for_legacy_file(self, tmp_path, monkeypatch):
        import fuica.agent as agent

        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"base_url": "https://opencode.ai/zen/go/v1", '
            '"api_key": "secret", "model": "kimi-k3"}',
            encoding="utf-8",
        )
        monkeypatch.setattr(agent, "CONFIG_FILE", config_file)
        config = load_config()
        assert config.provider == "opencode-go"
        assert config.reasoning_effort == "medium"

    def test_fetch_opencode_go_models_falls_back_on_error(self, monkeypatch):
        import asyncio

        async def fake_get(self, url, headers=None):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        models = asyncio.run(fetch_opencode_go_models("invalid"))
        assert models == OPENCODE_GO_MODELS

    def test_fetch_opencode_go_models_parses_payload(self, monkeypatch):
        import asyncio

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"id": "kimi-k3"}, {"id": "grok-4.5"}]}

        async def fake_get(self, url, headers=None):
            return FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        models = asyncio.run(fetch_opencode_go_models("valid"))
        assert models == ["kimi-k3", "grok-4.5"]

    def test_opencode_go_constants(self):
        assert OPENCODE_GO_BASE_URL == "https://opencode.ai/zen/go/v1"
        assert "deepseek-v4-flash" in OPENCODE_GO_MODELS

    def test_opencode_models_filter_unsupported_ids(self):
        import asyncio

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"id": "gpt-5.6-luna"}, {"id": "kimi-k3"}]}

        async def fake_get(self, url, headers=None):
            return FakeResponse()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        try:
            models = asyncio.run(fetch_opencode_go_models("valid"))
        finally:
            monkeypatch.undo()
        assert models == ["kimi-k3"]
