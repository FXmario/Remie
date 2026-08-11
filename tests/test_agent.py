import pytest
import httpx
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from chaldea.agent import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_MODELS,
    configure_openai,
    edit_file_tool,
    extract_thinking,
    extract_tool_invocations,
    fetch_opencode_go_models,
    get_config,
    get_connection_error_message,
    get_tool_summary,
    list_files_tool,
    read_file_tool,
    render_assistant_message,
    render_assistant_panel,
    render_user_message,
    resolve_abs_path,
    run_tool,
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


class TestEditFileTool:
    def test_replaces_first_occurrence(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("aaa bbb aaa", encoding="utf-8")
        result = edit_file_tool(str(path), "aaa", "X")
        assert result == {"path": str(path.resolve()), "action": "edited"}
        assert path.read_text(encoding="utf-8") == "X bbb aaa"

    def test_old_str_not_found(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("hello", encoding="utf-8")
        result = edit_file_tool(str(path), "zzz", "X")
        assert result == {"path": str(path.resolve()), "action": "old_str not found"}
        assert path.read_text(encoding="utf-8") == "hello"

    def test_empty_old_str_creates_file(self, tmp_path):
        path = tmp_path / "new.txt"
        result = edit_file_tool(str(path), "", "created content")
        assert result == {"path": str(path.resolve()), "action": "created_file"}
        assert path.read_text(encoding="utf-8") == "created content"

    def test_empty_old_str_overwrites_existing(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("old content", encoding="utf-8")
        result = edit_file_tool(str(path), "", "new content")
        assert result["action"] == "created_file"
        assert path.read_text(encoding="utf-8") == "new content"


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


class TestGetToolSummary:
    def test_known_tools_return_summary(self):
        assert get_tool_summary("read_file") == "read a file"
        assert get_tool_summary("list_files") == "list the files in a directory"
        assert get_tool_summary("edit_file") == "edit a file"

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
