import pytest
import httpx
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from chaldea.agent import (
    edit_file_tool,
    extract_thinking,
    extract_tool_invocations,
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
        error = httpx.ReadTimeout("request timed out")
        assert get_connection_error_message(error) == (
            "The LLM request to http://localhost:1234/v1 timed out. "
            "Check that the model server is responding."
        )

    def test_connect_error(self):
        error = httpx.ConnectError("connection refused")
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
        assert isinstance(renderable, Text)
        assert "just a reply" in renderable.plain

    def test_assistant_message_highlights_code(self):
        renderable = render_assistant_message("Here:\n```python\nx = 1\n```\nDone.")
        assert isinstance(renderable, Group)
        syntax_items = [
            item for item in renderable.renderables if isinstance(item, Syntax)
        ]
        assert len(syntax_items) == 1
        assert syntax_items[0].code == "x = 1"

    def test_assistant_message_multiple_code_blocks(self):
        renderable = render_assistant_message(
            "```python\na = 1\n```\nand\n```python\nb = 2\n```"
        )
        assert isinstance(renderable, Group)
        syntax_items = [
            item for item in renderable.renderables if isinstance(item, Syntax)
        ]
        assert [item.code for item in syntax_items] == ["a = 1", "b = 2"]

    def test_assistant_panel_is_panel(self):
        panel = render_assistant_panel("final answer")
        assert isinstance(panel, Panel)
        assert panel.title == "Assistant"
