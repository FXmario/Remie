import pytest

from chaldea.agent import (
    edit_file_tool,
    extract_thinking,
    extract_tool_invocations,
    list_files_tool,
    read_file_tool,
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
