"""Rendering of tool results, diffs, and syntax-highlighted output."""

import json
from typing import Any

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


class _PlainWrite:
    """Wrap a rich renderable so RichLog can still extract plain text for
    selection and tests, while the actual rendering stays highlighted."""

    def __init__(self, plain: str, renderable: RenderableType) -> None:
        self.plain = plain
        self._renderable = renderable

    def __rich_console__(self, console, options):
        yield from console.render(self._renderable, options)


def _render_diff(diff: str, code_theme: str = "ansi_dark") -> Panel:
    """Render a unified diff as a highlightable panel."""
    return Panel(
        _make_syntax(diff, "diff", code_theme),
        title="Diff",
        border_style="cyan",
        padding=(0, 1),
    )


def _make_syntax(code: str, language: str, code_theme: str) -> RenderableType:
    """Highlight code with Pygments; fall back to escaped text on any error."""
    try:
        syntax = Syntax(code, language, theme=code_theme, word_wrap=False)
        # Rich does not raise for unknown lexer names; it leaves lexer None.
        if syntax.lexer is None:
            raise ValueError(f"Unknown lexer: {language}")
        return syntax
    except Exception:
        return Text.from_markup(escape(code))


def _guess_lexer_name(filename: str) -> str | None:
    """Guess a Pygments lexer name from a filename, or None when unknown."""
    try:
        from pygments.lexers import get_lexer_for_filename
        from pygments.util import ClassNotFound

        return get_lexer_for_filename(filename, "").name
    except ClassNotFound, TypeError, ValueError, OSError:
        return None


def _command_body(result: dict[str, Any]) -> str:
    """Raw stdout/stderr joined the same way the summary shows them."""
    output = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    return "\n".join(part for part in (output, stderr) if part)


def _pretty_json(text: str) -> str | None:
    """Return consistently indented JSON when *text* is a complete JSON value."""
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _command_output_lexer(output: str) -> str | None:
    """Detect a lexer for shell command output: json, unified diff, or a
    Python traceback; None (plain text) otherwise."""
    full_output = output.strip()
    if not full_output:
        return None
    if _pretty_json(full_output) is not None:
        return "json"
    sample = full_output[:500]
    if sample.startswith(("--- ", "+++ ", "@@ ")) or "\n--- " in sample:
        return "diff"
    lines = sample.splitlines()[:3]
    if len(lines) >= 2 and lines[0].startswith("Traceback") and 'File "' in sample:
        return "pytb"
    return None


TOOL_RESULT_MAX_CHARS = 2000


def _format_tool_result(name: str, result: dict[str, Any]) -> str:
    """Return a readable text summary of a tool result."""
    if result.get("blocked"):
        return ""
    if "error" in result:
        return f"Error: {result['error']}"
    if name == "read_file":
        content = result.get("content", "")
        return f"Read {result.get('file_path')} ({len(content)} chars)"
    if name == "edit_file":
        return f"{result.get('action', 'edited')}: {result.get('path')}"
    if name == "run_command":
        parts = [f"exit {result.get('exit_code')}"]
        if result.get("timed_out"):
            parts.append("timed out")
        output = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        body = "\n".join(part for part in (output, stderr) if part)
        summary = " · ".join(parts)
        return f"{summary}\n{body}" if body else summary
    if name == "list_files":
        entries = result.get("files", [])
        names = ", ".join(e.get("filename", "") for e in entries[:50])
        suffix = f" (+{len(entries) - 50} more)" if len(entries) > 50 else ""
        return f"{len(entries)} entries: {names}{suffix}"
    if name == "glob_files":
        matches = result.get("matches", [])
        names = ", ".join(matches[:50])
        suffix = f" (+{len(matches) - 50} more)" if len(matches) > 50 else ""
        return f"{result.get('count', len(matches))} matches: {names}{suffix}"
    if name == "tree_files":
        return result.get("tree", "")
    if name == "ask_user":
        answer = result.get("answer")
        return f"Answer: {answer}" if answer is not None else "Cancelled"
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def _truncate_body(text: str, limit: int = TOOL_RESULT_MAX_CHARS) -> tuple[str, bool]:
    """Truncate a body to `limit` chars; return (text, was_truncated)."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n… (result truncated)", True


def _plain_tool_panel(name: str, text: str) -> Panel:
    """Render a plain (unhighlighted) tool result panel."""
    return Panel(
        Text.from_markup(escape(text)),
        title=f"Tool result · {name}",
        border_style="blue",
        padding=(0, 1),
    )


def _render_read_file_result(result: dict[str, Any], code_theme: str) -> Panel:
    """Render a read_file result: a summary line plus the file content,
    syntax-highlighted by extension when a lexer can be guessed."""
    content = result.get("content", "")
    path = str(result.get("file_path", ""))
    body_text, truncated = _truncate_body(content)
    summary = f"Read {path} ({len(content)} chars)"
    if truncated:
        summary += " \u00b7 (result truncated)"
    lexer = _guess_lexer_name(path)
    if lexer is not None and body_text.strip():
        body: RenderableType = _PlainWrite(
            body_text, _make_syntax(body_text, lexer, code_theme)
        )
    else:
        body = Text.from_markup(escape(body_text))
    return Panel(
        Group(Text(summary, style="bold"), Text(), body),
        title="Tool result · read_file",
        border_style="blue",
        padding=(0, 1),
    )


def _render_run_command_result(
    result: dict[str, Any], code_theme: str
) -> RenderableType:
    """Render a run_command result: summary plus output, highlighted when the
    output looks like JSON, a unified diff, or a Python traceback."""
    output = _command_body(result)
    lexer = _command_output_lexer(output) if output else None
    parts = [f"exit {result.get('exit_code')}"]
    if result.get("timed_out"):
        parts.append("timed out")
    summary = " · ".join(parts)
    if lexer is not None:
        if lexer == "json":
            output = _pretty_json(output) or output
        body_text, truncated = _truncate_body(output)
        if truncated:
            summary += " \u00b7 (result truncated)"
        body = _PlainWrite(body_text, _make_syntax(body_text, lexer, code_theme))
        return Panel(
            Group(Text(summary, style="bold"), Text(), body),
            title="Tool result · run_command",
            border_style="blue",
            padding=(0, 1),
        )
    text = f"{summary}\n{output}" if output else summary
    body_text, truncated = _truncate_body(text)
    if truncated:
        text = body_text
    return _plain_tool_panel("run_command", text)


def _render_tool_result(
    name: str, result: dict[str, Any], code_theme: str = "ansi_dark"
) -> RenderableType | None:
    """Render a readable, truncated panel for a tool result, or None."""
    if result.get("blocked"):
        return None
    if "error" in result:
        return _plain_tool_panel(name, f"Error: {result['error']}")
    if name == "read_file":
        return _render_read_file_result(result, code_theme)
    if name == "run_command":
        return _render_run_command_result(result, code_theme)
    text = _format_tool_result(name, result)
    if not text:
        return None
    pretty = _pretty_json(text)
    body_text, _ = _truncate_body(pretty or text)
    if pretty is not None:
        return Panel(
            _PlainWrite(body_text, _make_syntax(body_text, "json", code_theme)),
            title=f"Tool result · {name}",
            border_style="blue",
            padding=(0, 1),
        )
    return _plain_tool_panel(name, body_text)
