from rich.console import Console

from remie.tui.render import _format_tool_call, _render_tool_call


def test_shell_command_is_verbatim():
    command = 'for file in *.py; do\n  echo "$file"\ndone\n'
    assert _format_tool_call("run_command", {"command": command, "cwd": "/tmp"}) == command


def test_named_arguments_and_empty_call():
    assert _format_tool_call("read_file", {"filename": "a.py"}) == "read_file(filename: 'a.py')"
    assert _format_tool_call("tab_status", {}) == "tab_status()"


def test_values_preserve_types_and_order():
    args = {"options": ["yes", "no"], "enabled": True, "count": 2, "missing": None}
    assert _format_tool_call("example", args) == (
        "example(options: ['yes', 'no'], enabled: True, count: 2, missing: None)"
    )


def test_long_calls_use_multiple_lines():
    value = "x" * 100
    assert _format_tool_call("edit_file", {"path": "a", "new_str": value}) == (
        f"edit_file(\n  path: 'a',\n  new_str: '{value}'\n)"
    )


def test_literal_markup_and_narrow_wrapping():
    console = Console(width=30, record=True, color_system=None)
    call = _render_tool_call("run_command", {"command": "echo [bold]hello[/bold] " + "x" * 60})
    call.expanded = True
    console.print(call)
    output = console.export_text()
    assert "[bold]hello[/bold]" in output
    assert output.count("x") == 60


def test_invalid_shell_argument_uses_function_format():
    assert _format_tool_call("run_command", {"command": None}) == "run_command(command: None)"


def test_disclosure_starts_collapsed():
    console = Console(record=True, color_system=None)
    call = _render_tool_call("read_file", {"filename": "secret.py"})
    console.print(call)
    output = console.export_text()
    assert "▶ Agent" in output
    assert "secret.py" not in output


def test_click_toggle_preserves_other_calls_results_and_stream():
    import asyncio
    from textual.app import App, ComposeResult
    from remie.tui.widgets import StreamingRichLog

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield StreamingRichLog(wrap=True)

    async def exercise():
        app = TestApp()
        async with app.run_test(size=(60, 20)) as pilot:
            log = app.query_one(StreamingRichLog)
            first = _render_tool_call("read_file", {"filename": "first.py"})
            second = _render_tool_call("run_command", {"command": "echo second"})
            log.write(first)
            log.write(second)
            log.write("existing result")
            log.begin_stream()
            log.update_stream("streaming answer")
            await pilot.pause()
            old_start = log._stream_start
            await pilot.click(log, offset=(0, 0))
            await pilot.pause()
            assert first.expanded
            assert not second.expanded
            assert log._stream_start > old_start
            text = "\n".join(line.text for line in log.lines)
            assert "first.py" in text
            assert "echo second" not in text
            assert "existing result" in text
            log.update_stream("updated answer")
            await pilot.click(log, offset=(0, 0))
            await pilot.pause()
            assert not first.expanded
            assert log._stream_start == old_start
            assert "first.py" not in "\n".join(line.text for line in log.lines)
            log.toggle_tool_call(second.token)
            assert second.expanded
            assert "echo second" in "\n".join(line.text for line in log.lines)
            log.clear()
            assert not log._tool_calls
            assert log._stream_start is None

    asyncio.run(exercise())


def test_expanded_details_highlight_and_preserve_metadata():
    for theme in ("ansi_dark", "github-light"):
        for name, args in (
            ("run_command", {"command": 'echo "$HOME" # comment\n'}),
            ("read_file", {"filename": "file.py"}),
        ):
            call = _render_tool_call(name, args, theme)
            call.expanded = True
            console = Console()
            detail = list(call.__rich_console__(console, console.options))[1]
            assert detail.plain == _format_tool_call(name, args)
            segments = list(console.render(detail))
            assert any(segment.style and segment.style.color for segment in segments)
            assert all(
                segment.style and segment.style.meta.get("tool_block") == call.token
                for segment in segments if segment.text.strip()
            )


def test_highlighting_failure_falls_back_to_literal_text(monkeypatch):
    from remie.tui import render

    def fail(*args, **kwargs):
        raise ValueError("highlight unavailable")

    monkeypatch.setattr(render.Syntax, "highlight", fail)
    call = _render_tool_call("run_command", {"command": "echo [bold]literal[/bold]"})
    call.expanded = True
    console = Console()
    detail = list(call.__rich_console__(console, console.options))[1]
    assert detail.plain == "echo [bold]literal[/bold]"
