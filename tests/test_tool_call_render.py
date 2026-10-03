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
    console.print(_render_tool_call("run_command", {"command": "echo [bold]hello[/bold] " + "x" * 60}))
    output = console.export_text()
    assert "[bold]hello[/bold]" in output
    assert output.count("x") == 60


def test_invalid_shell_argument_uses_function_format():
    assert _format_tool_call("run_command", {"command": None}) == "run_command(command: None)"
