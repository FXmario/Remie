"""Regression checks for removal of persistent agent notes."""

from remie.prompts import build_system_prompt
from remie.tools import TOOL_PARAMETERS, TOOL_REGISTRY, get_tool_schemas
from remie.tools.executor import run_tool
from remie.tui.screens.open import OpenScreen
from remie.tui.slash_commands import resolve_slash_command


def test_memory_is_not_a_callable_tool():
    assert "memory" not in TOOL_REGISTRY
    assert "memory" not in TOOL_PARAMETERS
    assert all(schema["name"] != "memory" for schema in get_tool_schemas())
    assert run_tool("memory", {"action": "add", "text": "note"})["action"] == "unknown_tool_memory"


def test_prompt_ignores_existing_memory_files(tmp_path, monkeypatch):
    import remie.tools

    state = tmp_path / "state"
    notes = state / "memory"
    notes.mkdir(parents=True)
    (notes / "example.md").write_text("PRIVATE_DURABLE_NOTE", encoding="utf-8")
    (state / "active_memory").write_text("example", encoding="utf-8")
    monkeypatch.setattr(remie.tools, "_remie_dir", lambda: state)
    monkeypatch.chdir(tmp_path)

    prompt = build_system_prompt()
    assert "PRIVATE_DURABLE_NOTE" not in prompt
    assert "Agent memory" not in prompt
    assert "'memory' tool" not in prompt


def test_memory_picker_is_unavailable():
    assert resolve_slash_command("/memories") is None
    assert "open-memories" not in OpenScreen._TAB_CONTENT
