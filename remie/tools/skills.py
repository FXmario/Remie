"""Activation tool shared by native and text-protocol providers."""
from pathlib import Path
from remie.skills import activate
from remie.tools.common import tool_working_directory


def activate_skill_tool(name: str) -> dict:
    """Load a discovered skill's instructions and resource list, without executing scripts."""
    return activate(name, tool_working_directory.get() or Path.cwd())
