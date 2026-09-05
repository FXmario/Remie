"""Read-only cross-tab status tool marker."""


def tab_status_tool() -> dict:
    """Return a current, directory-scoped summary of other Remie tabs."""
    return {"action": "tab_status_interactive"}
