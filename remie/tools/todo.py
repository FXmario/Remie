"""Lazy TODO adapter."""


def todo_tool(action: str = "read", title: str = "", item_id: str = "", status: str = "", summary: str = "") -> dict:
    """Persist steps for long tasks only. Read on resume to inspect saved outputs. Add a title, then update by item_id with pending/in_progress/completed/blocked status. Completion requires an evidence summary and recorded tool output. No approval gate for ordinary tools."""
    from remie.storage.todo import todo_tool as implementation
    return implementation(action, title, item_id, status, summary)
