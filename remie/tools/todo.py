"""Lazy TODO adapter."""


def todo_tool(action: str = "read", title: str = "", item_id: str = "", status: str = "", summary: str = "") -> dict:
    """For long tasks, add the full initial checklist before implementation, not one step at a time. Follow it in order with one step in_progress and future steps pending; revise the checklist when the plan changes. Read on resume to inspect saved outputs. Add a title, then update by item_id with pending/in_progress/completed/blocked status. Completion requires an evidence summary and recorded tool output. No approval gate for ordinary tools."""
    from remie.storage.todo import todo_tool as implementation
    return implementation(action, title, item_id, status, summary)
