"""Lazy adapters keep storage independent of registry initialization."""


def memory_tool(action: str = "read", scope: str = "tab", content: str = "", old_text: str = "") -> dict:
    """Read or explicitly update durable global, shared project, or private tab notes. Writes require approval. Never save secrets or temporary task logs. Replace/remove use a unique substring to identify a whole entry."""
    from remie.storage.memory import memory_tool as implementation
    return implementation(action or "read", scope or "tab", content or "", old_text or "")


def history_search_tool(query: str, scope: str = "tab", limit: int = 5) -> dict:
    """Search saved messages with FTS5 keywords. Default: current tab only. Use scope='project' only when the user explicitly requests cross-tab recall. Never searches other projects."""
    from remie.storage.memory import history_search_tool as implementation
    return implementation(query or "", scope or "tab", 5 if limit is None else limit)
