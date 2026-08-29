"""Model-callable adapter around durable memory storage."""

import datetime as _dt
from typing import Any

from remie.storage.memories import (
    _migrate_to_uuid_index,
    create_memory,
    delete_memory,
    ensure_general_memory,
    find_memory_by_id,
    find_memory_by_name,
    get_active_memory_id,
    list_memories,
    memory_file_path,
    set_active_memory_id,
)


def memory_tool(
    action: str, text: str = "", id: str = "", name: str = ""
) -> dict[str, Any]:
    """
    Persists a note to a project memory file under
    ~/.remie/projects/<project-id>/memory/<uuid>.md, reads it, clears it, or
    deletes the whole memory. Use to remember durable facts,
    decisions, user preferences, and open tasks across sessions; do not log
    routine progress.
    :param action: 'add' to append a timestamped note, 'read' to return all
        notes, 'clear' to wipe the memory, 'delete' to remove the memory, or
        'list' to list memories as [{id, name, chars}].
    :param text: The note to add (ignored unless action is 'add').
    :param id: The memory uuid to target; wins over `name`.
    :param name: The memory name to target (or create on 'add'); defaults to
        the active launch memory when both id and name are empty.
    :return: A dictionary with the action taken, the memory id/name, and the
        full memory content where relevant.
    """
    _migrate_to_uuid_index()
    action = (action or "read").strip().lower()
    if action == "list":
        return {"action": "list", "memories": list_memories()}

    def _resolve() -> dict[str, Any] | None:
        if id:
            return find_memory_by_id(id)
        if name:
            return find_memory_by_name(name)
        active = get_active_memory_id()
        return find_memory_by_id(active) if active else None

    if action == "delete":
        memory = _resolve()
        if memory is None:
            return {"action": "delete", "error": "memory not found"}
        delete_memory(memory["id"])
        return {"action": "delete", "id": memory["id"], "name": memory["name"]}

    memory = _resolve()
    if memory is None:
        if action == "add":
            if name.strip():
                memory = create_memory(name.strip())
            else:
                memory = ensure_general_memory()
            if not get_active_memory_id():
                set_active_memory_id(memory["id"])
        else:
            memory = ensure_general_memory()
    path = memory_file_path(memory["id"])
    if action == "add":
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.now().isoformat(timespec="seconds")
        line = f"- [{timestamp}] {text.strip()}"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        content = f"{existing.rstrip()}\n{line}\n" if existing.strip() else f"{line}\n"
        path.write_text(content, encoding="utf-8")
        return {
            "action": "add",
            "id": memory["id"],
            "name": memory["name"],
            "file": str(path),
            "content": content,
        }
    if action == "clear":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return {
            "action": "clear",
            "id": memory["id"],
            "name": memory["name"],
            "file": str(path),
            "content": "",
        }
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "action": "read",
        "id": memory["id"],
        "name": memory["name"],
        "file": str(path),
        "content": content,
    }
