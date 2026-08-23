"""Project-local durable memory persistence."""

import datetime as _dt
import json
import re
import uuid
from pathlib import Path
from typing import Any

from remie.tools.common import _remie_dir

DEFAULT_MEMORY_NAME = "general"
MEMORY_INDEX_VERSION = 2
MEMORY_NAME_MAX_CHARS = 60
_UUID_STEM_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def memory_dir() -> Path:
    return _remie_dir() / "memory"


def memory_index_path() -> Path:
    return memory_dir() / "index.json"


def memory_file_path(memory_id: str) -> Path:
    return memory_dir() / f"{memory_id}.md"


def active_memory_file_path() -> Path:
    return _remie_dir() / "active_memory"


def load_memory_index() -> dict[str, Any]:
    """Return {uuid: {name, created_at}} from index.json (empty if absent)."""
    path = memory_index_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeError, json.JSONDecodeError:
        return {}
    memories = data.get("memories")
    return memories if isinstance(memories, dict) else {}


def save_memory_index(memories: dict[str, Any]) -> None:
    path = memory_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MEMORY_INDEX_VERSION, "memories": memories}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _memory_name_valid(name: str) -> bool:
    name = name.strip()
    return bool(name) and len(name) <= MEMORY_NAME_MAX_CHARS


def find_memory_by_id(memory_id: str) -> dict[str, Any] | None:
    """Return {id, name, created_at} for a uuid, or None."""
    entry = load_memory_index().get(memory_id)
    if entry is None:
        return None
    return {
        "id": memory_id,
        "name": entry.get("name", ""),
        "created_at": entry.get("created_at", ""),
    }


def find_memory_by_name(name: str) -> dict[str, Any] | None:
    """Return the memory whose name matches (case-insensitive), or None."""
    target = name.strip().lower()
    for memory_id, entry in load_memory_index().items():
        if str(entry.get("name", "")).strip().lower() == target:
            return find_memory_by_id(memory_id)
    return None


def list_memories() -> list[dict[str, Any]]:
    """Return [{id, name, created_at, chars}] sorted by name."""
    memories = []
    for memory_id, entry in load_memory_index().items():
        path = memory_file_path(memory_id)
        chars = path.stat().st_size if path.is_file() else 0
        memories.append(
            {
                "id": memory_id,
                "name": entry.get("name", ""),
                "created_at": entry.get("created_at", ""),
                "chars": chars,
            }
        )
    return sorted(memories, key=lambda item: item["name"].lower())


def create_memory(name: str) -> dict[str, Any]:
    """Create a memory with a fresh uuid; returns {id, name, created_at}."""
    name = name.strip()
    if not _memory_name_valid(name):
        raise ValueError("Memory name must be 1-60 non-empty characters")
    if find_memory_by_name(name) is not None:
        raise ValueError(f"A memory named '{name}' already exists")
    memory_id = str(uuid.uuid4())
    memories = load_memory_index()
    memories[memory_id] = {
        "name": name,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    save_memory_index(memories)
    return {
        "id": memory_id,
        "name": name,
        "created_at": memories[memory_id]["created_at"],
    }


def rename_memory(memory_id: str, name: str) -> dict[str, Any]:
    """Rename a memory while preserving its UUID and contents."""
    memory = find_memory_by_id(memory_id)
    if memory is None:
        raise ValueError(f"Unknown memory id: {memory_id}")
    name = " ".join(name.split())[:MEMORY_NAME_MAX_CHARS].rstrip()
    if not _memory_name_valid(name):
        raise ValueError("Memory name must be 1-60 non-empty characters")
    existing = find_memory_by_name(name)
    if existing is not None and existing["id"] != memory_id:
        base = name
        suffix = 2
        candidate = f"{base} {suffix}"
        while find_memory_by_name(candidate) is not None:
            suffix += 1
            suffix_text = f" {suffix}"
            candidate = f"{base[: MEMORY_NAME_MAX_CHARS - len(suffix_text)].rstrip()}{suffix_text}"
        name = candidate
    memories = load_memory_index()
    memories[memory_id]["name"] = name
    save_memory_index(memories)
    return find_memory_by_id(memory_id)  # type: ignore[return-value]


def get_active_memory_id() -> str | None:
    """Return the active memory uuid, or None when unset/invalid."""
    path = active_memory_file_path()
    try:
        memory_id = path.read_text(encoding="utf-8").strip()
    except OSError, UnicodeError:
        return None
    if not memory_id or find_memory_by_id(memory_id) is None:
        return None
    return memory_id


def set_active_memory_id(memory_id: str) -> None:
    """Persist the active memory uuid."""
    if find_memory_by_id(memory_id) is None:
        raise ValueError(f"Unknown memory id: {memory_id}")
    path = active_memory_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory_id, encoding="utf-8")


def ensure_general_memory() -> dict[str, Any]:
    """Return the memory named 'general', creating it if needed."""
    general = find_memory_by_name(DEFAULT_MEMORY_NAME)
    if general is not None:
        return general
    return create_memory(DEFAULT_MEMORY_NAME)


def delete_memory(memory_id: str) -> dict[str, Any]:
    """Remove a memory (file + index entry). If it was active, active falls
    back to the 'general' memory."""
    memory = find_memory_by_id(memory_id)
    if memory is None:
        raise ValueError(f"Unknown memory id: {memory_id}")
    was_active = get_active_memory_id() == memory_id
    path = memory_file_path(memory_id)
    if path.is_file():
        path.unlink()
    memories = load_memory_index()
    memories.pop(memory_id, None)
    save_memory_index(memories)
    if was_active:
        set_active_memory_id(ensure_general_memory()["id"])
    return memory


def _migrate_to_uuid_index() -> None:
    """One-time migration of legacy name-keyed memories to uuid-keyed files.

    Idempotent: returns immediately once index.json exists. Handles both the
    old single-file .remie/memory.md and name-keyed .remie/memory/<name>.md.
    """
    if memory_index_path().is_file():
        return
    directory = memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    memories: dict[str, Any] = {}

    # Legacy single file .remie/memory.md -> the 'general' memory.
    legacy = _remie_dir() / "memory.md"
    if legacy.is_file():
        target = find_memory_by_name(DEFAULT_MEMORY_NAME)
        memory_id = target["id"] if target else str(uuid.uuid4())
        legacy.rename(memory_file_path(memory_id))
        memories.setdefault(
            memory_id,
            {
                "name": DEFAULT_MEMORY_NAME,
                "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
        )

    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if not entry.is_file() or entry.suffix != ".md":
            continue
        stem = entry.stem
        if _UUID_STEM_RE.match(stem):
            continue  # already uuid-named
        memory_id = str(uuid.uuid4())
        entry.rename(memory_file_path(memory_id))
        memories[memory_id] = {
            "name": stem,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }

    save_memory_index(memories)

    # Migrate active_memory from a name to the corresponding uuid.
    active_path = active_memory_file_path()
    if active_path.is_file():
        raw = active_path.read_text(encoding="utf-8").strip()
        if raw and not _UUID_STEM_RE.match(raw):
            for memory_id, entry in memories.items():
                if str(entry.get("name", "")).strip().lower() == raw.lower():
                    active_path.write_text(memory_id, encoding="utf-8")
                    break


def create_launch_memory() -> dict[str, Any]:
    """Create and activate the next blank, launch-scoped memory."""
    _migrate_to_uuid_index()
    number = 1
    while find_memory_by_name(f"session {number}") is not None:
        number += 1
    memory = create_memory(f"session {number}")
    set_active_memory_id(memory["id"])
    return memory


def ensure_active_memory() -> dict[str, Any]:
    """Return the active durable-note memory, falling back to 'general'.

    Keeps the previously selected memory across launches; creates and
    activates 'general' when no valid active memory is set.
    """
    _migrate_to_uuid_index()
    active = get_active_memory_id()
    if active is not None:
        memory = find_memory_by_id(active)
        if memory is not None:
            return memory
    memory = ensure_general_memory()
    set_active_memory_id(memory["id"])
    return memory
