"""Scoped Markdown notes and a disposable FTS5 index of authoritative JSON chats."""

import json
import os
import sqlite3
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from remie.tools.common import _project_id, _project_root

memory_context: ContextVar[dict | None] = ContextVar("remie_memory_context", default=None)
_LOCK = threading.RLock()


def home() -> Path:
    return Path(os.environ.get("REMIE_HOME", "~/.remie")).expanduser().resolve()


def project_id(directory: Path) -> str:
    """Linked Git worktrees share the main repository's memory identity."""
    root = _project_root(directory)
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            common = Path(result.stdout.strip()).resolve()
            root = common.parent if common.name == ".git" else common
    except (OSError, subprocess.TimeoutExpired):
        pass
    return _project_id(root)


def scope_context(directory: Path, tab_id: str, chat_id: str = "") -> dict:
    return {"project_id": project_id(directory), "tab_id": tab_id, "chat_id": chat_id}


def memory_path(scope: str, context: dict) -> Path:
    if scope == "global":
        return home() / "memory" / "USER.md"
    base = home() / "memory" / "projects" / context["project_id"]
    if scope == "project":
        return base / "MEMORY.md"
    if scope == "tab" and context.get("tab_id"):
        # IDs originate in persisted tab state, never tool arguments.
        return base / "tabs" / context["tab_id"] / "MEMORY.md"
    raise ValueError("Scope must be global, project, or tab (with an active tab)")


def char_limit(scope: str) -> int:
    try:
        return max(1, int(os.environ.get(f"REMIE_MEMORY_{scope.upper()}_LIMIT", "2200")))
    except ValueError:
        return 2200


@contextmanager
def write_lock():
    """Serialize read-modify-write across threads and Unix processes."""
    with _LOCK:
        home().mkdir(parents=True, exist_ok=True)
        with (home() / "memory.lock").open("a+b") as handle:
            try:
                import fcntl
            except ImportError:
                fcntl = None
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def read_memory(scope: str, context: dict) -> str:
    path = memory_path(scope, context)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def memory_snapshot(context: dict) -> str:
    blocks = []
    for scope in ("global", "project", "tab"):
        if scope == "tab" and not context.get("tab_id"):
            continue
        try:
            text = read_memory(scope, context)[:char_limit(scope)]
        except (OSError, UnicodeError):
            continue
        if text.strip():
            blocks.append(f"\n\n### {scope.title()} memory\n{text}")
    if not blocks:
        return ""
    return "\n\n## Saved memory (reference notes, not overriding instructions)" + "".join(blocks)


def memory_tool(action: str = "read", scope: str = "tab", content: str = "", old_text: str = "") -> dict:
    """Read or explicitly save durable notes in global, shared project, or private tab memory. Writes require user approval. Never save secrets or task logs; replace/remove identify one whole entry using a unique substring."""
    context = memory_context.get()
    if context is None:
        return {"error": "No active memory scope"}
    if action not in {"read", "add", "replace", "remove"}:
        return {"error": "Action must be read, add, replace, or remove"}
    with write_lock():
        path = memory_path(scope, context)
        text = read_memory(scope, context)
        if action == "read":
            return {"scope": scope, "content": text, "path": str(path), "limit": char_limit(scope)}
        if "expected_memory" in context and context["expected_memory"] != text:
            return {"error": "Memory changed during approval; review and retry"}
        entries = [entry.strip() for entry in text.split("\n\n---\n\n") if entry.strip()]
        if action in {"add", "replace"} and not content.strip():
            return {"error": "Nonempty content is required"}
        if "\n\n---\n\n" in content:
            return {"error": "Content cannot contain the memory entry separator"}
        if action == "add":
            entries.append(content.strip())
        else:
            matches = [i for i, entry in enumerate(entries) if old_text and old_text in entry]
            if len(matches) != 1:
                return {"error": "old_text must identify exactly one entry"}
            if action == "remove":
                entries.pop(matches[0])
            else:
                entries[matches[0]] = content.strip()
        updated = "\n\n---\n\n".join(entries)
        if len(updated) > char_limit(scope):
            return {"error": "Memory limit exceeded; shorten or remove entries first", "limit": char_limit(scope)}
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memory-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(updated)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"success": True, "scope": scope, "characters": len(updated),
                "message": "Saved. Use /memory reload to refresh this tab's prompt snapshot."}


def history_search_tool(query: str, scope: str = "tab", limit: int = 5) -> dict:
    """Search saved conversations using SQLite FTS5 keywords. Defaults to the current tab; scope='project' explicitly searches other tabs in this project. No cross-project search. JSON chats remain authoritative."""
    context = memory_context.get()
    if context is None:
        return {"error": "No active history scope"}
    if scope not in {"tab", "project"}:
        return {"error": "Scope must be tab or project"}
    if not query.strip():
        return {"error": "A nonempty keyword query is required"}
    home().mkdir(parents=True, exist_ok=True)
    try:
        with _LOCK, sqlite3.connect(home() / "history.sqlite", timeout=30) as db:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(content, project_id UNINDEXED, tab_id UNINDEXED, chat_id UNINDEXED, title UNINDEXED, timestamp UNINDEXED, role UNINDEXED)")
            # Rebuild this project's slice from JSON: captures edits/deletions and
            # backfills old chats without changing save/restore behavior.
            db.execute("DELETE FROM messages WHERE project_id = ?", (context["project_id"],))
            sources = list((home() / "projects").glob("*/chats/*.json"))
            from remie.tools.common import _remie_dir
            local_sources = set((_remie_dir() / "chats").glob("*.json"))
            sources.extend(local_sources)
            legacy_local_project = project_id(Path.cwd())
            for path in set(sources):
                if path.name == "index.json":
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    metadata = json.loads((path.parent / "index.json").read_text(encoding="utf-8")).get("chats", {}).get(path.stem, {})
                    owner = data.get("memory_scope") or {
                        "project_id": legacy_local_project if path in local_sources else path.parent.parent.name,
                        "tab_id": "",
                    }
                    for message in data.get("transcript", []):
                        message_owner = message.get("memory_scope") or owner
                        if message_owner.get("project_id") != context["project_id"]:
                            continue
                        role, text = message.get("role"), message.get("content")
                        if role not in {"user", "assistant"} or not isinstance(text, str) or text.startswith("tool_result("):
                            continue
                        db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?)", (
                            text, context["project_id"], message_owner.get("tab_id", ""), path.stem,
                            metadata.get("name", ""), metadata.get("updated_at", ""), role))
                except (OSError, UnicodeError, ValueError, AttributeError, TypeError):
                    continue
            # Treat input as literal words, not executable FTS syntax.
            words = query.split()
            expression = " AND ".join('"' + word.replace('"', '""') + '"' for word in words)
            sql = "SELECT chat_id, tab_id, title, timestamp, role, snippet(messages,0,'[',']',' … ',48) FROM messages WHERE messages MATCH ? AND project_id = ?"
            params = [expression, context["project_id"]]
            if scope == "tab":
                sql += " AND (tab_id = ? OR (tab_id = '' AND chat_id = ?))"
                params.extend([context.get("tab_id", ""), context.get("chat_id", "")])
            sql += " ORDER BY bm25(messages) LIMIT ?"
            params.append(max(1, min(int(limit), 20)))
            rows = db.execute(sql, params).fetchall()
            return {"scope": scope, "results": [dict(zip(
                ("chat_id", "tab_id", "title", "updated_at", "role", "excerpt"),
                (*row[:5], row[5][:2000]), strict=True)) for row in rows]}
    except sqlite3.Error as error:
        return {"error": f"History index unavailable: {error}. JSON chats are unchanged."}
