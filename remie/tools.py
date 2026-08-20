"""Tool implementations for the Remie coding assistant."""

import datetime as _dt
import difflib
import fnmatch
import inspect
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any


def _remie_dir() -> Path:
    """Directory holding per-project memory and session data."""
    return Path.cwd() / ".remie"


DEFAULT_MEMORY_NAME = "general"
MEMORY_INDEX_VERSION = 2
MEMORY_NAME_MAX_CHARS = 60
_UUID_STEM_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


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
    except (OSError, UnicodeError, json.JSONDecodeError):
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
    return {"id": memory_id, "name": entry.get("name", ""), "created_at": entry.get("created_at", "")}


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
    return {"id": memory_id, "name": name, "created_at": memories[memory_id]["created_at"]}


def get_active_memory_id() -> str | None:
    """Return the active memory uuid, or None when unset/invalid."""
    path = active_memory_file_path()
    try:
        memory_id = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
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


def session_file_path() -> Path:
    return _remie_dir() / "session.json"


def resolve_abs_path(path_str: str) -> Path:
    """
    file.py -> /Users/home/username/project/file.py
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def read_file_tool(filename: str) -> dict[str, Any]:
    """
    Gets the full content of a file provided by the user.
    :param filename: The name of the file to read.
    :return: The full content of the file.
    """
    full_path = resolve_abs_path(filename)
    with open(str(full_path), "r") as f:
        content = f.read()
    return {"file_path": str(full_path), "content": content}


def list_files_tool(path: str) -> dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    :param path: The path to a directory to list files from.
    :return: A list of files in the directory.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append(
            {"filename": item.name, "type": "file" if item.is_file() else "dir"}
        )
    return {"path": str(full_path), "files": all_files}


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

IGNORED_SUFFIXES = (".egg-info",)


def _is_ignored(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(IGNORED_SUFFIXES)


GLOB_MAX_RESULTS = 200
TREE_MAX_ENTRIES = 500
TREE_MAX_DEPTH = 6


def glob_files_tool(pattern: str, path: str = ".") -> dict[str, Any]:
    """
    Finds files matching a glob pattern, searching recursively.
    :param pattern: The glob pattern to match (e.g. '*.py' or 'src/*.py').
    :param path: The directory to search from (defaults to the project).
    :return: A list of matching file paths relative to the search root.
    """
    full_path = resolve_abs_path(path)
    matches: list[str] = []
    for root, dirs, files in os.walk(full_path):
        dirs[:] = sorted(d for d in dirs if not _is_ignored(d))
        rel_root = os.path.relpath(root, full_path)
        for filename in sorted(files):
            rel_path = os.path.join(rel_root, filename) if rel_root != "." else filename
            rel_path = rel_path.replace(os.sep, "/")
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel_path, pattern):
                matches.append(rel_path)
            if len(matches) >= GLOB_MAX_RESULTS:
                break
        if len(matches) >= GLOB_MAX_RESULTS:
            break
    truncated = len(matches) >= GLOB_MAX_RESULTS
    return {
        "path": str(full_path),
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
        "truncated": truncated,
    }


def _tree_walk(
    directory: Path,
    prefix: str,
    depth: int,
    max_depth: int,
    lines: list[str],
    limit: int,
) -> None:
    entries = sorted(
        (entry for entry in directory.iterdir() if not _is_ignored(entry.name)),
        key=lambda entry: (not entry.is_dir(), entry.name.lower()),
    )
    for index, entry in enumerate(entries):
        if len(lines) >= limit:
            return
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir() and depth < max_depth:
            _tree_walk(
                entry,
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                max_depth,
                lines,
                limit,
            )


def tree_files_tool(path: str = ".", max_depth: int = 3) -> dict[str, Any]:
    """
    Shows the directory tree of a path.
    :param path: The directory to show the tree of (defaults to the project).
    :param max_depth: How many levels of subdirectories to descend into (max 6).
    :return: The tree rendered as text.
    """
    full_path = resolve_abs_path(path)
    depth = max(1, min(int(max_depth), TREE_MAX_DEPTH))
    lines = [f"{full_path.name}/"]
    _tree_walk(full_path, "", 1, depth, lines, TREE_MAX_ENTRIES)
    truncated = len(lines) >= TREE_MAX_ENTRIES
    if truncated:
        lines.append("[... truncated ...]")
    return {
        "path": str(full_path),
        "tree": "\n".join(lines),
        "truncated": truncated,
    }


DIFF_MAX_CHARS = 4000


def _make_diff(original: str, edited: str, path: str) -> str:
    """
    Build a unified diff between two file contents, truncated to DIFF_MAX_CHARS.
    """
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            edited.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    diff = "".join(lines)
    if len(diff) > DIFF_MAX_CHARS:
        diff = diff[:DIFF_MAX_CHARS].rstrip("\n") + "\n...diff truncated...\n"
    return diff


def edit_file_tool(path: str, old_str: str, new_str: str) -> dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file. If old_str is empty,
    create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path to the file, the action taken, and a diff.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        original = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
        full_path.write_text(new_str, encoding="utf-8")
        return {
            "path": str(full_path),
            "action": "created_file",
            "diff": _make_diff(original, new_str, str(full_path)),
        }
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {"path": str(full_path), "action": "old_str not found"}
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {
        "path": str(full_path),
        "action": "edited",
        "diff": _make_diff(original, edited, str(full_path)),
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


#: Maximum seconds a shell command may run before it is killed. Configurable via
#: REMIE_COMMAND_TIMEOUT (e.g. long-running test suites).
RUN_COMMAND_TIMEOUT = _env_int("REMIE_COMMAND_TIMEOUT", 30)
RUN_COMMAND_MAX_OUTPUT = 30_000
TIMED_OUT_EXIT_CODE = 124


def _truncate_stream(stream: str, limit: int) -> str:
    """Truncate a stream to limit chars, keeping a marker on the final line."""
    if len(stream) <= limit:
        return stream
    return stream[:limit].rstrip("\n") + "\n[output truncated]\n"


# --- Command safety ---------------------------------------------------------
#
# run_command_tool refuses to execute commands that match any of the patterns
# below. Each entry is a (regex, human-readable reason) pair. The regex is
# matched (case-insensitively) against the whole command string, so commands
# chained with ;, &&, or | are checked too.

DANGEROUS_COMMAND_PATTERNS: list[tuple[str, str]] = [
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?rm\s+(?:-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(?:--\s+)?"
        r"(?:.*\s)?(?:/|/\*|~|~/|~/\*|\.\.|\.|\*)(?:\s|$)",
        "recursive forced delete of the filesystem root, home, current, or parent directory",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?(?:mkfs\S*|shred)\b",
        "disk formatting or permanent file shredding",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?(?:fdisk|parted|gdisk|sfdisk)\s+(?!-l\b|-h\b|--help\b|--list\b|--version\b)\S+",
        "disk partitioning",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?(?:shutdown|reboot|poweroff|halt)\b",
        "system shutdown or reboot",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?chmod\s+-[a-zA-Z]*r[a-zA-Z]*\s+[0-7]{3,4}\s+(?:/|~)(?:\s|$)",
        "recursive chmod on the filesystem root or home directory",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?chown\s+-[a-zA-Z]*r[a-zA-Z]*\s+\S+\s+(?:/|~)(?:\s|$)",
        "recursive chown on the filesystem root or home directory",
    ),
    (
        r"\(\)\s*\{\s*:\s*\|",
        "fork bomb",
    ),
    (
        r"(^|[;&|]\s*)(?:curl|wget)\s+[^\n|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh)\b",
        "piping a downloaded script directly into a shell",
    ),
]

#: Writing bytes to these devices is harmless, so `dd of=/dev/null` etc. stays
#: allowed even though `dd of=/dev/sda` is blocked.
_DD_SAFE_DEVICES = {"null", "zero", "random", "urandom", "stdin", "stdout"}


def get_custom_blocked_commands() -> list[str]:
    """
    Extra blocked substrings from the REMIE_BLOCKED_COMMANDS environment
    variable (comma-separated, case-insensitive). Example:
    REMIE_BLOCKED_COMMANDS="git push --force,aws s3 rm"
    """
    return [
        item.strip().lower()
        for item in os.environ.get("REMIE_BLOCKED_COMMANDS", "").split(",")
        if item.strip()
    ]


def get_blocked_command_reason(command: str) -> str | None:
    """
    Return why `command` is blocked, or None if it may be executed.

    Blocks command strings that match a destructive pattern (recursive
    deletion of / or ~, disk formatting/partitioning, shutdown, fork bombs,
    piping a downloaded script into a shell, ...) as well as any custom
    substrings listed in REMIE_BLOCKED_COMMANDS.
    """
    low = command.lower().strip()
    for pattern, reason in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, low):
            return reason
    # `dd` is only blocked when it writes to a raw block device such as
    # /dev/sda; safe devices like /dev/null stay allowed.
    if re.search(r"(^|[;&|]\s*)(?:sudo\s+)?dd\b", low):
        match = re.search(r"\bof\s*=\s*([^\s;&|]+)", low)
        if match:
            target = match.group(1)
            if target.startswith("/dev/"):
                device = target[len("/dev/") :].split("/", 1)[0]
                if device not in _DD_SAFE_DEVICES:
                    return "dd writing directly to a raw block device"
    for blocked in get_custom_blocked_commands():
        if blocked in low:
            return f"matches blocked pattern '{blocked}'"
    return None


def run_command_tool(command: str, cwd: str = ".") -> dict[str, Any]:
    """
    Runs a shell command in the project and returns its exit code and output.
    Destructive commands (rm -rf on / or ~, disk formatting, shutdown, fork
    bombs, curl|sh, ...) are blocked before execution.
    :param command: The shell command to run.
    :param cwd: The directory to run the command in (defaults to the project).
    :return: A dictionary with the exit code, stdout, stderr, cwd, and whether it timed out.
    """
    full_path = resolve_abs_path(cwd)
    reason = get_blocked_command_reason(command)
    if reason is not None:
        return {
            "command": command,
            "cwd": str(full_path),
            "blocked": True,
            "reason": reason,
            "exit_code": None,
            "stdout": "",
            "stderr": f"Command blocked: {reason}",
            "timed_out": False,
            "truncated": False,
        }
    try:
        result = subprocess.run(
            command,
            cwd=str(full_path),
            shell=True,
            capture_output=True,
            text=True,
            timeout=RUN_COMMAND_TIMEOUT,
            input=None,
        )
        exit_code = result.returncode
        stdout, stderr = result.stdout, result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        exit_code = TIMED_OUT_EXIT_CODE
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        timed_out = True
        hint = (
            f"\n\n[command timed out after {RUN_COMMAND_TIMEOUT}s] The command was "
            "killed before finishing. Do not retry the exact same command "
            "unchanged; instead reduce its scope (fewer files, shorter input, "
            "one step at a time) or increase REMIE_COMMAND_TIMEOUT and try again."
        )
        budget = max(RUN_COMMAND_MAX_OUTPUT - len(stdout) - len(hint) - 10, 1)
        stderr = _truncate_stream(stderr, budget).rstrip("\n") + hint

    truncated = False
    if len(stdout) + len(stderr) > RUN_COMMAND_MAX_OUTPUT:
        truncated = True
        budget = max(RUN_COMMAND_MAX_OUTPUT - 40, 1)
        if stdout and stderr:
            stdout_share = int(budget * len(stdout) / (len(stdout) + len(stderr)))
            stdout = _truncate_stream(stdout, stdout_share)
            stderr = _truncate_stream(stderr, budget - stdout_share)
        elif stdout:
            stdout = _truncate_stream(stdout, budget)
        else:
            stderr = _truncate_stream(stderr, budget)

    return {
        "command": command,
        "cwd": str(full_path),
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "truncated": truncated,
    }


def ask_user_tool(
    question: str, options: list[str] | None = None
) -> dict[str, Any]:
    """
    Asks the user a question and waits for their answer. Use this when you need
    a decision or clarification from the user instead of guessing.
    :param question: The question to ask the user.
    :param options: Optional list of predefined choices to offer.
    :return: The user's answer.
    """
    return {"question": question, "options": options or []}


def memory_tool(
    action: str, text: str = "", id: str = "", name: str = ""
) -> dict[str, Any]:
    """
    Persists a note to a project memory file (.remie/memory/<uuid>.md), reads
    it, clears it, or deletes the whole memory. Use to remember durable facts,
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


TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
    "run_command": run_command_tool,
    "glob_files": glob_files_tool,
    "tree_files": tree_files_tool,
    "ask_user": ask_user_tool,
    "memory": memory_tool,
}

TOOL_SUMMARIES = {
    "read_file": "read a file",
    "list_files": "list the files in a directory",
    "edit_file": "edit a file",
    "run_command": "run a shell command",
    "glob_files": "find files matching a glob pattern",
    "tree_files": "show the directory tree",
    "ask_user": "ask the user a question",
    "memory": "saving or recalling a memory note",
}


def get_tool_summary(name: str) -> str:
    """
    Return a short human-readable summary for a tool name.
    """
    return TOOL_SUMMARIES.get(name, name)


def get_tool_str_representation(tool_name: str) -> str:
    tool = TOOL_REGISTRY[tool_name]
    return f"""
    Name: {tool_name}
    Description: {tool.__doc__}
    Signature: {inspect.signature(tool)}
    """
