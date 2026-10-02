"""Allocate separate working directories for tabs sharing a project."""

import subprocess
import uuid
from pathlib import Path


def workspace_label(directory: Path) -> str:
    """Identify a tab's worktree by branch, falling back to its directory name."""
    try:
        branch = subprocess.run(
            ["git", "-C", str(directory), "symbolic-ref", "--quiet", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        branch = None
    if branch is not None and branch.returncode == 0 and branch.stdout.strip():
        return branch.stdout.strip()
    return directory.name


class WorkspaceError(Exception):
    """A separate tab workspace could not be created."""


def separate_workspace(directory: Path, tab_id: str) -> Path:
    """Create a Git worktree or an empty subdirectory for a second tab.

    Git worktrees use a branch from HEAD (uncommitted changes are not copied).
    Worktrees live beside the repository rather than inside its working tree.
    Paths and branch names include the tab UUID so existing directories are
    never reused or overwritten; closing a tab does not delete its files.
    """
    directory = directory.resolve()
    workspace_id = f"{tab_id}-{uuid.uuid4().hex[:8]}"
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        result = None  # Git unavailable; treat this as a non-Git directory.
    except OSError as error:
        raise WorkspaceError(f"Could not check Git status: {error}") from error
    if result is None or result.returncode != 0:
        target = directory / f"remie-tab-{workspace_id}"
        try:
            target.mkdir()  # Never claim an existing directory.
        except OSError as error:
            raise WorkspaceError(f"Could not create {target}: {error}") from error
        return target

    root = Path(result.stdout.strip()).resolve()
    target = root.parent / f"{root.name}-remie-tab-{workspace_id}"
    branch = f"remie/tab-{workspace_id}"
    if target.exists():
        raise WorkspaceError(f"Workspace already exists: {target}")
    try:
        created = subprocess.run(
            ["git", "-C", str(root), "worktree", "add", "-b", branch,
             str(target), "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError as error:
        raise WorkspaceError(f"Could not create Git worktree: {error}") from error
    if created.returncode != 0:
        raise WorkspaceError(
            f"Could not create Git worktree: {created.stderr.strip() or created.stdout.strip()}"
        )
    workspace = target / directory.relative_to(root)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _git(directory: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args], capture_output=True,
            text=True, check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkspaceError(str(error)) from error
    if result.returncode:
        raise WorkspaceError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def list_worktrees(directory: Path) -> list[dict]:
    """Read NUL-delimited records so paths containing whitespace remain intact."""
    records = []
    for block in _git(directory, "worktree", "list", "--porcelain", "-z").split("\0\0"):
        fields = {}
        for entry in block.split("\0"):
            key, _, value = entry.partition(" ")
            if key:
                fields[key] = value
        if "worktree" not in fields:
            continue
        path = Path(fields["worktree"]).resolve()
        try:
            dirty = bool(_git(path, "status", "--porcelain", "--untracked-files=all").strip())
        except WorkspaceError:
            dirty = True
        records.append({"path": path, "branch": fields.get("branch", "detached").removeprefix("refs/heads/"),
                        "main": not records, "dirty": dirty,
                        "locked": "locked" in fields})
    return records


def linked_worktree(directory: Path) -> dict | None:
    try:
        root = Path(_git(directory, "rev-parse", "--show-toplevel").strip()).resolve()
        return next((item for item in list_worktrees(directory)
                     if item["path"] == root and not item["main"]), None)
    except WorkspaceError:
        return None


def remove_worktree(directory: Path) -> None:
    item = linked_worktree(directory)
    if item is None:
        raise WorkspaceError("Only linked worktrees can be deleted; the main worktree is protected.")
    if Path.cwd().resolve().is_relative_to(item["path"]):
        raise WorkspaceError("This worktree contains Remie’s launch directory. Restart Remie from the main worktree before deleting it.")
    if item["dirty"] or item["locked"]:
        raise WorkspaceError("Worktree has uncommitted/untracked files or is locked. Save or clean it first.")
    main = list_worktrees(directory)[0]["path"]
    _git(main, "worktree", "remove", str(item["path"]))


def name_worktree(directory: Path, title: str) -> Path:
    """Move a generated workspace once; callers persist the naming decision."""
    import re

    item = linked_worktree(directory)
    if item is None or "-remie-tab-" not in item["path"].name or not item["branch"].startswith("remie/tab-"):
        return directory
    root = item["path"]
    if Path.cwd().resolve().is_relative_to(root):
        return directory  # Keep the process cwd and project-scoped storage stable.
    main = list_worktrees(directory)[0]["path"]
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:64].rstrip("-") or "chat"
    base = root.parent / f"{main.name}-{slug}"
    target = base
    number = 2
    while target.exists():
        target = base.with_name(f"{base.name}-{number}")
        number += 1
    _git(main, "worktree", "move", str(root), str(target))
    return target / directory.relative_to(root)
