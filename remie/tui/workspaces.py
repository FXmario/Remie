"""Allocate separate working directories for tabs sharing a project."""

import subprocess
import uuid
from pathlib import Path


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
