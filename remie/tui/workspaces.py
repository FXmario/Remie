"""Allocate separate working directories for tabs sharing a project."""

import re
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


def _branch_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:64].rstrip("-") or "chat"


def _folder_title(value: str) -> str:
    # Preserve readable titles, but never allow path separators or control characters.
    return re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]+', "-", value).strip(" .")[:100] or "New chat"


def separate_workspace(directory: Path, tab_id: str) -> Path:
    """Create a Git worktree or an empty subdirectory for a second tab.

    Git worktrees use a branch from HEAD (uncommitted changes are not copied).
    Worktrees live beside the repository rather than inside its working tree.
    Paths and branch names include the tab UUID so existing directories are
    never reused or overwritten; closing a tab keeps its files unless the user
    explicitly chooses to delete the linked worktree.
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
    main = list_worktrees(root)[0]["path"]
    target = main.parent / f"remie|{main.name} - New chat-{workspace_id}"
    branch = f"remie|{_branch_slug(main.name)}-new-chat-{workspace_id}"
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
    """Delete a linked worktree and its root folder, never the main worktree."""
    item = linked_worktree(directory)
    if item is None:
        raise WorkspaceError("Only linked worktrees can be deleted; the main worktree is protected.")
    if Path.cwd().resolve().is_relative_to(item["path"]):
        raise WorkspaceError("This worktree contains Remie’s launch directory. Restart Remie from the main worktree before deleting it.")
    if item["dirty"] or item["locked"]:
        raise WorkspaceError("Worktree has uncommitted/untracked files or is locked. Save or clean it first.")
    main = list_worktrees(directory)[0]["path"]
    _git(main, "worktree", "remove", str(item["path"]))
    if item["path"].exists() or item["path"].is_symlink():
        raise WorkspaceError(
            f"Git worktree removal left its folder behind: {item['path']}. "
            "Check the folder before removing it manually."
        )


def name_worktree(directory: Path, title: str) -> Path:
    """Move a generated workspace once; callers persist the naming decision."""
    item = linked_worktree(directory)
    if item is None:
        return directory
    generated = ("-remie-tab-" in item["path"].name and item["branch"].startswith("remie/tab-"))
    generated = generated or (item["path"].name.startswith("remie|")
                              and " - New chat-" in item["path"].name
                              and item["branch"].startswith("remie|"))
    if not generated:
        return directory
    root = item["path"]
    if Path.cwd().resolve().is_relative_to(root):
        return directory  # Keep the process cwd and project-scoped storage stable.
    main = list_worktrees(directory)[0]["path"]
    base = root.parent / f"remie|{main.name} - {_folder_title(title)}"
    branch_base = f"remie|{_branch_slug(main.name)}-{_branch_slug(title)}"
    branches = set(_git(main, "for-each-ref", "--format=%(refname:short)", "refs/heads/").splitlines())
    target, branch = base, branch_base
    number = 2
    while target.exists() or branch in branches:
        target = base.with_name(f"{base.name}-{number}")
        branch = f"{branch_base}-{number}"
        number += 1
    _git(root, "branch", "-m", branch)
    try:
        _git(main, "worktree", "move", str(root), str(target))
    except WorkspaceError:
        _git(root, "branch", "-m", item["branch"])
        raise
    return target / directory.relative_to(root)


def ensure_merge_ready(directory: Path) -> None:
    """Refuse detached HEADs and unfinished Git operations."""
    _git(directory, "symbolic-ref", "--quiet", "HEAD")
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply"):
        path = Path(_git(directory, "rev-parse", "--git-path", marker).strip())
        if not path.is_absolute():
            path = directory / path
        if path.exists():
            raise WorkspaceError(f"Finish the existing Git operation in {directory} first.")


def worktree_dirty(directory: Path) -> bool:
    return bool(_git(directory, "status", "--porcelain", "--untracked-files=all").strip())


def commit_message_context(directory: Path) -> str:
    """Bounded tracked diffs and status, including names of untracked files."""
    status = _git(directory, "status", "--short", "--untracked-files=all")
    staged = _git(directory, "diff", "--cached", "--no-ext-diff", "--no-textconv")
    unstaged = _git(directory, "diff", "--no-ext-diff", "--no-textconv")
    return (f"Status (including untracked files):\n{status[:12000]}\n"
            f"Staged changes:\n{staged[:24000]}\n"
            f"Unstaged changes:\n{unstaged[:24000]}")


def commit_worktree(directory: Path, message: str) -> None:
    """Explicitly approved snapshot, including non-ignored untracked files."""
    if not message.strip():
        raise WorkspaceError("A commit message is required.")
    ensure_merge_ready(directory)
    _git(directory, "add", "--all")
    _git(directory, "commit", "-m", message.strip())


def merge_worktree(source: Path, main: Path) -> None:
    """Merge a clean source HEAD into the main worktree's current branch."""
    for directory in (source, main):
        ensure_merge_ready(directory)
        if worktree_dirty(directory):
            raise WorkspaceError(f"Commit changes in {directory} before merging.")
    revision = _git(source, "rev-parse", "HEAD").strip()
    try:
        _git(main, "merge", "--no-edit", revision)
    except WorkspaceError as error:
        raise WorkspaceError(
            f"Merge failed in {main}: {error}\n"
            "The tab and worktree were kept. If Git reports conflicts, resolve and "
            "commit them there, or run git merge --abort before retrying."
        ) from error
