"""Shared helpers for Remie's tool implementations."""

import hashlib
import json
import os
import re
import shutil
import subprocess  # noqa: F401 -- re-exported; tests patch remie.tools.subprocess.run
from pathlib import Path
from typing import Any


def _project_root(start: Path | None = None) -> Path:
    """Return the canonical project root, preferring the nearest Git root."""
    current = (start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        # ``.git`` may be a directory or a file for linked worktrees.
        if (candidate / ".git").exists():
            return candidate
    return current


def _project_id(project_root: Path) -> str:
    """Build a readable, collision-resistant id for an absolute project path."""
    canonical = project_root.expanduser().resolve()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", canonical.name).strip("-._")
    if not name:
        name = "root"
    digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:12]
    return f"{name}-{digest}"


def _migrate_project_state(legacy: Path, destination: Path) -> bool:
    """Safely copy legacy project state and remove it only after verification.

    Returns ``False`` on a conflict or I/O failure, allowing the caller to use
    the legacy directory for the current run instead of hiding existing data.
    """
    if not legacy.is_dir():
        return True
    try:
        for source in legacy.rglob("*"):
            relative = source.relative_to(legacy)
            target = destination / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or source.read_bytes() != target.read_bytes():
                    return False
            else:
                shutil.copy2(source, target)

        # Verify every source file before deleting anything.
        for source in legacy.rglob("*"):
            if source.is_file():
                target = destination / source.relative_to(legacy)
                if not target.is_file() or source.read_bytes() != target.read_bytes():
                    return False
        shutil.rmtree(legacy)
        (destination / ".migrated-from-project-dir").write_text(
            "Legacy project-local .remie data migrated here.\n", encoding="utf-8"
        )
        return True
    except OSError:
        return False


def _remie_dir() -> Path:
    """Return this project's state directory under ``~/.remie/projects``.

    ``REMIE_HOME`` overrides ``~/.remie``. Existing project-local ``.remie``
    data is migrated lazily; if migration cannot complete safely, the legacy
    directory remains active for that run.

    Resolved through the package namespace at call time so tests that
    monkeypatch ``remie.tools._remie_dir`` continue to affect all submodules.
    """
    import sys

    package = sys.modules.get("remie.tools")
    if package is not None:
        override = package.__dict__.get("_remie_dir")
        if override is not None and override is not _remie_dir:
            return override()

    project_root = _project_root()
    state_home = Path(os.environ.get("REMIE_HOME", "~/.remie")).expanduser()
    destination = state_home / "projects" / _project_id(project_root)
    legacy = project_root / ".remie"
    if legacy != destination and legacy.is_dir():
        if not _migrate_project_state(legacy, destination):
            return legacy
    return destination


def resolve_abs_path(path_str: str) -> Path:
    """
    file.py -> /Users/home/username/project/file.py
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a sibling temp file so interruptions cannot leave a
    truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
