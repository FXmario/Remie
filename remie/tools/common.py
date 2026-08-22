"""Shared helpers for Remie's tool implementations."""

import json
import os
import subprocess  # noqa: F401 -- re-exported; tests patch remie.tools.subprocess.run
from pathlib import Path
from typing import Any


def _remie_dir() -> Path:
    """Directory holding per-project memory and session data.

    Resolved through the package namespace at call time so tests that
    monkeypatch ``remie.tools._remie_dir`` take effect for all submodules.
    """
    import sys

    package = sys.modules.get("remie.tools")
    if package is not None:
        override = package.__dict__.get("_remie_dir")
        if override is not None and override is not _remie_dir:
            return override()
    return Path.cwd() / ".remie"


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
