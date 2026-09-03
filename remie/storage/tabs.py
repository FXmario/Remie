"""Global, project-scoped open-tab layout persistence."""

import datetime as _dt
import json
import os
import uuid
from pathlib import Path
from typing import Any

from remie.tools.common import _write_json_atomic

TAB_LAYOUT_VERSION = 1


def tabs_path() -> Path:
    """Return the global layout file (REMIE_HOME remains test/config friendly)."""
    # Honor the package-level storage override used by embedders and tests.
    import remie.tools as tools

    override = tools.__dict__.get("_remie_dir")
    from remie.tools.common import _remie_dir as default_remie_dir

    if override is not None and override is not default_remie_dir:
        return override() / "tabs.json"
    return Path(os.environ.get("REMIE_HOME", "~/.remie")).expanduser() / "tabs.json"


def project_key() -> str:
    return str(Path.cwd().resolve())


def _empty_document() -> dict[str, Any]:
    return {"version": TAB_LAYOUT_VERSION, "projects": {}}


def load_tab_document() -> dict[str, Any]:
    path = tabs_path()
    if not path.is_file():
        return _empty_document()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty_document()
    if not isinstance(data, dict) or not isinstance(data.get("projects"), dict):
        return _empty_document()
    return data


def load_tab_layout() -> dict[str, Any]:
    """Load this project's layout, returning a normalized empty layout."""
    layout = load_tab_document()["projects"].get(project_key(), {})
    if not isinstance(layout, dict):
        layout = {}
    tabs = layout.get("tabs", [])
    if not isinstance(tabs, list):
        tabs = []
    valid = [
        tab for tab in tabs
        if isinstance(tab, dict)
        and isinstance(tab.get("id"), str)
        and isinstance(tab.get("chat_id"), str)
    ]
    return {
        "active_tab_id": layout.get("active_tab_id"),
        "sidebar_visible": layout.get("sidebar_visible", True) is not False,
        "tabs": valid,
    }


def save_tab_layout(layout: dict[str, Any]) -> None:
    document = load_tab_document()
    document["version"] = TAB_LAYOUT_VERSION
    document["projects"][project_key()] = layout
    _write_json_atomic(tabs_path(), document)


def new_tab(chat_id: str) -> dict[str, str]:
    return {
        "id": str(uuid.uuid4()),
        "chat_id": chat_id,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
