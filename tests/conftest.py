import os

import pytest

os.environ.setdefault("LLAMA_BASE_URL", "http://localhost:7070/v1")


@pytest.fixture(autouse=True)
def _isolate_remie_dir(tmp_path, monkeypatch):
    """Point the agent's .remie/ memory and session paths at a per-test temp
    dir so tests don't write into (or read stale state from) the repo."""
    monkeypatch.setattr("remie.tools._remie_dir", lambda: tmp_path / ".remie")
