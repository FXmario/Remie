"""Import-cost regression tests for lazy compatibility facades."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_isolated(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tui_package_does_not_eagerly_initialize_frontend():
    _run_isolated(
        "import sys, remie.tui; "
        "assert 'remie.tui.app' not in sys.modules; "
        "assert 'PIL.Image' not in sys.modules; "
        "assert 'textual.app' not in sys.modules"
    )


def test_agent_defers_openai_sdk_and_compatibility_rendering():
    _run_isolated(
        "import sys, remie.agent; "
        "assert 'openai' not in sys.modules; "
        "assert 'rich.markdown' not in sys.modules"
    )


def test_lazy_compatibility_exports_still_resolve():
    _run_isolated(
        "from remie.agent import estimate_tokens; "
        "from remie.tui import MAX_AUTO_CONTINUATIONS; "
        "assert estimate_tokens('hello') == 1; "
        "assert MAX_AUTO_CONTINUATIONS > 0"
    )


@pytest.mark.parametrize("shell_config", [None, "shell-config"])
def test_startup_ignores_dotenv_files(tmp_path, shell_config):
    (tmp_path / ".env").write_text(
        "REMIE_CONFIG_DIR=dotenv-config\nREMIE_DOTENV_TEST=loaded\n"
    )
    env = os.environ.copy()
    env.pop("REMIE_CONFIG_DIR", None)
    env.pop("REMIE_DOTENV_TEST", None)
    env["HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    expected = tmp_path / ".config" / "remie"
    if shell_config is not None:
        expected = tmp_path / shell_config
        env["REMIE_CONFIG_DIR"] = str(expected)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; import remie.agent, remie.tui; "
            "from remie.config import CONFIG_DIR; "
            "assert str(CONFIG_DIR) == sys.argv[1]; "
            "assert 'REMIE_DOTENV_TEST' not in os.environ; "
            "assert 'dotenv' not in sys.modules",
            str(expected),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
