"""Import-cost regression tests for lazy compatibility facades."""

import subprocess
import sys


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
