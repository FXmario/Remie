"""Release smoke test: run outside the checkout against an installed wheel."""
import os
from pathlib import Path
import subprocess
import tempfile

from remie.tools.common import tool_working_directory
from remie.tools.sandbox import command_launch, find_bubblewrap

helper = find_bubblewrap()
assert helper and "_vendor" in helper, helper
subprocess.run([helper, "--version"], check=True)
os.environ["REMIE_SANDBOX"] = "on"
os.environ["REMIE_SANDBOX_NETWORK"] = "off"
os.environ["SECRET_API_KEY"] = "must-not-leak"
with tempfile.TemporaryDirectory() as directory:
    parent = Path(directory)
    root = parent / "project"
    root.mkdir()
    secret = parent / "outside-secret"
    secret.write_text("private")
    (root / "escape").symlink_to(secret)
    token = tool_working_directory.set(root)
    try:
        args, env = command_launch(
            'echo ok > allowed; test ! -r escape && '
            'test -z "$SECRET_API_KEY" && test "$HOME" = /tmp/home',
            root,
        )
        assert "--unshare-all" in args and "--share-net" not in args
        subprocess.run(args, env=env, check=True, timeout=30)
        assert (root / "allowed").read_text() == "ok\n"
    finally:
        tool_working_directory.reset(token)
