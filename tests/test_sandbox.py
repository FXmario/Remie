import shutil
import sys

import pytest

from remie.tools.common import tool_working_directory
from remie.tools.commands import run_command_tool
from remie.tools.sandbox import command_launch, SandboxError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("REMIE_SANDBOX", "on")
    monkeypatch.setenv("REMIE_SANDBOX_NETWORK", "off")
    token = tool_working_directory.set(tmp_path)
    yield tmp_path
    tool_working_directory.reset(token)


def test_policy(workspace, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/bwrap")
    args, env = command_launch("echo hi", workspace)
    assert "--unshare-all" in args
    assert "--share-net" not in args
    assert args[-3:] == ["/bin/sh", "-c", "echo hi"]
    assert env["HOME"] == "/tmp/home"
    monkeypatch.setenv("SECRET_API_KEY", "secret")
    assert "SECRET_API_KEY" not in command_launch("true", workspace)[1]
    monkeypatch.setenv("REMIE_SANDBOX_NETWORK", "on")
    assert "--share-net" in command_launch("true", workspace)[0]


def test_missing_and_outside(workspace, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = run_command_tool("true", str(workspace))
    assert result["exit_code"] is None
    assert "requires bubblewrap" in result["stderr"]
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/bwrap")
    with pytest.raises(SandboxError):
        command_launch("true", workspace.parent)


def test_opt_out(workspace, monkeypatch):
    monkeypatch.setenv("REMIE_SANDBOX", "off")
    assert command_launch("true", workspace) == ("true", None)


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bwrap"), reason="needs Linux bwrap")
def test_real_isolation(workspace, monkeypatch):
    probe = run_command_tool("true", str(workspace))
    if probe["exit_code"] != 0:
        pytest.skip(f"bwrap unavailable in this environment: {probe['stderr']}")
    outside = workspace.parent / "secret"
    outside.write_text("secret")
    (workspace / "escape").symlink_to(outside)
    monkeypatch.setenv("SECRET_API_KEY", "secret")
    result = run_command_tool(
        'echo ok > allowed; test ! -r escape && test -z "$SECRET_API_KEY" && test "$HOME" = /tmp/home',
        str(workspace),
    )
    assert result["exit_code"] == 0, result
    assert (workspace / "allowed").read_text() == "ok\n"
    assert run_command_tool("sleep 10", str(workspace), 1)["timed_out"]
