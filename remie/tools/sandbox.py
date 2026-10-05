"""Linux command sandbox. This does not sandbox Python tools or the agent."""

import os
import platform
from pathlib import Path
import shutil
import sys

from remie.tools.common import _project_root, tool_working_directory


class SandboxError(RuntimeError):
    pass


_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "_vendor" / "bubblewrap"


def find_bubblewrap() -> str | None:
    """Prefer the packaged helper; never bypass a broken bundled helper."""
    architecture = {"amd64": "x86_64", "x86_64": "x86_64",
                    "arm64": "aarch64", "aarch64": "aarch64"}.get(platform.machine().lower())
    if architecture:
        bundled = _VENDOR_ROOT / f"linux-{architecture}" / "bwrap"
        if os.path.lexists(bundled):
            if not bundled.is_file() or not os.access(bundled, os.X_OK):
                raise SandboxError(f"Bundled Bubblewrap is not executable: {bundled}")
            return str(bundled)
    return shutil.which("bwrap")


def command_launch(command: str, cwd: Path) -> tuple[str | list[str], dict[str, str] | None]:
    """Return a launch specification; never fall back after sandbox failure."""
    mode = os.environ.get("REMIE_SANDBOX", "on").lower()
    if mode not in {"on", "off"}:
        raise SandboxError("REMIE_SANDBOX must be 'on' or 'off'")
    if sys.platform != "linux" or mode == "off":
        return command, None
    network = os.environ.get("REMIE_SANDBOX_NETWORK", "off").lower()
    if network not in {"on", "off"}:
        raise SandboxError("REMIE_SANDBOX_NETWORK must be 'on' or 'off'")
    executable = find_bubblewrap()
    if not executable:
        raise SandboxError("Linux sandbox requires bubblewrap (bwrap). Install it or explicitly set REMIE_SANDBOX=off.")
    root = _project_root(tool_working_directory.get() or Path.cwd()).resolve()
    cwd = cwd.resolve()
    if root == Path("/") or root == Path.home().resolve():
        raise SandboxError("Refusing to expose the filesystem root or entire home as a sandbox workspace.")
    if not cwd.is_relative_to(root):
        raise SandboxError("Sandbox command cwd must be inside the active project.")
    args = [executable, "--die-with-parent", "--new-session", "--unshare-all",
            "--cap-drop", "ALL"]
    if network == "on":
        args += ["--share-net"]
    # Fresh root: never bind the host / or home. Preserve merged-/usr symlinks.
    for name in ("usr", "bin", "sbin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            args += ["--symlink", os.readlink(path), str(path)]
        elif path.exists():
            args += ["--ro-bind", str(path), str(path)]
    for name in ("ld.so.cache", "resolv.conf", "hosts", "nsswitch.conf", "ssl/certs"):
        path = Path("/etc") / name
        if path.exists():
            args += ["--ro-bind", str(path.resolve()), str(path)]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--dir", "/tmp/home", "--bind", str(root), str(root)]
    gitfile = root / ".git"
    if gitfile.is_file():
        text = gitfile.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = (root / text.split(":", 1)[1].strip()).resolve()
            commonfile = gitdir / "commondir"
            common = (gitdir / commonfile.read_text().strip()).resolve() if commonfile.exists() else gitdir
            if not common.is_relative_to(root):
                args += ["--ro-bind", str(common), str(common)]
            if not gitdir.is_relative_to(common) and not gitdir.is_relative_to(root):
                args += ["--ro-bind", str(gitdir), str(gitdir)]
    # Allowlist rather than filtering known API-key names.
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/home",
           "TMPDIR": "/tmp", "LANG": os.environ.get("LANG", "C.UTF-8")}
    args += ["--chdir", str(cwd), "--", "/bin/sh", "-c", command]
    return args, env
