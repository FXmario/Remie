"""Shell command execution with destructive-command blocking."""

import os
import re
import subprocess
from typing import Any

import remie.tools as _tools_pkg
from remie.tools.common import resolve_abs_path

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


#: Maximum seconds a shell command may run before it is killed. Configurable via
#: REMIE_COMMAND_TIMEOUT (e.g. long-running test suites).
RUN_COMMAND_TIMEOUT = _env_int("REMIE_COMMAND_TIMEOUT", 180)
RUN_COMMAND_MAX_OUTPUT = 30_000
TIMED_OUT_EXIT_CODE = 124


def _truncate_stream(stream: str, limit: int) -> str:
    """Truncate a stream to limit chars, keeping a marker on the final line."""
    if len(stream) <= limit:
        return stream
    return stream[:limit].rstrip("\n") + "\n[output truncated]\n"


# --- Command safety ---------------------------------------------------------
#
# run_command_tool refuses to execute commands that match any of the patterns
# below. Each entry is a (regex, human-readable reason) pair. The regex is
# matched (case-insensitively) against the whole command string, so commands
# chained with ;, &&, or | are checked too.

DANGEROUS_COMMAND_PATTERNS: list[tuple[str, str]] = [
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?rm\s+(?:-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(?:--\s+)?"
        r"(?:.*\s)?(?:/|/\*|~|~/|~/\*|\.\.|\.|\*)(?:\s|$)",
        "recursive forced delete of the filesystem root, home, current, or parent directory",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?(?:mkfs\S*|shred)\b",
        "disk formatting or permanent file shredding",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?(?:fdisk|parted|gdisk|sfdisk)\s+(?!-l\b|-h\b|--help\b|--list\b|--version\b)\S+",
        "disk partitioning",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?(?:shutdown|reboot|poweroff|halt)\b",
        "system shutdown or reboot",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?chmod\s+-[a-zA-Z]*r[a-zA-Z]*\s+[0-7]{3,4}\s+(?:/|~)(?:\s|$)",
        "recursive chmod on the filesystem root or home directory",
    ),
    (
        r"(^|[;&|]\s*)(?:sudo\s+)?chown\s+-[a-zA-Z]*r[a-zA-Z]*\s+\S+\s+(?:/|~)(?:\s|$)",
        "recursive chown on the filesystem root or home directory",
    ),
    (
        r"\(\)\s*\{\s*:\s*\|",
        "fork bomb",
    ),
    (
        r"(^|[;&|]\s*)(?:curl|wget)\s+[^\n|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh)\b",
        "piping a downloaded script directly into a shell",
    ),
]

#: Writing bytes to these devices is harmless, so `dd of=/dev/null` etc. stays
#: allowed even though `dd of=/dev/sda` is blocked.
_DD_SAFE_DEVICES = {"null", "zero", "random", "urandom", "stdin", "stdout"}


def get_custom_blocked_commands() -> list[str]:
    """
    Extra blocked substrings from the REMIE_BLOCKED_COMMANDS environment
    variable (comma-separated, case-insensitive). Example:
    REMIE_BLOCKED_COMMANDS="git push --force,aws s3 rm"
    """
    return [
        item.strip().lower()
        for item in os.environ.get("REMIE_BLOCKED_COMMANDS", "").split(",")
        if item.strip()
    ]


def get_blocked_command_reason(command: str) -> str | None:
    """
    Return why `command` is blocked, or None if it may be executed.

    Blocks command strings that match a destructive pattern (recursive
    deletion of / or ~, disk formatting/partitioning, shutdown, fork bombs,
    piping a downloaded script into a shell, ...) as well as any custom
    substrings listed in REMIE_BLOCKED_COMMANDS.
    """
    low = command.lower().strip()
    for pattern, reason in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, low):
            return reason
    # `dd` is only blocked when it writes to a raw block device such as
    # /dev/sda; safe devices like /dev/null stay allowed.
    if re.search(r"(^|[;&|]\s*)(?:sudo\s+)?dd\b", low):
        match = re.search(r"\bof\s*=\s*([^\s;&|]+)", low)
        if match:
            target = match.group(1)
            if target.startswith("/dev/"):
                device = target[len("/dev/") :].split("/", 1)[0]
                if device not in _DD_SAFE_DEVICES:
                    return "dd writing directly to a raw block device"
    for blocked in get_custom_blocked_commands():
        if blocked in low:
            return f"matches blocked pattern '{blocked}'"
    return None


def run_command_tool(command: str, cwd: str = ".") -> dict[str, Any]:
    """
    Runs a shell command in the project and returns its exit code and output.
    Destructive commands (rm -rf on / or ~, disk formatting, shutdown, fork
    bombs, curl|sh, ...) are blocked before execution.
    :param command: The shell command to run.
    :param cwd: The directory to run the command in (defaults to the project).
    :return: A dictionary with the exit code, stdout, stderr, cwd, and whether it timed out.
    """
    full_path = resolve_abs_path(cwd)
    reason = get_blocked_command_reason(command)
    if reason is not None:
        return {
            "command": command,
            "cwd": str(full_path),
            "blocked": True,
            "reason": reason,
            "exit_code": None,
            "stdout": "",
            "stderr": f"Command blocked: {reason}",
            "timed_out": False,
            "truncated": False,
        }
    try:
        result = subprocess.run(
            command,
            cwd=str(full_path),
            shell=True,
            capture_output=True,
            text=True,
            timeout=_tools_pkg.RUN_COMMAND_TIMEOUT,
            input=None,
        )
        exit_code = result.returncode
        stdout, stderr = result.stdout, result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        exit_code = TIMED_OUT_EXIT_CODE
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        timed_out = True
        hint = (
            f"\n\n[command timed out after {_tools_pkg.RUN_COMMAND_TIMEOUT}s] The command was "
            "killed before finishing. Do not retry the exact same command "
            "unchanged; instead reduce its scope (fewer files, shorter input, "
            "one step at a time) or increase REMIE_COMMAND_TIMEOUT and try again."
        )
        budget = max(RUN_COMMAND_MAX_OUTPUT - len(stdout) - len(hint) - 10, 1)
        stderr = _truncate_stream(stderr, budget).rstrip("\n") + hint

    truncated = False
    if len(stdout) + len(stderr) > RUN_COMMAND_MAX_OUTPUT:
        truncated = True
        budget = max(RUN_COMMAND_MAX_OUTPUT - 40, 1)
        if stdout and stderr:
            stdout_share = int(budget * len(stdout) / (len(stdout) + len(stderr)))
            stdout = _truncate_stream(stdout, stdout_share)
            stderr = _truncate_stream(stderr, budget - stdout_share)
        elif stdout:
            stdout = _truncate_stream(stdout, budget)
        else:
            stderr = _truncate_stream(stderr, budget)

    return {
        "command": command,
        "cwd": str(full_path),
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "truncated": truncated,
    }


