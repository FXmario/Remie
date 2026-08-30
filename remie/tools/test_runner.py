"""Runtime-balanced four-way test-suite orchestration."""

from __future__ import annotations

import concurrent.futures
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

from remie.tools.commands import run_command_tool
from remie.tools.common import resolve_abs_path

_DEFAULT_PATTERNS = (
    "tests/test_*.py", "tests/**/*_test.py", "**/*.test.js", "**/*.test.ts",
    "**/*.spec.js", "**/*.spec.ts",
)
_DURATION = re.compile(r"^([0-9.]+)s\s+call\s+(.+)$", re.MULTILINE)


def _discover(root: Path, patterns: list[str] | None) -> list[str]:
    found: set[str] = set()
    for pattern in patterns or _DEFAULT_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and not any(part in {".git", "node_modules", ".venv"} for part in path.parts):
                found.add(path.relative_to(root).as_posix())
    return sorted(found)


def _load_timings(root: Path) -> dict[str, float]:
    try:
        raw = json.loads((root / ".remie" / "test-timings.json").read_text())
        return {str(key): float(value) for key, value in raw.items()}
    except (OSError, ValueError, TypeError):
        return {}


def _save_timings(root: Path, timings: dict[str, float]) -> None:
    try:
        target = root / ".remie" / "test-timings.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass  # Timing history is an optimization, never a reason to fail tests.


def _collect_pytest(command: str, root: Path, files: list[str]) -> list[str]:
    if "pytest" not in command:
        return []
    # Collect per file so run_command's output cap cannot cut a node ID in half
    # on a large suite. A partial ID makes the entire pytest shard exit with 4.
    collected: set[str] = set()
    for file in files:
        result = run_command_tool(
            f"{command} --collect-only {shlex.quote(file)}", str(root), 120
        )
        if result.get("exit_code") != 0 or result.get("truncated"):
            return []
        collected.update(
            line.strip()
            for line in str(result.get("stdout", "")).splitlines()
            if "::" in line and not line.startswith((" ", "="))
        )
    return sorted(collected)


def _balanced_shards(tests: list[str], timings: dict[str, float]) -> tuple[list[list[str]], list[float]]:
    fallback = max(sum(timings.values()) / len(timings), 1.0) if timings else 1.0
    shards: list[list[str]] = [[] for _ in range(4)]
    totals = [0.0] * 4
    for test in sorted(tests, key=lambda item: timings.get(item, fallback), reverse=True):
        index = min(range(4), key=totals.__getitem__)
        shards[index].append(test)
        totals[index] += timings.get(test, fallback)
    return shards, totals


def run_test_shards_tool(
    command: str = "pytest",
    cwd: str = ".",
    threshold_seconds: int = 120,
    estimated_seconds: float | None = None,
    patterns: list[str] | None = None,
    worker_timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Discover tests, run short suites once, or run long suites in four parallel test workers."""
    root = resolve_abs_path(cwd)
    files = _discover(root, patterns)
    if not files:
        return {"status": "no_tests", "cwd": str(root), "tests": 0}

    timings = _load_timings(root)
    cases = _collect_pytest(command, root, files)
    tests = cases or files
    historical = sum(timings.get(test, 0.0) for test in tests)
    if estimated_seconds is not None:
        estimate = float(estimated_seconds)
    elif historical and all(test in timings for test in tests):
        estimate = historical
    else:
        estimate = len(tests) * (1.0 if cases else 15.0)
    parallel = estimate >= threshold_seconds and len(tests) >= 4
    started = time.monotonic()

    if not parallel:
        result = run_command_tool(command, str(root), worker_timeout_seconds)
        return {
            "status": "passed" if result.get("exit_code") == 0 else "failed",
            "mode": "local", "estimated_seconds": estimate,
            "threshold_seconds": threshold_seconds, "tests": len(tests),
            "duration_seconds": round(time.monotonic() - started, 3), "result": result,
        }

    shards, predicted = _balanced_shards(tests, timings)

    def run_worker(item: tuple[int, list[str]]) -> dict[str, Any]:
        worker_id, assigned = item
        args = " ".join(shlex.quote(test) for test in assigned)
        duration_flags = " --durations=0 --durations-min=0" if cases else ""
        result = run_command_tool(
            f"{command} {args}{duration_flags}", str(root), worker_timeout_seconds
        )
        return {"worker": worker_id, "tests": assigned,
                "predicted_seconds": round(predicted[worker_id - 1], 3), **result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_worker, item) for item in enumerate(shards, 1)]
        results = [future.result() for future in futures]

    for result in results:
        for duration, node_id in _DURATION.findall(str(result.get("stdout", ""))):
            timings[node_id.strip()] = float(duration)
    _save_timings(root, timings)

    passed = all(result.get("exit_code") == 0 for result in results)
    durations = [float(result.get("predicted_seconds", 0)) for result in results]
    return {
        "status": "passed" if passed else "failed", "mode": "parallel",
        "workers": 4, "granularity": "test_case" if cases else "file",
        "estimated_seconds": estimate, "threshold_seconds": threshold_seconds,
        "tests": len(tests), "duration_seconds": round(time.monotonic() - started, 3),
        "predicted_shard_seconds": durations, "shards": results,
    }
