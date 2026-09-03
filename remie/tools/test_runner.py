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
    # Python / JavaScript / TypeScript
    "tests/test_*.py", "tests/**/*_test.py", "**/*.test.js", "**/*.test.ts",
    "**/*.spec.js", "**/*.spec.ts", "**/*.test.jsx", "**/*.test.tsx",
    "**/*.spec.jsx", "**/*.spec.tsx",
    # JVM, native, scripting, and mobile ecosystems
    "**/src/test/**/*.java", "**/src/test/**/*.kt", "**/*_test.go",
    "**/test_*.c", "**/*_test.c", "**/test_*.cc", "**/*_test.cc",
    "**/test_*.cpp", "**/*_test.cpp", "**/*_spec.rb", "**/test_*.rb",
    "**/*Test.php", "**/*_test.exs", "**/*Tests.cs", "Tests/**/*.swift",
    "test/**/*_test.dart", "test/**/*.bats",
)
_IGNORED_PARTS = {".git", "node_modules", ".venv", "venv", "target", "build", "dist"}
_DURATION = re.compile(r"^([0-9.]+)s\s+call\s+(.+)$", re.MULTILINE)


def _discover(root: Path, patterns: list[str] | None) -> list[str]:
    found: set[str] = set()
    for pattern in patterns or _DEFAULT_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and not any(part in _IGNORED_PARTS for part in path.parts):
                found.add(path.relative_to(root).as_posix())
    return sorted(found)


def _ecosystem(root: Path, command: str) -> str:
    """Identify runners whose safe shard unit is not an individual source file."""
    executable = shlex.split(command)[0] if command.strip() else ""
    if executable == "go" or (root / "go.mod").is_file():
        return "go"
    if executable == "cargo" or (root / "Cargo.toml").is_file():
        return "cargo"
    if executable in {"dotnet"} or list(root.glob("*.sln")):
        return "dotnet"
    if executable in {"mvn", "mvnw", "./mvnw"} or (root / "pom.xml").is_file():
        return "maven"
    if "gradle" in executable or (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
        return "gradle"
    return "files"


def _native_targets(root: Path, command: str, ecosystem: str) -> list[str]:
    """Discover package/module targets for runners that do their own collection."""
    if ecosystem == "go":
        result = run_command_tool("go list ./...", str(root), 120)
        return str(result.get("stdout", "")).split() if result.get("exit_code") == 0 else []
    if ecosystem == "cargo":
        result = run_command_tool("cargo metadata --no-deps --format-version 1", str(root), 120)
        try:
            data = json.loads(str(result.get("stdout", "")))
            members = set(data.get("workspace_members", []))
            return sorted(
                str(package["name"]) for package in data.get("packages", [])
                if package.get("id") in members and package.get("name")
            )
        except (TypeError, ValueError, KeyError):
            return []
    if ecosystem == "dotnet":
        return sorted(
            path.relative_to(root).as_posix() for path in root.glob("**/*[Tt]est*.csproj")
            if not any(part in _IGNORED_PARTS for part in path.parts)
        )
    if ecosystem in {"maven", "gradle"}:
        marker = "pom.xml" if ecosystem == "maven" else "build.gradle*"
        modules = {
            path.parent.relative_to(root).as_posix() or "."
            for path in root.glob(f"**/{marker}")
            if (path.parent / "src" / "test").is_dir()
            and not any(part in _IGNORED_PARTS for part in path.parts)
        }
        return sorted(modules)
    return []


def _worker_command(command: str, assigned: list[str], ecosystem: str) -> str:
    quoted = [shlex.quote(target) for target in assigned]
    parts = shlex.split(command)
    if ecosystem == "go":
        base = shlex.join([part for part in parts if part != "./..."])
        return base + " " + " ".join(quoted)
    if ecosystem == "cargo":
        base = shlex.join([part for part in parts if part not in {"--workspace", "--all"}])
        return base + " " + " ".join(f"-p {target}" for target in quoted)
    if ecosystem == "dotnet":
        # dotnet test accepts one project, so one shell worker runs its balanced
        # assignment serially while the four workers remain parallel.
        return " && ".join(f"{command} {target}" for target in quoted)
    if ecosystem == "maven":
        modules = ",".join(assigned)
        return f"{command} -pl {shlex.quote(modules)}"
    if ecosystem == "gradle":
        base = shlex.join([part for part in parts if part not in {"test", "check"}])
        tasks = " ".join(
            "test" if target == "." else shlex.quote(":" + target.replace("/", ":") + ":test")
            for target in assigned
        )
        return f"{base} {tasks}"
    return command + " " + " ".join(quoted)


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
    targets: list[str] | None = None,
    worker_timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Discover tests across common ecosystems and run long suites in four balanced workers.

    Explicit ``targets`` take precedence and may be package, module, project, or
    file identifiers appropriate for the supplied command. If safe shard units
    cannot be discovered, the original command is run once instead of reporting
    a misleading "no tests" result.
    """
    root = resolve_abs_path(cwd)
    ecosystem = _ecosystem(root, command)
    files = _discover(root, patterns)
    discovered = (
        list(targets) if targets is not None
        else _native_targets(root, command, ecosystem) if ecosystem != "files"
        else files
    )
    if not discovered:
        started = time.monotonic()
        result = run_command_tool(command, str(root), worker_timeout_seconds)
        return {
            "status": "passed" if result.get("exit_code") == 0 else "failed",
            "mode": "fallback", "ecosystem": ecosystem,
            "reason": "No safe shard targets were discovered; ran the command once.",
            "tests": 0, "duration_seconds": round(time.monotonic() - started, 3),
            "result": result,
        }

    timings = _load_timings(root)
    cases = _collect_pytest(command, root, discovered) if ecosystem == "files" else []
    tests = cases or discovered
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
        worker_command = _worker_command(command, assigned, ecosystem)
        duration_flags = " --durations=0 --durations-min=0" if cases else ""
        result = run_command_tool(
            f"{worker_command}{duration_flags}", str(root), worker_timeout_seconds
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
        "workers": 4,
        "granularity": "test_case" if cases else ("file" if ecosystem == "files" else "package"),
        "ecosystem": ecosystem,
        "estimated_seconds": estimate, "threshold_seconds": threshold_seconds,
        "tests": len(tests), "duration_seconds": round(time.monotonic() - started, 3),
        "predicted_shard_seconds": durations, "shards": results,
    }
