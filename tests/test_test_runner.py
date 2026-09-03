from remie.tools import test_runner


def test_default_discovery_covers_additional_ecosystems(tmp_path):
    files = [
        "pkg/widget_test.go",
        "src/test/java/WidgetTest.java",
        "spec/widget_spec.rb",
        "test/widget_test.exs",
        "Tests/AppTests.swift",
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    assert test_runner._discover(tmp_path, None) == sorted(files)


def test_worker_commands_use_native_package_syntax():
    assert test_runner._worker_command(
        "go test ./...", ["example/a", "example/b"], "go"
    ) == "go test example/a example/b"
    assert test_runner._worker_command(
        "cargo test --workspace", ["core", "cli"], "cargo"
    ) == "cargo test -p core -p cli"
    assert test_runner._worker_command(
        "./gradlew test", ["api", "tools/cli"], "gradle"
    ) == "./gradlew :api:test :tools:cli:test"


def test_no_discovery_falls_back_to_original_command(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, cwd, timeout):
        calls.append((command, cwd, timeout))
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(test_runner, "run_command_tool", fake_run)
    result = test_runner.run_test_shards_tool(
        command="custom-test", cwd=str(tmp_path), worker_timeout_seconds=17
    )
    assert result["status"] == "passed"
    assert result["mode"] == "fallback"
    assert calls == [("custom-test", str(tmp_path), 17)]


def test_explicit_targets_take_precedence(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, cwd, timeout):
        calls.append(command)
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(test_runner, "run_command_tool", fake_run)
    result = test_runner.run_test_shards_tool(
        command="custom-test",
        cwd=str(tmp_path),
        targets=["suite-a", "suite-b", "suite-c", "suite-d"],
        estimated_seconds=500,
    )
    assert result["mode"] == "parallel"
    assert result["tests"] == 4
    assert sorted(calls) == [
        "custom-test suite-a",
        "custom-test suite-b",
        "custom-test suite-c",
        "custom-test suite-d",
    ]
