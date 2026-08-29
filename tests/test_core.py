import asyncio

from remie.config import ConnectionConfig
from remie.core.runner import AgentRunner
from remie.core.events import ToolCompleted, TurnCompleted
from remie.providers.events import (
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from remie.providers.router import RoutedProvider
from remie.tools.executor import ToolExecutor


async def _no_answer(_question, _options):
    return None


def _runner(run=lambda _name, _args: {}):
    return AgentRunner(ToolExecutor(_no_answer, run=run))


def test_runner_parses_text_protocol_calls():
    prepared = _runner().prepare_response(
        'thinking: inspect\ntool: read_file({"filename":"main.py"})',
        native_tool_calling=False,
    )
    assert prepared.content == ""
    assert prepared.tool_invocations == [("read_file", {"filename": "main.py"})]


def test_runner_preserves_native_argument_bytes_and_reasoning():
    runner = _runner()
    prepared = runner.prepare_response(
        "",
        native_tool_calling=True,
        native_calls=[
            {
                "id": "call-1",
                "name": "list_files",
                "arguments": '{ "path" : "." }',
            }
        ],
    )
    metadata = runner.assistant_metadata(
        prepared, [{"id": "reason-1", "encrypted_content": "ciphertext"}]
    )
    assert metadata["tool_calls"][0]["arguments"] == '{ "path" : "." }'
    assert metadata["codex_reasoning"][0]["id"] == "reason-1"


def test_runner_repairs_only_unanswered_tool_calls():
    conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "done", "name": "read_file"},
                {"id": "missing", "name": "list_files"},
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "done"},
    ]
    repaired, changed = AgentRunner.close_dangling_tool_calls(conversation)
    assert changed is True
    outputs = [m for m in repaired if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in outputs] == ["missing", "done"]


def test_tool_executor_requires_permission_outside_project(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside" / "secret.txt"
    project.mkdir()
    outside.parent.mkdir()
    outside.write_text("secret", encoding="utf-8")
    prompts = []

    async def deny(question, options):
        prompts.append((question, options))
        return "Deny"

    executor = ToolExecutor(deny, project_root=project)
    result = asyncio.run(executor.execute("read_file", {"filename": str(outside)}))

    assert result["error"].startswith("Permission denied")
    assert result["paths"] == [str(outside)]
    assert prompts and prompts[0][1] == ["Allow once", "Deny"]


def test_tool_executor_allows_approved_outside_access_once(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside.txt"
    project.mkdir()
    outside.write_text("allowed", encoding="utf-8")

    async def allow(_question, _options):
        return "Allow once"

    executor = ToolExecutor(allow, project_root=project)
    result = asyncio.run(executor.execute("read_file", {"filename": str(outside)}))

    assert result["content"] == "allowed"


def test_tool_executor_prompts_for_outside_path_in_shell_command(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    questions = []

    async def deny(question, _options):
        questions.append(question)
        return "Deny"

    executor = ToolExecutor(deny, project_root=project)
    result = asyncio.run(
        executor.execute(
            "run_command", {"command": "cat ../secret.txt", "cwd": str(project)}
        )
    )

    assert result["error"].startswith("Permission denied")
    assert str(tmp_path / "secret.txt") in questions[0]


def test_tool_executor_does_not_prompt_for_project_paths(tmp_path):
    project = tmp_path / "project"
    target = project / "file.txt"
    project.mkdir()
    target.write_text("inside", encoding="utf-8")

    async def unexpected_prompt(_question, _options):
        raise AssertionError("inside-project access should not prompt")

    executor = ToolExecutor(unexpected_prompt, project_root=project)
    result = asyncio.run(executor.execute("read_file", {"filename": str(target)}))

    assert result["content"] == "inside"


def test_tool_executor_uses_injected_question_handler():
    async def ask(question, options):
        assert question == "Choose"
        assert options == ["A", "B"]
        return "B"

    executor = ToolExecutor(ask)
    result = asyncio.run(
        executor.execute("ask_user", {"question": "Choose", "options": ["A", "B"]})
    )
    assert result == {"answer": "B"}


def test_routed_provider_emits_typed_events(monkeypatch):
    import remie.providers.router as router

    async def fake_stream(
        _config,
        _conversation,
        *,
        usage_box,
        reasoning_box,
        finish_box,
        tool_calls_box,
        reasoning_items_box,
        **_kwargs,
    ):
        reasoning_box.append("think")
        yield "answer"
        usage_box.update(prompt_tokens=10, completion_tokens=3)
        tool_calls_box.append({"id": "call-1", "name": "read_file", "arguments": "{}"})
        reasoning_items_box.append({"id": "r-1", "encrypted_content": "x"})
        finish_box.update(
            finish_reason="tool_calls", truncated=False, stream_complete=True
        )

    monkeypatch.setattr(router, "stream_provider_call", fake_stream)
    provider = RoutedProvider(
        ConnectionConfig("http://local", "key", "model"),
        get_http_client=lambda: None,
        get_local_openai_client=lambda: None,
        max_output_tokens=100,
        reasoning_supported=True,
        reasoning_poll_interval=0.001,
    )

    async def collect():
        return [event async for event in provider.stream([])]

    events = asyncio.run(collect())
    assert any(isinstance(event, TextDelta) for event in events)
    assert any(isinstance(event, ReasoningDelta) for event in events)
    assert any(isinstance(event, UsageEvent) for event in events)
    assert any(isinstance(event, ToolCallEvent) for event in events)
    finish = next(event for event in events if isinstance(event, FinishEvent))
    assert finish.complete is True
    assert finish.provider_metadata["reasoning_items"][0]["id"] == "r-1"


def test_headless_runner_completes_text_tool_loop():
    class ScriptedProvider:
        def __init__(self):
            self.round = 0

        async def stream(self, _conversation):
            self.round += 1
            if self.round == 1:
                yield TextDelta('tool: read_file({"filename":"main.py"})')
            else:
                yield TextDelta("Done")
            yield FinishEvent("stop")

    calls = []
    runner = AgentRunner(
        ToolExecutor(
            _no_answer,
            run=lambda name, args: calls.append((name, args)) or {"content": "x"},
        ),
        provider=ScriptedProvider(),
    )
    conversation = [{"role": "system", "content": "system"}]

    async def collect():
        return [
            event
            async for event in runner.run_turn(
                conversation, "inspect", native_tool_calling=False
            )
        ]

    events = asyncio.run(collect())
    assert calls == [("read_file", {"filename": "main.py"})]
    assert any(isinstance(event, ToolCompleted) for event in events)
    assert events[-1] == TurnCompleted("Done", "")
    assert [message["role"] for message in conversation] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_headless_runner_preserves_native_tool_pairing():
    class NativeProvider:
        def __init__(self):
            self.round = 0

        async def stream(self, _conversation):
            self.round += 1
            if self.round == 1:
                yield ToolCallEvent("call-1", "list_files", '{"path":"."}')
                yield FinishEvent(
                    "tool_calls",
                    provider_metadata={
                        "reasoning_items": [
                            {"id": "reason-1", "encrypted_content": "cipher"}
                        ]
                    },
                )
            else:
                yield TextDelta("Complete")
                yield FinishEvent("stop")

    runner = AgentRunner(
        ToolExecutor(_no_answer, run=lambda _name, _args: {"files": []}),
        provider=NativeProvider(),
    )
    conversation = [{"role": "system", "content": "system"}]

    async def collect():
        return [
            event
            async for event in runner.run_turn(
                conversation, "list", native_tool_calling=True
            )
        ]

    asyncio.run(collect())
    assistant_call = conversation[2]
    tool_result = conversation[3]
    assert assistant_call["tool_calls"][0]["id"] == "call-1"
    assert assistant_call["codex_reasoning"][0]["id"] == "reason-1"
    assert tool_result["tool_call_id"] == "call-1"
