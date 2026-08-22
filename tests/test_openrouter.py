import asyncio
import json

import httpx
import pytest

from remie import openrouter_client
from remie.agent import LLMRequestError


def sse_body(chunks, done=True):
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def content_chunk(text):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def reasoning_chunk(text, key="reasoning"):
    return {
        "choices": [
            {"index": 0, "delta": {key: text}, "finish_reason": None}
        ]
    }


def tool_chunk(index, *, call_id=None, name=None, arguments=None):
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    fragment = {"index": index, "function": function}
    if call_id is not None:
        fragment["id"] = call_id
        fragment["type"] = "function"
    return {
        "choices": [
            {"index": 0, "delta": {"tool_calls": [fragment]}, "finish_reason": None}
        ]
    }


def finish_chunk(reason):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def install_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        openrouter_client,
        "_CLIENT_FACTORY",
        lambda: httpx.AsyncClient(
            transport=transport, timeout=openrouter_client.HTTP_TIMEOUT
        ),
    )


def collect(conversation=None, *, api_key="sk-or-test", monkeypatch=None, client=None, **kwargs):
    async def run():
        return [
            chunk
            async for chunk in openrouter_client.stream_openrouter_call(
                api_key,
                conversation if conversation is not None else [],
                model="anthropic/claude-sonnet-4.6",
                **kwargs,
            )
        ]

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# Conversation translation
# ---------------------------------------------------------------------------


def test_conversation_to_messages_maps_all_roles():
    conversation = [
        {"role": "system", "content": "You are Remie."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "list_files", "arguments": '{"path": "."}'}],
        },
        {
            "role": "tool",
            "content": '{"files": []}',
            "tool_call_id": "c1",
            "name": "list_files",
        },
    ]
    messages = openrouter_client.conversation_to_messages(conversation)

    assert messages[0] == {"role": "system", "content": "You are Remie."}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi there"}
    assert messages[3] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "list_files",
                    "arguments": '{"path": "."}',
                },
            }
        ],
    }
    assert messages[4] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"files": []}',
    }


def test_conversation_to_messages_multimodal_user_parts():
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,xx"},
                },
            ],
        }
    ]
    messages = openrouter_client.conversation_to_messages(conversation)
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,xx"},
                },
            ],
        }
    ]


def test_conversation_to_images_only_user_message():
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,yy"},
                }
            ],
        }
    ]
    messages = openrouter_client.conversation_to_messages(conversation)
    assert messages[0]["content"][0]["type"] == "image_url"


def test_empty_assistant_and_user_parts_are_dropped():
    conversation = [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": []},
        {"role": "user", "content": "still here"},
    ]
    messages = openrouter_client.conversation_to_messages(conversation)
    assert messages == [{"role": "user", "content": "still here"}]


def test_chat_tool_schemas_nests_function_fields():
    flat = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    schemas = openrouter_client.chat_tool_schemas(flat)
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_build_request_payload_basics():
    payload = openrouter_client.build_request_payload(
        [{"role": "user", "content": "hi"}], "openai/gpt-5.6"
    )
    assert payload["model"] == "openai/gpt-5.6"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload
    assert "max_tokens" not in payload


def test_build_request_payload_reasoning_effort_mapping():
    off = openrouter_client.build_request_payload([], "m", "off")
    assert "reasoning" not in off

    low = openrouter_client.build_request_payload([], "m", "low")
    assert low["reasoning"] == {"effort": "low"}

    maximum = openrouter_client.build_request_payload([], "m", "max")
    # OpenRouter models do not uniformly accept xhigh.
    assert maximum["reasoning"] == {"effort": "high"}


def test_build_request_payload_tools_and_limits():
    flat_schemas = [
        {
            "type": "function",
            "name": "edit_file",
            "description": "d",
            "parameters": {"type": "object"},
        }
    ]
    payload = openrouter_client.build_request_payload(
        [],
        "m",
        "medium",
        tools=flat_schemas,
        max_tokens=4096,
    )
    assert payload["tools"] == openrouter_client.chat_tool_schemas(flat_schemas)
    assert payload["tools"][0]["function"]["name"] == "edit_file"
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False
    assert payload["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


def test_stream_yields_text_and_captures_telemetry(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-or-test"
        assert request.url == (
            openrouter_client.OPENROUTER_BASE_URL + "/chat/completions"
        )
        return httpx.Response(
            200,
            content=sse_body(
                [
                    reasoning_chunk("thinking...", key="reasoning"),
                    content_chunk("Hel"),
                    content_chunk("lo"),
                    finish_chunk("stop"),
                    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(monkeypatch, handler)
    usage_box = {}
    reasoning_box = []
    finish_box = {}
    chunks = collect(
        [{"role": "user", "content": "hi"}],
        reasoning_effort="low",
        usage_box=usage_box,
        reasoning_box=reasoning_box,
        finish_box=finish_box,
    )
    assert chunks == ["Hel", "lo"]
    assert reasoning_box == ["thinking..."]
    assert usage_box == {"prompt_tokens": 12, "completion_tokens": 4}
    assert finish_box["finish_reason"] == "stop"
    assert finish_box["truncated"] is False
    assert finish_box["stream_complete"] is True


def test_stream_accepts_reasoning_content_alias(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body([reasoning_chunk("deep thought", key="reasoning_content")]),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(monkeypatch, handler)
    reasoning_box = []
    collect([], reasoning_box=reasoning_box)
    assert reasoning_box == ["deep thought"]


def test_tool_call_fragments_assemble_across_chunks(monkeypatch):
    captured_payloads = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            content=sse_body(
                [
                    tool_chunk(0, call_id="call_a", name="read_file"),
                    tool_chunk(0, arguments='{"file'),
                    tool_chunk(0, arguments='name": "x.py"}'),
                    finish_chunk("tool_calls"),
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    flat_schemas = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object"},
        }
    ]
    install_transport(monkeypatch, handler)
    tool_calls_box = []
    finish_box = {}
    chunks = collect(
        tools=flat_schemas,
        finish_box=finish_box,
        tool_calls_box=tool_calls_box,
    )
    assert chunks == []  # no prose, just the call
    assert tool_calls_box == [
        {
            "id": "call_a",
            "name": "read_file",
            "arguments": '{"filename": "x.py"}',
        }
    ]
    assert finish_box["finish_reason"] == "tool_calls"
    # Tools travel nested in chat-completions form.
    body = captured_payloads["body"]
    sent_tools = body["tools"]
    assert sent_tools[0]["function"]["name"] == "read_file"
    assert all(tool["type"] == "function" for tool in sent_tools)
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is False


def test_parallel_tool_calls_assemble_by_index(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(
                [
                    tool_chunk(0, call_id="call_1", name="read_file"),
                    tool_chunk(1, call_id="call_2", name="list_files"),
                    tool_chunk(0, arguments='{"filename": "a"}'),
                    tool_chunk(1, arguments='{"path": "."}'),
                    finish_chunk("tool_calls"),
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(monkeypatch, handler)
    tool_calls_box = []
    collect(tool_calls_box=tool_calls_box)
    assert [call["id"] for call in tool_calls_box] == ["call_1", "call_2"]
    assert tool_calls_box[0]["arguments"] == '{"filename": "a"}'
    assert tool_calls_box[1]["arguments"] == '{"path": "."}'


def test_truncated_finish_marks_truncation(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body([finish_chunk("length")]),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(monkeypatch, handler)
    finish_box = {}
    collect(finish_box=finish_box)
    assert finish_box["finish_reason"] == "length"
    assert finish_box["truncated"] is True


@pytest.mark.parametrize(
    ("status", "match"),
    [
        (401, "API key"),
        (402, "credits"),
        (429, "rate limit"),
        (500, "OpenRouter request failed (500)"),
    ],
)
def test_http_errors_surface_friendly_messages(monkeypatch, status, match):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": f"backend says {status}"}},
        )

    install_transport(monkeypatch, handler)
    with pytest.raises(LLMRequestError) as excinfo:
        collect()
    assert excinfo.value.status_code == status
    assert match in excinfo.value.message


def test_mid_stream_error_chunk_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(
                [
                    content_chunk("partial"),
                    {"error": {"message": "provider exploded"}},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(monkeypatch, handler)
    with pytest.raises(LLMRequestError, match="provider exploded"):
        collect()


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


def test_fetch_openrouter_models_parses_catalog(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == openrouter_client.OPENROUTER_MODELS_URL
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/claude-sonnet-4.6", "context_length": 200000},
                    {"id": "openai/gpt-5.6", "context_length": 400000},
                    {"id": "no-context-model"},
                    "not-a-dict",
                ]
            },
        )

    install_transport(monkeypatch, handler)
    rows = asyncio.run(openrouter_client.fetch_openrouter_models())
    assert rows == [
        ("anthropic/claude-sonnet-4.6", 200000),
        ("openai/gpt-5.6", 400000),
        ("no-context-model", 0),
    ]


def test_fetch_openrouter_models_returns_empty_on_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    install_transport(monkeypatch, handler)
    assert asyncio.run(openrouter_client.fetch_openrouter_models()) == []


# ---------------------------------------------------------------------------
# Agent + TUI integration
# ---------------------------------------------------------------------------


def test_agent_routes_openrouter_with_native_tools(monkeypatch):
    import remie.agent as agent

    previous = agent.get_config()
    calls = {}

    async def fake_stream(api_key, conversation, model, reasoning_effort="off", **kw):
        calls["api_key"] = api_key
        calls["model"] = model
        calls["effort"] = reasoning_effort
        calls["tools"] = kw.get("tools")
        calls["max_tokens"] = kw.get("max_tokens")
        box = kw.get("tool_calls_box")
        if box is not None:
            box.append({"id": "c1", "name": "read_file", "arguments": "{}"})
        yield "or-reply"

    monkeypatch.setattr(openrouter_client, "stream_openrouter_call", fake_stream)
    try:
        agent.configure_openai(
            agent.OPENROUTER_BASE_URL,
            "sk-or-key",
            "anthropic/claude-sonnet-4.6",
            provider="openrouter",
            reasoning_effort="medium",
        )
        assert agent._provider_defaults("openrouter").base_url == (
            agent.OPENROUTER_BASE_URL
        )
        assert agent.get_max_output_tokens("openrouter") == 32_768

        async def run():
            box = []
            chunks = [
                chunk
                async for chunk in agent.stream_llm_call([], tool_calls_box=box)
            ]
            return chunks, box

        chunks, box = asyncio.run(run())
        assert chunks == ["or-reply"]
        assert calls["api_key"] == "sk-or-key"
        assert calls["model"] == "anthropic/claude-sonnet-4.6"
        assert calls["effort"] == "medium"
        assert calls["max_tokens"] == 32_768
        names = [tool["name"] for tool in calls["tools"]]
        assert "read_file" in names and "memory" in names
        assert box == [{"id": "c1", "name": "read_file", "arguments": "{}"}]

        # Live context windows feed compaction.
        agent._openrouter_model_context.clear()
        rows = [("big/model", 262144)]
        for model_id, context_length in rows:
            agent._openrouter_model_context[model_id] = context_length
        assert agent.get_model_context_limit("big/model", "openrouter") == 262144
        assert (
            agent.get_model_context_limit("unknown/model", "openrouter")
            == agent.OPENROUTER_DEFAULT_CONTEXT_LIMIT
        )
    finally:
        agent.configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def test_openrouter_is_native_tool_calling_provider(tmp_path, monkeypatch):
    """OpenRouter uses the same unified native-call loop as Codex."""
    import remie.agent as agent
    from remie.agent import configure_openai

    previous = get_config_safe()
    try:
        configure_openai(
            agent.OPENROUTER_BASE_URL, "sk-or-k", "openai/gpt-5.6", provider="openrouter"
        )
        from remie.tui import AgentApp

        app = AgentApp()
        assert app._native_tool_calling() is True
        app._refresh_system_prompt()
        prompt = app.conversation[0]["content"]
        assert "tool: TOOL_NAME" not in prompt
        assert "function tools provided with each request" in prompt
    finally:
        configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


def get_config_safe():
    from remie.agent import get_config

    return get_config()
