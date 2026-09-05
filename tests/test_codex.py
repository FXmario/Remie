import asyncio
import base64
import json
import urllib.parse

import httpx
import pytest
from types import SimpleNamespace

from remie import codex_auth, codex_client
from remie.agent import LLMRequestError


def make_jwt(claims: dict) -> str:
    def b64(value: dict) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(value).encode())
            .rstrip(b"=")
            .decode()
        )

    header = b64({"alg": "none", "typ": "JWT"})
    payload = b64(claims)
    return f"{header}.{payload}.signature"


def make_access_token(exp_offset: float = 3600) -> str:
    import time

    return make_jwt(
        {
            "exp": time.time() + exp_offset,
            codex_auth.OPENAI_AUTH_CLAIM: {
                "chatgpt_account_id": "acc_123",
                "chatgpt_plan_type": "plus",
            },
        }
    )


def make_id_token() -> str:
    return make_jwt(
        {
            "email": "dev@example.com",
            codex_auth.OPENAI_AUTH_CLAIM: {
                "chatgpt_account_id": "acc_123",
                "chatgpt_plan_type": "plus",
            },
        }
    )


@pytest.fixture
def auth_file(tmp_path, monkeypatch):
    path = tmp_path / "codex-home" / "auth.json"
    monkeypatch.setattr(codex_auth, "auth_json_path", lambda: path)
    return path


# ---------------------------------------------------------------------------
# PKCE + authorize URL
# ---------------------------------------------------------------------------


def test_generate_pkce_produces_matching_pair():
    verifier, challenge = codex_auth.generate_pkce()
    digest = base64.urlsafe_b64encode(verifier.encode()).decode().rstrip("=")
    assert len(verifier) >= 43
    assert challenge.endswith("=") is False
    # S256 challenge must be base64url(SHA256(verifier)).
    import hashlib

    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected
    assert digest != challenge  # sanity: raw b64 of verifier is not the challenge


def test_build_authorize_url_contains_required_params():
    url = codex_auth.build_authorize_url("state123", "challenge456")
    parsed = urllib.parse.urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == codex_auth.CODEX_AUTH_URL
    params = urllib.parse.parse_qs(parsed.query)
    assert params["response_type"] == ["code"]
    assert params["client_id"] == [codex_auth.CODEX_CLIENT_ID]
    assert params["redirect_uri"] == [codex_auth.CODEX_REDIRECT_URI]
    assert params["scope"] == [codex_auth.CODEX_SCOPE]
    assert params["state"] == ["state123"]
    assert params["code_challenge"] == ["challenge456"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["prompt"] == ["login"]


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def test_save_and_load_auth_roundtrip(auth_file):
    auth = codex_auth.CodexAuth(
        access_token=make_access_token(),
        refresh_token="refresh-1",
        id_token=make_id_token(),
        account_id="acc_123",
        plan_type="plus",
        email="dev@example.com",
        last_refresh=1750000000.0,
    )
    codex_auth.save_auth(auth)
    loaded = codex_auth.load_auth()

    assert loaded is not None
    assert loaded.access_token == auth.access_token
    assert loaded.refresh_token == "refresh-1"
    assert loaded.id_token == auth.id_token
    assert loaded.account_id == "acc_123"
    assert loaded.plan_type == "plus"
    assert loaded.email == "dev@example.com"
    assert loaded.last_refresh == 1750000000.0
    # Stored in the Codex CLI layout so both tools share credentials.
    stored = json.loads(auth_file.read_text())
    assert stored["OPENAI_API_KEY"] is None
    assert stored["tokens"]["refresh_token"] == "refresh-1"
    assert stored["tokens"]["account_id"] == "acc_123"
    assert "last_refresh" in stored


def test_load_accepts_flat_token_layout(auth_file):
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_file.write_text(
        json.dumps(
            {
                "access": make_access_token(),
                "refresh": "flat-refresh",
                "id_token": make_id_token(),
            }
        )
    )
    loaded = codex_auth.load_auth()
    assert loaded is not None
    assert loaded.refresh_token == "flat-refresh"
    assert loaded.account_id == "acc_123"


@pytest.mark.parametrize("content", [None, "", "{not json", '{"tokens": "nope"}'])
def test_load_returns_none_for_missing_or_invalid_files(auth_file, content):
    if content is not None:
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        auth_file.write_text(content)
    assert codex_auth.load_auth() is None


def test_clear_auth_reports_whether_anything_was_removed(auth_file):
    assert codex_auth.clear_auth() is False
    codex_auth.save_auth(
        codex_auth.CodexAuth(make_access_token(), "r", make_id_token(), "a")
    )
    assert codex_auth.clear_auth() is True
    assert not auth_file.exists()


def test_account_summary_includes_email_and_plan():
    auth = codex_auth.CodexAuth(
        access_token="",
        plan_type="chatgpt_plus",
        email="dev@example.com",
    )
    summary = codex_auth.account_summary(auth)
    assert "dev@example.com" in summary
    assert "Plus" in summary
    assert codex_auth.account_summary(
        codex_auth.CodexAuth(access_token="")
    ) == "ChatGPT account"


# ---------------------------------------------------------------------------
# Refresh / validity
# ---------------------------------------------------------------------------


async def _noop_async(value):
    return value


def test_ensure_valid_auth_raises_when_signed_out(auth_file):
    with pytest.raises(codex_auth.CodexAuthError):
        asyncio.run(codex_auth.ensure_valid_auth())


def test_ensure_valid_auth_refreshes_expired_token(auth_file, monkeypatch):
    expired = codex_auth.CodexAuth(
        make_access_token(exp_offset=-100), "stale-refresh", make_id_token()
    )
    fresh = codex_auth.CodexAuth(
        make_access_token(exp_offset=3600), "new-refresh", make_id_token()
    )
    codex_auth.save_auth(expired)
    calls = []

    async def fake_refresh(auth):
        calls.append(auth.refresh_token)
        return fresh

    monkeypatch.setattr(codex_auth, "refresh_auth", fake_refresh)
    result = asyncio.run(codex_auth.ensure_valid_auth())
    assert result is fresh
    assert calls == ["stale-refresh"]


def test_ensure_valid_auth_keeps_valid_token_when_refresh_fails(
    auth_file, monkeypatch
):
    valid = codex_auth.CodexAuth(make_access_token(exp_offset=7200), "r", make_id_token())
    codex_auth.save_auth(valid)

    async def failing_refresh(auth):
        raise codex_auth.CodexAuthError("backend down")

    monkeypatch.setattr(codex_auth, "refresh_auth", failing_refresh)
    result = asyncio.run(codex_auth.ensure_valid_auth())
    assert result.access_token == valid.access_token


def test_refresh_auth_persists_new_tokens_and_keeps_fallbacks(auth_file, monkeypatch):
    old = codex_auth.CodexAuth("old-access", "old-refresh", make_id_token())

    async def fake_request(payload):
        assert payload["grant_type"] == "refresh_token"
        assert payload["client_id"] == codex_auth.CODEX_CLIENT_ID
        assert payload["refresh_token"] == "old-refresh"
        return {"access_token": make_access_token(), "refresh_token": "new-refresh"}

    monkeypatch.setattr(codex_auth, "_token_request", fake_request)
    refreshed = asyncio.run(codex_auth.refresh_auth(old))
    assert refreshed.refresh_token == "new-refresh"
    assert refreshed.id_token == make_id_token()
    assert codex_auth.load_auth().access_token == refreshed.access_token


def test_exchange_code_persists_tokens(auth_file, monkeypatch):
    async def fake_request(payload):
        assert payload["grant_type"] == "authorization_code"
        assert payload["code_verifier"]
        return {
            "access_token": make_access_token(),
            "refresh_token": "r",
            "id_token": make_id_token(),
        }

    monkeypatch.setattr(codex_auth, "_token_request", fake_request)
    auth = asyncio.run(codex_auth.exchange_code("abc", "verifier"))
    assert auth.email == "dev@example.com"
    assert codex_auth.is_signed_in()


# ---------------------------------------------------------------------------
# Browser login flow (loopback callback)
# ---------------------------------------------------------------------------


def test_login_completes_through_loopback_callback(auth_file, monkeypatch):
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url) or True)

    async def fake_token_request(payload):
        return {
            "access_token": make_access_token(),
            "refresh_token": "r1",
            "id_token": make_id_token(),
        }

    monkeypatch.setattr(codex_auth, "_token_request", fake_token_request)

    login_urls = []

    async def exercise():
        task = asyncio.create_task(
            codex_auth.login(on_login_url=login_urls.append)
        )
        for _ in range(200):
            if login_urls:
                break
            await asyncio.sleep(0.02)
        assert login_urls, "login never surfaced the authorize URL"
        authorize_url = login_urls[0]
        assert opened_urls == [authorize_url]
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(authorize_url).query)[
            "state"
        ][0]
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:1455/auth/callback",
                params={"code": "ac_123", "state": state},
            )
        assert response.status_code == 200
        assert "Sign-in complete" in response.text
        return await asyncio.wait_for(task, 10)

    auth = asyncio.run(exercise())
    assert auth.email == "dev@example.com"
    assert auth.account_id == "acc_123"
    assert codex_auth.is_signed_in()


def test_login_rejects_state_mismatch(auth_file, monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    login_urls = []

    async def exercise():
        task = asyncio.create_task(codex_auth.login(on_login_url=login_urls.append))
        for _ in range(200):
            if login_urls:
                break
            await asyncio.sleep(0.02)
        async with httpx.AsyncClient() as client:
            await client.get(
                "http://localhost:1455/auth/callback",
                params={"code": "ac_123", "state": "evil-state"},
            )
        with pytest.raises(codex_auth.CodexAuthError, match="state"):
            await asyncio.wait_for(task, 10)

    asyncio.run(exercise())


def test_login_reports_port_conflict(auth_file, monkeypatch):
    async def busy_server():
        server = await asyncio.start_server(
            lambda r, w: None, "127.0.0.1", codex_auth.CODEX_REDIRECT_PORT
        )
        try:
            with pytest.raises(codex_auth.CodexAuthError, match="1455"):
                await codex_auth.login()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(busy_server())


# ---------------------------------------------------------------------------
# Responses conversion + payload building
# ---------------------------------------------------------------------------


def test_conversation_to_input_maps_roles_and_images():
    conversation = [
        {"role": "system", "content": "You are Remie."},
        {"role": "system", "content": "Extra instructions."},
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        },
        {"role": "assistant", "content": "sure"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]},
    ]
    instructions, items = codex_client.conversation_to_input(conversation)
    assert instructions == "You are Remie.\n\nExtra instructions."
    assert items[0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }
    assert items[1]["content"] == [
        {"type": "input_text", "text": "look at this"},
        {"type": "input_image", "image_url": "data:image/png;base64,xx"},
    ]
    assert items[2] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "sure"}],
    }
    # Empty messages (no text and no usable image) are dropped entirely.
    assert len(items) == 3


def test_build_request_payload_effort_mapping():
    base_conversation = [{"role": "user", "content": "hi"}]

    off = codex_client.build_request_payload(base_conversation, "m", "off")
    assert "reasoning" not in off
    assert off["store"] is False
    assert off["stream"] is True
    assert off["include"] == ["reasoning.encrypted_content"]

    medium = codex_client.build_request_payload(base_conversation, "m", "medium")
    assert medium["reasoning"] == {"effort": "medium", "summary": "auto"}

    maximum = codex_client.build_request_payload(base_conversation, "m", "max")
    assert maximum["reasoning"]["effort"] == "xhigh"

    empty = codex_client.build_request_payload([], "m", "low")
    assert empty["input"][0]["content"][0] == {"type": "input_text", "text": "Continue."}


# ---------------------------------------------------------------------------
# OpenAI SDK streaming backend (native function calling)
# ---------------------------------------------------------------------------


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = events
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            if isinstance(event, Exception):
                raise event
            yield event

    def close(self):
        self.closed = True


class FakeCodexSDK:
    """Scripted stand-in for AsyncOpenAI pointed at the Codex backend."""

    def __init__(self, script):
        # script: list of Exception (raise) or list-of-events (yield)
        self.script = list(script)
        self.requests: list[dict] = []
        self.streams: list[_FakeAsyncStream] = []

    @property
    def responses(self):
        return self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        stream = _FakeAsyncStream(step)
        self.streams.append(stream)
        return stream


def api_status_error(status_code: int, message: str) -> Exception:
    from openai import APIStatusError

    request = httpx.Request("POST", codex_client.CODEX_BACKEND_BASE + "/responses")
    response = httpx.Response(
        status_code, request=request, json={"error": {"message": message}}
    )
    return APIStatusError(message, response=response, body=None)


def make_event(event_type: str, **fields):
    return SimpleNamespace(type=event_type, **fields)


def completed_event(usage=(11, 7), output=None):
    return make_event(
        "response.completed",
        response=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
            output=output or [],
        ),
    )


def install_sdk(monkeypatch, client: FakeCodexSDK) -> FakeCodexSDK:
    monkeypatch.setattr(
        codex_client,
        "_SDK_FACTORY",
        lambda access_token, account_id: client,
    )
    return client


def signed_in_auth(monkeypatch, refresh_token="r1"):
    auth = codex_auth.CodexAuth(
        make_access_token(exp_offset=7200),
        refresh_token=refresh_token,
        id_token=make_id_token(),
        account_id="acc_123",
        plan_type="plus",
    )

    async def fake_ensure():
        return auth

    monkeypatch.setattr(codex_client, "ensure_valid_auth", fake_ensure)
    return auth


def collect(conversation=None, monkeypatch=None, client=None, **kwargs):
    async def run():
        return [
            chunk
            async for chunk in codex_client.stream_codex_call(
                conversation if conversation is not None else [],
                model="gpt-5.5",
                **kwargs,
            )
        ]

    return asyncio.run(run())


def test_build_request_payload_effort_mapping():
    base_conversation = [{"role": "user", "content": "hi"}]

    off = codex_client.build_request_payload(base_conversation, "m", "off")
    assert "reasoning" not in off
    assert off["store"] is False
    assert off["include"] == ["reasoning.encrypted_content"]

    medium = codex_client.build_request_payload(base_conversation, "m", "medium")
    assert medium["reasoning"] == {"effort": "medium", "summary": "auto"}

    maximum = codex_client.build_request_payload(base_conversation, "m", "max")
    assert maximum["reasoning"]["effort"] == "xhigh"

    empty = codex_client.build_request_payload([], "m", "low")
    assert empty["input"][0]["content"][0] == {"type": "input_text", "text": "Continue."}


def test_build_request_payload_includes_tools_for_native_calling():
    schemas = [
        {"type": "function", "name": "read_file", "description": "d", "parameters": {}}
    ]
    payload = codex_client.build_request_payload(
        [{"role": "user", "content": "hi"}], "m", "low", tools=schemas
    )
    assert payload["tools"] == schemas
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False

    bare = codex_client.build_request_payload([{"role": "user", "content": "hi"}], "m")
    assert "tools" not in bare and "tool_choice" not in bare


def test_conversation_to_input_maps_native_tool_messages():
    conversation = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "list_files", "arguments": '{"path": "."}'}
            ],
        },
        {
            "role": "tool",
            "content": '{"files": ["main.py"]}',
            "tool_call_id": "c1",
            "name": "list_files",
        },
        {
            "role": "assistant",
            "content": "Here is your file.",
            "tool_calls": [
                {"id": "c2", "name": "read_file", "arguments": "{}"}
            ],
        },
        {
            "role": "tool",
            "content": '{"content": "..."}',
            "tool_call_id": "c2",
            "name": "read_file",
        },
    ]
    instructions, items = codex_client.conversation_to_input(conversation)
    assert instructions == "sys"
    assert items[0]["type"] == "message" and items[0]["role"] == "user"
    assert items[1] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "list_files",
        "arguments": '{"path": "."}',
    }
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": '{"files": ["main.py"]}',
    }
    # Assistant text preceding a tool call is replayed as its own message item.
    assert items[3] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Here is your file."}],
    }
    assert items[4]["type"] == "function_call"
    assert items[5]["type"] == "function_call_output"


def test_conversation_to_input_drops_dangling_tool_calls():
    """Interrupted turns (call without result) are not replayed: the backend
    rejects an unpaired function_call with a 400."""
    conversation = [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "list_files", "arguments": '{"path": "."}'}
            ],
        },
        # No role=="tool" result for c1: the turn was interrupted here.
    ]
    instructions, items = codex_client.conversation_to_input(conversation)
    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "list the files"}],
        }
    ]


def test_conversation_to_input_drops_orphaned_tool_outputs():
    """A result whose call was dropped (e.g. by compaction) is not replayed."""
    conversation = [
        {"role": "user", "content": "go"},
        {
            "role": "tool",
            "content": '{"files": []}',
            "tool_call_id": "ghost-call",
            "name": "list_files",
        },
    ]
    _, items = codex_client.conversation_to_input(conversation)
    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "go"}],
        }
    ]


def test_conversation_to_input_replays_encrypted_reasoning_before_calls():
    """Reasoning items stored on the assistant message are replayed before
    their function_call: with store=false the backend 400s without them."""
    conversation = [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "list_files", "arguments": '{"path": "."}'}
            ],
            "codex_reasoning": [
                {"id": "rs_1", "encrypted_content": "enc-blob"},
                {"id": "rs_empty", "encrypted_content": ""},  # dropped: no blob
                "not-a-dict",
            ],
        },
        {
            "role": "tool",
            "content": "{}",
            "tool_call_id": "c1",
            "name": "list_files",
        },
    ]
    _, items = codex_client.conversation_to_input(conversation)
    assert items[0]["type"] == "message"
    assert items[1] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "enc-blob",
    }
    assert items[2] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "list_files",
        "arguments": '{"path": "."}',
    }
    assert items[3]["type"] == "function_call_output"


def test_conversation_to_input_skips_message_when_all_calls_dangling():
    """Reasoning/text of a fully-unanswered assistant message is skipped too,
    so interrupted reasoning blobs are not replayed against nothing."""
    conversation = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "partial text",
            "tool_calls": [{"id": "c1", "name": "list_files", "arguments": "{}"}],
            "codex_reasoning": [{"id": "rs_1", "encrypted_content": "enc"}],
        },
    ]
    _, items = codex_client.conversation_to_input(conversation)
    assert [item["type"] for item in items] == ["message"]


def test_stream_codex_call_parses_sdk_events_and_tool_calls(monkeypatch):
    signed_in_auth(monkeypatch)
    # The backend streams tools as item events; response.completed may carry
    # an empty output array (observed against the live API).
    client = install_sdk(
        monkeypatch,
        FakeCodexSDK(
            [
                [
                    make_event("response.created"),
                    make_event(
                        "response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            call_id="call_1",
                            name="read_file",
                            arguments="",
                        ),
                    ),
                    make_event(
                        "response.function_call_arguments.delta",
                        output_index=0,
                        delta='{"filename":',
                    ),
                    make_event(
                        "response.function_call_arguments.delta",
                        output_index=0,
                        delta=' "a.py"}',
                    ),
                    make_event(
                        "response.output_item.done",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            call_id="call_1",
                            name="read_file",
                            arguments='{"filename": "a.py"}',
                        ),
                    ),
                    make_event(
                        "response.reasoning_summary_text.delta", delta="pondering"
                    ),
                    make_event("response.output_text.delta", delta="Hel"),
                    make_event("response.output_text.delta", delta="lo"),
                    completed_event(output=[]),
                ]
            ]
        ),
    )
    usage_box = {}
    reasoning_box = []
    finish_box = {}
    tool_calls_box = []
    schemas = [{"type": "function", "name": "read_file", "description": "d", "parameters": {}}]
    chunks = collect(
        [{"role": "user", "content": "hi"}],
        reasoning_effort="medium",
        tools=schemas,
        usage_box=usage_box,
        reasoning_box=reasoning_box,
        finish_box=finish_box,
        tool_calls_box=tool_calls_box,
    )
    assert chunks == ["Hel", "lo"]
    assert reasoning_box == ["pondering"]
    assert usage_box == {"prompt_tokens": 11, "completion_tokens": 7}
    assert finish_box["finish_reason"] == "stop"
    assert tool_calls_box == [
        {"id": "call_1", "name": "read_file", "arguments": '{"filename": "a.py"}'}
    ]
    request = client.requests[0]
    assert request["model"] == "gpt-5.5"
    assert request["tools"] == schemas
    assert request["tool_choice"] == "auto"
    assert request["stream"] is True
    assert request["store"] is False
    assert client.streams[0].closed


def test_stream_codex_call_merges_completed_output_tool_calls(monkeypatch):
    """When completed.output does carry function calls they are collected too."""
    signed_in_auth(monkeypatch)
    install_sdk(
        monkeypatch,
        FakeCodexSDK(
            [
                [
                    completed_event(
                        output=[
                            SimpleNamespace(
                                type="function_call",
                                call_id="call_2",
                                name="list_files",
                                arguments='{"path": "."}',
                            )
                        ]
                    ),
                ]
            ]
        ),
    )
    tool_calls_box = []
    collect([], tool_calls_box=tool_calls_box)
    assert tool_calls_box == [
        {"id": "call_2", "name": "list_files", "arguments": '{"path": "."}'}
    ]


def test_stream_codex_call_retries_once_after_refresh_on_401(monkeypatch):
    auth = signed_in_auth(monkeypatch, refresh_token="stale")

    refreshes = []

    async def fake_refresh(current):
        refreshes.append(current.refresh_token)
        assert current is auth
        return codex_auth.CodexAuth(
            make_access_token(exp_offset=9999),
            "fresh",
            make_id_token(),
            account_id="acc_123",
        )

    monkeypatch.setattr(codex_client, "refresh_auth", fake_refresh)
    client = install_sdk(
        monkeypatch,
        FakeCodexSDK(
            [
                api_status_error(401, "expired"),
                [make_event("response.output_text.delta", delta="ok")],
            ]
        ),
    )
    assert collect([{"role": "user", "content": "hi"}]) == ["ok"]
    assert len(client.requests) == 2
    assert refreshes == ["stale"]
    # The retried request is identical.
    first = dict(client.requests[0])
    second = dict(client.requests[1])
    first.pop("prompt_cache_key"), second.pop("prompt_cache_key")
    assert first == second


def test_stream_codex_call_retry_clears_boxes_from_failed_attempt(monkeypatch):
    """A 401 that arrives after partial item events must not leave the failed
    attempt's tool calls/reasoning in the boxes (duplicates -> 400 on replay)."""
    signed_in_auth(monkeypatch, refresh_token="stale")

    async def fake_refresh(current):
        return codex_auth.CodexAuth(
            make_access_token(exp_offset=9999), "fresh", make_id_token()
        )

    monkeypatch.setattr(codex_client, "refresh_auth", fake_refresh)
    install_sdk(
        monkeypatch,
        FakeCodexSDK(
            [
                [
                    make_event(
                        "response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            call_id="call_stale",
                            name="list_files",
                            arguments="",
                        ),
                    ),
                    api_status_error(401, "expired mid-items"),
                ],
                [
                    make_event(
                        "response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            call_id="call_fresh",
                            name="list_files",
                            arguments="{}",
                        ),
                    ),
                    completed_event(),
                ],
            ]
        ),
    )
    tool_calls_box: list = []
    reasoning_items_box: list = []
    chunks = collect(
        [{"role": "user", "content": "hi"}],
        tool_calls_box=tool_calls_box,
        reasoning_items_box=reasoning_items_box,
    )
    assert chunks == []
    assert tool_calls_box == [
        {"id": "call_fresh", "name": "list_files", "arguments": "{}"}
    ]


def test_collector_captures_encrypted_reasoning_items(monkeypatch):
    signed_in_auth(monkeypatch)
    install_sdk(
        monkeypatch,
        FakeCodexSDK(
            [
                [
                    make_event(
                        "response.output_item.added",
                        output_index=0,
                        # The added event is intentionally incomplete; the
                        # collector must merge fields from item.done below.
                        item=SimpleNamespace(
                            type="reasoning",
                            id="rs_1",
                            encrypted_content=None,
                            summary=[],
                        ),
                    ),
                    make_event(
                        "response.reasoning_summary_text.delta",
                        delta="thinking out loud",
                    ),
                    make_event(
                        "response.output_item.done",
                        output_index=0,
                        item=SimpleNamespace(
                            type="reasoning",
                            id="rs_1",
                            encrypted_content="blob-1",
                        ),
                    ),
                    completed_event(output=[]),
                ]
            ]
        ),
    )
    reasoning_items_box: list = []
    reasoning_box: list = []
    collect([], reasoning_box=reasoning_box, reasoning_items_box=reasoning_items_box)
    # The same item arriving via added + done is captured once.
    assert reasoning_items_box == [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "blob-1",
        }
    ]
    # The visible reasoning text still streams as before.
    assert reasoning_box == ["thinking out loud"]


def test_collector_drops_anonymous_tool_calls():
    collector = codex_client._ToolCallCollector()
    collector._record({"id": "", "name": "", "arguments": '{"x": 1}'})
    assert collector.drain() == []


def test_collector_drain_keeps_named_call_with_empty_id():
    collector = codex_client._ToolCallCollector()
    collector._record({"id": "", "name": "list_files", "arguments": ""})
    assert collector.drain() == [
        {"id": "", "name": "list_files", "arguments": "{}"}
    ]


def test_stream_codex_call_does_not_retry_after_yielding(monkeypatch):
    signed_in_auth(monkeypatch)

    async def failing_refresh(auth):
        raise AssertionError("refresh must not run after deltas were yielded")

    monkeypatch.setattr(codex_client, "refresh_auth", failing_refresh)
    client = install_sdk(
        monkeypatch,
        FakeCodexSDK(
            [
                [
                    make_event("response.output_text.delta", delta="partial"),
                    LLMRequestError(401, "expired mid-stream"),
                ],
            ]
        ),
    )
    with pytest.raises(LLMRequestError, match="expired mid-stream"):
        collect([{"role": "user", "content": "hi"}])
    assert len(client.requests) == 1


def test_stream_codex_call_surfaces_backend_errors(monkeypatch):
    signed_in_auth(monkeypatch)
    install_sdk(
        monkeypatch,
        FakeCodexSDK([api_status_error(429, "limit reached")]),
    )
    with pytest.raises(LLMRequestError) as excinfo:
        collect([])
    assert excinfo.value.status_code == 429
    assert "limit reached" in excinfo.value.message


def test_stream_codex_call_raises_on_failed_event(monkeypatch):
    signed_in_auth(monkeypatch)
    failed = make_event(
        "response.failed",
        response=SimpleNamespace(
            status=400, error=SimpleNamespace(message="model unsupported")
        ),
    )
    install_sdk(monkeypatch, FakeCodexSDK([[failed]]))
    with pytest.raises(LLMRequestError, match="model unsupported"):
        collect([])


def test_fetch_codex_models_parses_slugs(monkeypatch):
    auth = signed_in_auth(monkeypatch)
    assert auth is not None

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == codex_client.CODEX_RELEASE_URL:
            assert "authorization" not in request.headers
            assert "chatgpt-account-id" not in request.headers
            return httpx.Response(200, json={"version": "0.201.0"})
        assert request.url.params["client_version"] == "0.201.0"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "display_name": "GPT-5.6-Sol",
                        "description": "Latest frontier agentic coding model.",
                        "context_window": 400000,
                    },
                    {"slug": "gpt-5.5", "context_window": 200000},
                    {"slug": "hidden-one", "visibility": "hidden"},
                    "not-a-dict",
                    {"slug": "gpt-5.6-sol"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        codex_client,
        "_CLIENT_FACTORY",
        lambda: httpx.AsyncClient(
            transport=transport, timeout=codex_client.HTTP_TIMEOUT
        ),
    )
    rows = asyncio.run(codex_client.fetch_codex_models())
    assert rows == [
        {
            "id": "gpt-5.6-sol",
            "display": "GPT-5.6-Sol",
            "description": "Latest frontier agentic coding model.",
            "context_window": 400000,
        },
        {
            "id": "gpt-5.5",
            "display": "GPT 5.5",
            "description": "",
            "context_window": 200000,
        },
    ]


def test_agent_routes_codex_provider_through_codex_stream(monkeypatch):
    import remie.agent as agent

    previous = agent.get_config()
    calls = {}

    async def fake_codex_stream(conversation, model, reasoning_effort="medium", **kw):
        calls["model"] = model
        calls["effort"] = reasoning_effort
        calls["tools"] = kw.get("tools")
        box = kw.get("tool_calls_box")
        if box is not None:
            box.append({"id": "c9", "name": "list_files", "arguments": "{}"})
        yield "codex-reply"

    monkeypatch.setattr(codex_client, "stream_codex_call", fake_codex_stream)
    try:
        agent.configure_openai(
            codex_client.CODEX_BACKEND_BASE,
            "",
            "gpt-5.5",
            provider="codex",
            reasoning_effort="high",
        )
        assert agent._provider_defaults("codex").model == agent.CODEX_MODELS[0]
        assert agent.get_model_context_limit("gpt-5.5", "codex") == 272_000
        assert agent.supports_reasoning_effort("gpt-5.5", "codex") is True

        async def collect_agent():
            box = []
            chunks = [
                chunk
                async for chunk in agent.stream_llm_call([], tool_calls_box=box)
            ]
            return chunks, box

        chunks, box = asyncio.run(collect_agent())
        assert chunks == ["codex-reply"]
        assert calls["model"] == "gpt-5.5"
        assert calls["effort"] == "high"
        tool_names = [tool["name"] for tool in calls["tools"]]
        assert "read_file" in tool_names and "memory" in tool_names
        assert all(tool["type"] == "function" for tool in calls["tools"])
        assert box == [{"id": "c9", "name": "list_files", "arguments": "{}"}]
        assert agent.get_full_system_prompt(native_tools=True).count("'thinking:'") == 0
        assert "'thinking:'" not in agent.get_full_system_prompt(native_tools=True)
        assert "tool: TOOL_NAME" in agent.get_full_system_prompt(native_tools=False)
    finally:
        agent.configure_openai(
            previous.base_url,
            previous.api_key,
            previous.model,
            previous.provider,
            previous.reasoning_effort,
            previous.verify_ssl,
        )


@pytest.mark.parametrize("failure", ["http", "invalid", "prerelease"])
def test_codex_release_discovery_refreshes_and_retains_last_success(monkeypatch, failure):
    monkeypatch.setattr(codex_client, "_last_codex_client_version", None)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert "authorization" not in request.headers
        if calls <= 2:
            return httpx.Response(200, json={"version": f"0.{200 + calls}.0"})
        if failure == "http":
            return httpx.Response(503)
        return httpx.Response(200, json={"version": "bad" if failure == "invalid" else "0.999.0-beta.1"})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await codex_client._discover_codex_client_version(client) == "0.201.0"
            assert await codex_client._discover_codex_client_version(client) == "0.202.0"
            assert await codex_client._discover_codex_client_version(client) == "0.202.0"
    asyncio.run(exercise())


def test_codex_release_discovery_failure_without_previous_version(monkeypatch):
    monkeypatch.setattr(codex_client, "_last_codex_client_version", None)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        )) as client:
            assert await codex_client._discover_codex_client_version(client) is None
    asyncio.run(exercise())
