"""Streaming client for the ChatGPT-subscription Codex Responses backend.

Talks directly to ``https://chatgpt.com/backend-api/codex`` using the official
OpenAI Python SDK pointed at that base URL, so a ChatGPT Plus/Pro plan can
drive Remie without an API key. Tool calling is native: Remie's tools are sent
as Responses-API function definitions and the model returns ``function_call``
output items that Remie executes and feeds back as ``function_call_output``.
"""

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from openai import APIStatusError, AsyncOpenAI

from remie.agent import LLMRequestError
from remie.codex_auth import CodexAuth, ensure_valid_auth, refresh_auth
from remie.model_names import prettify_model_name

CODEX_BACKEND_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_MODELS_URL = f"{CODEX_BACKEND_BASE}/models"
# The subscription backend expects the CLI originator header and a
# client_version query parameter on GET /models. The returned model list is
# gated by this version: older clients get older lineups (e.g. 0.136.0 misses
# the GPT-5.6 family), so track a recent CLI release here.
CODEX_CLIENT_VERSION = "0.149.0"
# The CLI originator header value expected by the backend.
CODEX_ORIGINATOR = "codex_cli_rs"

HTTP_TIMEOUT = httpx.Timeout(connect=15, read=600, write=60, pool=15)

REASONING_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    # Remie's "max" maps to the Responses API's xhigh tier.
    "max": "xhigh",
}

# Test seams: when set, these build the HTTP client used by fetch_codex_models
# (httpx.MockTransport) and the AsyncOpenAI client used by stream_codex_call.
_CLIENT_FACTORY: Callable[[], httpx.AsyncClient] | None = None
_SDK_FACTORY: Callable[[str, str], Any] | None = None


class CodexStreamError(RuntimeError):
    """Raised when the Codex backend stream fails mid-response."""


def _create_client() -> httpx.AsyncClient:
    if _CLIENT_FACTORY is not None:
        return _CLIENT_FACTORY()
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT)


def _create_sdk_client(access_token: str, account_id: str) -> Any:
    """Build an OpenAI SDK client bound to one access token."""
    if _SDK_FACTORY is not None:
        return _SDK_FACTORY(access_token, account_id)
    headers = {
        "OpenAI-Beta": "responses=experimental",
        "originator": CODEX_ORIGINATOR,
    }
    if account_id:
        headers["chatgpt-account-id"] = account_id
    return AsyncOpenAI(
        api_key=access_token,
        base_url=CODEX_BACKEND_BASE,
        default_headers=headers,
        timeout=HTTP_TIMEOUT,
        max_retries=0,
    )


def _text_and_images(content: Any) -> tuple[str, list[str]]:
    """Extract text and image data URLs from a chat message content value."""
    texts: list[str] = []
    images: list[str] = []
    if isinstance(content, str):
        return content, images
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
                image = part.get("image_url")
                if isinstance(image, dict) and isinstance(image.get("url"), str):
                    if image["url"]:
                        images.append(image["url"])
                elif isinstance(image, str) and image:
                    images.append(image)
    return "\n".join(texts), images


def conversation_to_input(
    conversation: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert a chat-completions conversation to Responses API form.

    System messages become top-level ``instructions``; user/assistant messages
    become input items with typed content parts; assistant ``tool_calls`` and
    role-``tool`` results become ``function_call`` / ``function_call_output``
    items.
    """
    instructions_parts: list[str] = []
    items: list[dict[str, Any]] = []
    for message in conversation:
        role = message.get("role")
        text, images = _text_and_images(message.get("content", ""))
        if role == "system":
            if text:
                instructions_parts.append(text)
            continue
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or ""),
                        "name": str(call.get("name") or ""),
                        "arguments": str(call.get("arguments") or "{}"),
                    }
                )
            continue
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": text,
                }
            )
            continue
        content: list[dict[str, Any]] = []
        if role == "assistant":
            if not text:
                continue
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
            continue
        if text:
            content.append({"type": "input_text", "text": text})
        for url in images:
            content.append({"type": "input_image", "image_url": url})
        if not content:
            continue
        items.append({"type": "message", "role": "user", "content": content})
    return "\n\n".join(instructions_parts), items


def build_request_payload(
    conversation: list[dict[str, Any]],
    model: str,
    reasoning_effort: str = "medium",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``client.responses.create`` (without stream flags)."""
    instructions, items = conversation_to_input(conversation)
    if not items:
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue."}],
            }
        ]
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": items,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": str(uuid.uuid4()),
    }
    effort = REASONING_EFFORT_MAP.get(reasoning_effort)
    if reasoning_effort != "off" and effort:
        payload["reasoning"] = {"effort": effort, "summary": "auto"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
    return payload


def _friendly_error(status_code: int, message: str) -> str:
    if status_code == 401:
        return (
            "ChatGPT session expired or invalid. Reconnect via Ctrl+P → "
            "Codex (ChatGPT); sign in again if it persists."
        )
    if status_code == 403:
        return (
            "This ChatGPT account cannot use Codex. A Plus, Pro, Business, Edu, "
            "or Enterprise subscription is required."
        )
    if status_code == 429:
        return "Codex usage limit reached for your ChatGPT plan. Try again later."
    prefix = f"{message}" if message else f"Codex request failed ({status_code})"
    return f"Codex request failed ({status_code}): {prefix}"


def _raise_llm_error(error: APIStatusError) -> None:
    status = int(getattr(error, "status_code", 500) or 500)
    body = getattr(error, "body", None)
    detail = ""
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            detail = str(inner.get("message") or "")
        elif isinstance(inner, str):
            detail = inner
    raise LLMRequestError(status, _friendly_error(status, detail)) from error


def _extract_tool_calls(output_items: Any) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for item in output_items or []:
        if getattr(item, "type", "") != "function_call":
            continue
        call_id = getattr(item, "call_id", None) or getattr(item, "id", "") or ""
        name = getattr(item, "name", "") or ""
        arguments = getattr(item, "arguments", None) or "{}"
        calls.append({"id": str(call_id), "name": str(name), "arguments": str(arguments)})
    return calls


class _ToolCallCollector:
    """Assembles function_call items from streaming item events.

    The Codex backend streams tools as ``response.output_item.added`` followed
    by ``response.function_call_arguments.delta`` chunks and
    ``response.output_item.done``; the final ``response.completed`` event may
    repeat the full output array (or be empty), so results are merged by
    ``call_id`` across all sources.
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}
        self._collected: dict[str, dict[str, str]] = {}
        self._order: list[str] = []

    def _record(self, call: dict[str, str]) -> None:
        call_id = call.get("id") or ""
        key = call_id or f"anon-{call.get('name')}-{len(self._order)}"
        existing = self._collected.get(key)
        if existing is None:
            self._collected[key] = dict(call)
            self._order.append(key)
            return
        for field, value in call.items():
            if not value:
                continue
            if field == "arguments":
                if len(value) > len(existing.get("arguments", "")):
                    existing["arguments"] = value
            elif not existing.get(field):
                existing[field] = value

    def on_item_added(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if getattr(item, "type", "") != "function_call":
            return
        index = getattr(event, "output_index", None)
        entry = {
            "id": str(
                getattr(item, "call_id", None) or getattr(item, "id", "") or ""
            ),
            "name": str(getattr(item, "name", "") or ""),
            "arguments": str(getattr(item, "arguments", "") or ""),
        }
        if isinstance(index, int):
            self._by_index[index] = entry
        else:
            self._record(entry)

    def on_arguments_delta(self, event: Any) -> None:
        delta = getattr(event, "delta", "")
        if not delta:
            return
        index = getattr(event, "output_index", None)
        target = self._by_index.get(index) if isinstance(index, int) else None
        if target is None and self._by_index:
            target = next(reversed(self._by_index.values()))
        if target is None:
            target = {"id": "", "name": "", "arguments": ""}
            new_index = (
                max(self._by_index.keys()) + 1 if self._by_index else 0
            )
            self._by_index[new_index] = target
        target["arguments"] += delta

    def on_item_done(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if getattr(item, "type", "") != "function_call":
            return
        index = getattr(event, "output_index", None)
        if isinstance(index, int):
            self._by_index.pop(index, None)
        self._record(
            {
                "id": str(
                    getattr(item, "call_id", None) or getattr(item, "id", "") or ""
                ),
                "name": str(getattr(item, "name", "") or ""),
                "arguments": str(getattr(item, "arguments", "") or "{}"),
            }
        )

    def on_completed(self, response: Any) -> None:
        for call in _extract_tool_calls(getattr(response, "output", None)):
            self._record(call)
        for entry in self._by_index.values():
            self._record(entry)
        self._by_index.clear()

    def drain(self) -> list[dict[str, str]]:
        calls = [dict(self._collected[key]) for key in self._order]
        for call in calls:
            if not call.get("arguments"):
                call["arguments"] = "{}"
        return calls


async def _stream_sdk_once(
    auth: CodexAuth,
    payload: dict[str, Any],
    usage_box: dict[str, int] | None,
    reasoning_box: list[str] | None,
    finish_box: dict[str, Any] | None,
    tool_calls_box: list[dict[str, str]] | None,
) -> AsyncIterator[str]:
    client = _create_sdk_client(auth.access_token, auth.account_id)
    try:
        stream = await client.responses.create(**payload, stream=True)
    except APIStatusError as error:
        _raise_llm_error(error)

    saw_completed = False
    collector = _ToolCallCollector()
    try:
        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = event.delta
                if delta:
                    yield delta
            elif event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                if reasoning_box is not None and event.delta:
                    reasoning_box.append(event.delta)
            elif event_type == "response.output_item.added":
                collector.on_item_added(event)
            elif event_type == "response.function_call_arguments.delta":
                collector.on_arguments_delta(event)
            elif event_type == "response.output_item.done":
                collector.on_item_done(event)
            elif event_type == "response.completed":
                saw_completed = True
                response = event.response
                usage = getattr(response, "usage", None)
                if usage_box is not None and usage is not None:
                    input_tokens = getattr(usage, "input_tokens", None)
                    output_tokens = getattr(usage, "output_tokens", None)
                    if isinstance(input_tokens, int):
                        usage_box["prompt_tokens"] = input_tokens
                    if isinstance(output_tokens, int):
                        usage_box["completion_tokens"] = output_tokens
                collector.on_completed(response)
                if finish_box is not None:
                    finish_box["finish_reason"] = "stop"
                    finish_box["truncated"] = False
                    finish_box["stream_complete"] = True
            elif event_type == "response.failed":
                failed = getattr(event, "response", None)
                error = getattr(failed, "error", None)
                message = getattr(error, "message", None) or "Codex response failed"
                status = getattr(failed, "status", None)
                code = int(status) if isinstance(status, int) else 502
                raise LLMRequestError(code, str(message))
            elif event_type == "error":
                message = getattr(event, "message", None) or "Codex stream error"
                raise LLMRequestError(502, str(message))
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass
    if tool_calls_box is not None:
        tool_calls_box.extend(collector.drain())
    if finish_box is not None and not saw_completed:
        finish_box["stream_complete"] = bool(finish_box.get("finish_reason"))


async def stream_codex_call(
    conversation: list[dict[str, Any]],
    model: str,
    reasoning_effort: str = "medium",
    tools: list[dict[str, Any]] | None = None,
    usage_box: dict[str, int] | None = None,
    reasoning_box: list[str] | None = None,
    finish_box: dict[str, Any] | None = None,
    tool_calls_box: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Yield assistant text deltas from the ChatGPT-subscription backend.

    Native function calling: when ``tools`` is provided the model may return
    ``function_call`` output items, which are collected into ``tool_calls_box``
    (each ``{"id", "name", "arguments"}``) after ``response.completed``.
    Refreshes expired tokens up front; retries once after a forced refresh
    when the backend answers 401 before anything has been streamed.
    """
    payload = build_request_payload(conversation, model, reasoning_effort, tools)
    auth = await ensure_valid_auth()
    yielded = False
    try:
        async for chunk in _stream_sdk_once(
            auth, payload, usage_box, reasoning_box, finish_box, tool_calls_box
        ):
            yielded = True
            yield chunk
        return
    except LLMRequestError as error:
        if error.status_code != 401 or yielded or not auth.refresh_token:
            raise
    auth = await refresh_auth(auth)
    async for chunk in _stream_sdk_once(
        auth, payload, usage_box, reasoning_box, finish_box, tool_calls_box
    ):
        yield chunk


async def fetch_codex_models() -> list[dict[str, Any]]:
    """Fetch the account's live Codex model metadata; [] on failure.

    Rows carry ``{"id", "display", "description", "context_window"}`` straight
    from the backend's ``display_name`` / ``description`` fields.
    """
    from remie.codex_auth import load_auth

    auth = load_auth()
    if auth is None:
        return []
    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "originator": CODEX_ORIGINATOR,
    }
    if auth.account_id:
        headers["chatgpt-account-id"] = auth.account_id
    try:
        async with _create_client() as client:
            response = await client.get(
                CODEX_MODELS_URL,
                params={"client_version": CODEX_CLIENT_VERSION},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return []
    rows = payload.get("models") if isinstance(payload, dict) else None
    results: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug or slug in seen:
            continue
        seen.add(slug)
        if row.get("visibility") == "hidden":
            continue
        display_name = row.get("display_name")
        description = row.get("description")
        context_window = row.get("context_window")
        results.append(
            {
                "id": slug,
                "display": (
                    display_name.strip()
                    if isinstance(display_name, str) and display_name.strip()
                    else prettify_model_name(slug)
                ),
                "description": (
                    description.strip() if isinstance(description, str) else ""
                ),
                "context_window": (
                    context_window if isinstance(context_window, int) else 0
                ),
            }
        )
    return results
