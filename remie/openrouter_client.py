"""Streaming client for OpenRouter (https://openrouter.ai) using httpx.

Native function calling: Remie's tools travel as chat-completions function
definitions and the model returns ``tool_calls`` deltas that are assembled by
index, executed locally, and replayed as role-``tool`` messages. No SDK — the
SSE stream is parsed directly from httpx.
"""

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from remie.errors import LLMRequestError
from remie.model_names import VENDORS, prettify_model_id

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

HTTP_TIMEOUT = httpx.Timeout(connect=15, read=600, write=60, pool=15)

REASONING_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    # Remie's "max" maps to the API's high tier; OpenRouter models do not
    # uniformly accept xhigh.
    "max": "high",
}

# Test seam: when set, returns the httpx.AsyncClient used for backend calls
# (lets tests inject an httpx.MockTransport without patching globals).
_CLIENT_FACTORY: Callable[[], httpx.AsyncClient] | None = None


def _create_client() -> httpx.AsyncClient:
    if _CLIENT_FACTORY is not None:
        return _CLIENT_FACTORY()
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT)


def conversation_to_messages(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a Remie conversation to chat-completions messages.

    Native tool calling: assistant ``tool_calls`` become
    ``{"role": "assistant", "tool_calls": [{"id", "type": "function",
    "function": {"name", "arguments"}}]}`` and role-``tool`` results become
    ``{"role": "tool", "tool_call_id", ...}``.
    """
    messages: list[dict[str, Any]] = []
    for message in conversation:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            images = [
                part["image_url"]["url"]
                for part in content
                if isinstance(part, dict)
                and isinstance(part.get("image_url"), dict)
                and isinstance(part["image_url"].get("url"), str)
                and part["image_url"]["url"]
            ]
        else:
            text_parts = [content] if isinstance(content, str) else []
            images = []
        text = "\n".join(text_parts)
        if role == "system":
            if text:
                messages.append({"role": "system", "content": text})
            continue
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            outgoing: dict[str, Any] = {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": str(call.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or ""),
                            "arguments": str(call.get("arguments") or "{}"),
                        },
                    }
                    for call in tool_calls
                    if isinstance(call, dict)
                ],
            }
            messages.append(outgoing)
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "content": text,
                }
            )
            continue
        if role == "assistant":
            if not text:
                continue
            messages.append({"role": "assistant", "content": text})
            continue
        # User message: plain text or multimodal parts.
        if not text and not images:
            continue
        if images and text:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        *[
                            {"type": "image_url", "image_url": {"url": url}}
                            for url in images
                        ],
                    ],
                }
            )
        elif images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in images
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": text})
    return messages


def chat_tool_schemas(flat_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert flat Responses-API tool definitions to chat-completions form."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema.get("name", ""),
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {"type": "object"}),
            },
        }
        for schema in flat_schemas
    ]


def build_request_payload(
    conversation: list[dict[str, Any]],
    model: str,
    reasoning_effort: str = "off",
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": conversation_to_messages(conversation),
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": f"remie-{uuid.uuid4().hex[:8]}",
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    effort = REASONING_EFFORT_MAP.get(reasoning_effort)
    if reasoning_effort != "off" and effort:
        payload["reasoning"] = {"effort": effort}
    if tools:
        payload["tools"] = chat_tool_schemas(tools)
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
    return payload


def _error_message(status_code: int, body: str) -> str:
    detail = ""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
        elif isinstance(error, str):
            detail = error
        detail = detail or str(parsed.get("message") or "")
    if status_code == 401:
        return "OpenRouter rejected the API key. Check it under Ctrl+P → OpenRouter."
    if status_code == 402:
        return (
            "Your OpenRouter account is out of credits. Top up at "
            "openrouter.ai/credits."
        )
    if status_code == 429:
        return "OpenRouter rate limit reached. Try again shortly."
    if status_code >= 400:
        prefix = detail or body[:300]
        return f"OpenRouter request failed ({status_code}): {prefix}"
    return detail or f"OpenRouter request failed ({status_code})"


class _ToolCallAssembler:
    """Assembles streamed ``delta.tool_calls`` fragments into complete calls.

    Fragments arrive keyed by ``index``; the id/name typically arrive once in
    the first fragment while arguments accumulate across chunks.
    """

    def __init__(self) -> None:
        self.calls: dict[int, dict[str, str]] = {}

    def feed(self, fragments: Any) -> None:
        for fragment in fragments or []:
            if not isinstance(fragment, dict):
                continue
            index = fragment.get("index")
            index = index if isinstance(index, int) else len(self.calls)
            entry = self.calls.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            call_id = fragment.get("id")
            if isinstance(call_id, str) and call_id and not entry["id"]:
                entry["id"] = call_id
            function = fragment.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name and not entry["name"]:
                entry["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry["arguments"] += arguments

    def drain(self) -> list[dict[str, str]]:
        calls = [dict(self.calls[index]) for index in sorted(self.calls)]
        for call in calls:
            if not call["id"]:
                call["id"] = f"call_{uuid.uuid4().hex[:16]}"
            if not call.get("arguments"):
                call["arguments"] = "{}"
        return calls


async def _handle_error_response(response: httpx.Response) -> LLMRequestError:
    body = (await response.aread()).decode("utf-8", errors="replace").strip()
    return LLMRequestError(
        response.status_code,
        _error_message(response.status_code, body),
    )


async def _stream_once(
    api_key: str,
    payload: dict[str, Any],
    usage_box: dict[str, int] | None,
    reasoning_box: list[str] | None,
    finish_box: dict[str, Any] | None,
    tool_calls_box: list[dict[str, str]] | None,
) -> AsyncIterator[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        # Attribution headers recommended by OpenRouter.
        "HTTP-Referer": "https://github.com/FXmario/Remie",
        "X-Title": "Remie",
    }
    saw_done = False
    assembler = _ToolCallAssembler()
    async with _create_client() as client:
        async with client.stream(
            "POST",
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                raise await _handle_error_response(response)
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    saw_done = True
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                error = chunk.get("error")
                if error:
                    message = (
                        error.get("message", "")
                        if isinstance(error, dict)
                        else str(error)
                    )
                    raise LLMRequestError(502, message or "OpenRouter stream error")
                usage = chunk.get("usage")
                if usage_box is not None and isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    if isinstance(prompt_tokens, int):
                        usage_box["prompt_tokens"] = prompt_tokens
                    if isinstance(completion_tokens, int):
                        usage_box["completion_tokens"] = completion_tokens
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_box is not None and finish_reason:
                    finish_box["finish_reason"] = finish_reason
                    finish_box["truncated"] = finish_reason in {
                        "length",
                        "max_tokens",
                        "max_completion_tokens",
                    }
                delta = choice.get("delta") or {}
                if reasoning_box is not None:
                    reason = delta.get("reasoning") or delta.get("reasoning_content")
                    if reason:
                        reasoning_box.append(reason)
                if tool_calls_box is not None and delta.get("tool_calls"):
                    assembler.feed(delta["tool_calls"])
                content = delta.get("content")
                if content:
                    yield content
    if tool_calls_box is not None:
        tool_calls_box.extend(assembler.drain())
    if finish_box is not None:
        finish_box["stream_complete"] = saw_done or bool(
            finish_box.get("finish_reason")
        )


async def stream_openrouter_call(
    api_key: str,
    conversation: list[dict[str, Any]],
    model: str,
    reasoning_effort: str = "off",
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    usage_box: dict[str, int] | None = None,
    reasoning_box: list[str] | None = None,
    finish_box: dict[str, Any] | None = None,
    tool_calls_box: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Yield assistant text deltas; completed native tool calls land in
    ``tool_calls_box`` as ``{"id", "name", "arguments"}``."""
    payload = build_request_payload(
        conversation, model, reasoning_effort, tools, max_tokens
    )
    async for chunk in _stream_once(
        api_key,
        payload,
        usage_box,
        reasoning_box,
        finish_box,
        tool_calls_box,
    ):
        yield chunk


async def fetch_openrouter_models() -> list[dict[str, Any]]:
    """Fetch live catalog entries as metadata rows; [] on failure.

    Rows carry ``{"id", "display", "vendor", "context_length", "free"}``.
    OpenRouter's ``name`` field is formatted like ``"Meta: Muse Spark 1.2"``,
    so vendor is split out when present; ids fall back to heuristics.
    """
    try:
        async with _create_client() as client:
            response = await client.get(OPENROUTER_MODELS_URL, timeout=10)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError, ValueError:
        return []
    rows = payload.get("data") if isinstance(payload, dict) else None
    results: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        display = ""
        vendor = ""
        raw_name = row.get("name")
        if isinstance(raw_name, str) and ": " in raw_name:
            vendor_part, _, display = raw_name.partition(": ")
            vendor = VENDORS.get(vendor_part.strip().lower(), vendor_part.strip())
        info = prettify_model_id(model_id)
        display = display or info.display or model_id
        vendor = vendor or info.vendor
        context_length = row.get("context_length")
        pricing = row.get("pricing")
        free = False
        if isinstance(pricing, dict):
            try:
                free = (
                    float(pricing.get("prompt") or 0) == 0
                    and float(pricing.get("completion") or 0) == 0
                )
            except TypeError, ValueError:
                free = False
        results.append(
            {
                "id": model_id,
                "display": display,
                "vendor": vendor,
                "context_length": (
                    context_length if isinstance(context_length, int) else 0
                ),
                "free": free,
            }
        )
    return results
