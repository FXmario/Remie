"""Routing and streaming adapters for all supported model providers.

This module owns provider-specific request construction.  Runtime objects are
injected by ``remie.agent`` during the compatibility migration, which keeps
connection state out of the provider implementation and makes the router
independently testable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIStatusError

from remie.config import ConnectionConfig, OPENCODE_GO_BASE_URL
from remie.errors import LLMRequestError
from remie.providers.events import (
    FinishEvent,
    ProviderEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from remie.tools import get_tool_schemas

TRUNCATED_REASONS = {"length", "max_tokens", "max_completion_tokens"}


async def _stream_local_sdk_call(
    client: Any,
    payload: dict[str, Any],
    usage_box: dict[str, int] | None,
    reasoning_box: list[str] | None,
    finish_box: dict[str, Any] | None,
) -> AsyncIterator[str]:
    try:
        stream = await client.chat.completions.create(**payload)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                usage = getattr(chunk, "usage", None)
                if usage_box is not None and usage is not None:
                    usage_box["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                    usage_box["completion_tokens"] = (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_box is not None and finish_reason is not None:
                finish_box["finish_reason"] = finish_reason
                finish_box["truncated"] = finish_reason in TRUNCATED_REASONS
            delta = getattr(choice, "delta", None)
            if reasoning_box is not None and delta is not None:
                reason = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reason:
                    reasoning_box.append(reason)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content
        if finish_box is not None:
            finish_box["stream_complete"] = True
    except APIStatusError as error:
        raise LLMRequestError(error.status_code, str(error)) from error


async def stream_provider_call(
    config: ConnectionConfig,
    conversation: list[dict[str, Any]],
    *,
    get_http_client: Callable[[], httpx.AsyncClient],
    get_local_openai_client: Callable[[], Any],
    max_output_tokens: int,
    reasoning_supported: bool,
    usage_box: dict[str, int] | None = None,
    reasoning_box: list[str] | None = None,
    finish_box: dict[str, Any] | None = None,
    tool_calls_box: list[dict[str, str]] | None = None,
    reasoning_items_box: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Stream one response from the configured provider.

    The mutable result boxes are retained as a temporary compatibility API;
    new provider-facing code should use the typed events in
    :mod:`remie.providers.events`.
    """
    if config.provider == "codex":
        from remie.codex_client import stream_codex_call

        async for content in stream_codex_call(
            conversation,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            tools=get_tool_schemas(),
            usage_box=usage_box,
            reasoning_box=reasoning_box,
            finish_box=finish_box,
            tool_calls_box=tool_calls_box,
            reasoning_items_box=reasoning_items_box,
        ):
            yield content
        return

    if config.provider == "openrouter":
        from remie.openrouter_client import stream_openrouter_call

        async for content in stream_openrouter_call(
            config.api_key,
            conversation,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            tools=get_tool_schemas(),
            max_tokens=max_output_tokens,
            usage_box=usage_box,
            reasoning_box=reasoning_box,
            finish_box=finish_box,
            tool_calls_box=tool_calls_box,
        ):
            yield content
        return

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": conversation,
        "max_tokens": max_output_tokens,
        "stream": True,
    }
    if config.reasoning_effort != "off" and reasoning_supported:
        payload["reasoning_effort"] = config.reasoning_effort
    if usage_box is not None and config.base_url.rstrip("/") == OPENCODE_GO_BASE_URL:
        payload["stream_options"] = {"include_usage": True}

    client = get_http_client()
    if config.provider == "local" and isinstance(client, httpx.AsyncClient):
        async for content in _stream_local_sdk_call(
            get_local_openai_client(), payload, usage_box, reasoning_box, finish_box
        ):
            yield content
        return

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", errors="replace").strip()
            raise LLMRequestError(
                response.status_code, body or f"HTTP {response.status_code}"
            )
        saw_done = False
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
                import json

                chunk = json.loads(data)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                usage = chunk.get("usage")
                if usage_box is not None and usage is not None:
                    usage_box["prompt_tokens"] = usage.get("prompt_tokens") or 0
                    usage_box["completion_tokens"] = usage.get("completion_tokens") or 0
                continue
            choice = choices[0]
            if finish_box is not None:
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    finish_box["finish_reason"] = finish_reason
                    finish_box["truncated"] = finish_reason in TRUNCATED_REASONS
            delta = choice.get("delta") or {}
            if reasoning_box is not None:
                reason = delta.get("reasoning_content") or delta.get("reasoning")
                if reason:
                    reasoning_box.append(reason)
            content = delta.get("content")
            if content:
                yield content
        if finish_box is not None:
            finish_box["stream_complete"] = saw_done or bool(
                finish_box.get("finish_reason")
            )


@dataclass
class RoutedProvider:
    """Provider-neutral adapter over Remie's existing backend clients."""

    config: ConnectionConfig
    get_http_client: Callable[[], httpx.AsyncClient]
    get_local_openai_client: Callable[[], Any]
    max_output_tokens: int
    reasoning_supported: bool
    reasoning_poll_interval: float = 0.02

    async def stream(
        self, conversation: list[dict[str, Any]]
    ) -> AsyncIterator[ProviderEvent]:
        usage: dict[str, int] = {}
        reasoning: list[str] = []
        finish: dict[str, Any] = {}
        tool_calls: list[dict[str, str]] = []
        reasoning_items: list[dict[str, str]] = []
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        failure: list[BaseException] = []

        async def produce() -> None:
            try:
                async for text in stream_provider_call(
                    self.config,
                    conversation,
                    get_http_client=self.get_http_client,
                    get_local_openai_client=self.get_local_openai_client,
                    max_output_tokens=self.max_output_tokens,
                    reasoning_supported=self.reasoning_supported,
                    usage_box=usage,
                    reasoning_box=reasoning,
                    finish_box=finish,
                    tool_calls_box=tool_calls,
                    reasoning_items_box=reasoning_items,
                ):
                    await text_queue.put(text)
            except BaseException as error:
                failure.append(error)
            finally:
                await text_queue.put(None)

        producer = asyncio.create_task(produce())
        consumed_reasoning = 0
        completed = False
        try:
            while not completed:
                try:
                    text = await asyncio.wait_for(
                        text_queue.get(), timeout=self.reasoning_poll_interval
                    )
                    completed = text is None
                    if text is not None:
                        yield TextDelta(text)
                except TimeoutError:
                    pass

                if len(reasoning) > consumed_reasoning:
                    for delta in reasoning[consumed_reasoning:]:
                        yield ReasoningDelta(delta)
                    consumed_reasoning = len(reasoning)

            await producer
            if failure:
                raise failure[0]

            if usage:
                yield UsageEvent(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
            for call in tool_calls:
                yield ToolCallEvent(
                    str(call.get("id") or ""),
                    str(call.get("name") or ""),
                    str(call.get("arguments") or "{}"),
                )
            yield FinishEvent(
                finish.get("finish_reason"),
                bool(finish.get("truncated")),
                bool(finish.get("stream_complete")),
                {"reasoning_items": list(reasoning_items)},
            )
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except asyncio.CancelledError:
                    pass
