"""Provider-independent response processing and tool execution.

The Textual application still owns presentation and stream throttling; this
class owns the model-neutral rules that were previously embedded in the UI.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from remie.core.events import (
    AgentEvent,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnReasoningDelta,
    TurnRetrying,
    TurnTextDelta,
    TurnUsage,
)
from remie.protocol import extract_tool_invocations, strip_protocol_lines
from remie.providers.base import Provider
from remie.providers.events import (
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
)
from remie.tools.executor import ToolExecutor


@dataclass(frozen=True)
class PendingToolCall:
    id: str | None
    name: str
    args: dict[str, Any]
    raw_arguments: str | None = None


@dataclass(frozen=True)
class PreparedResponse:
    raw_text: str
    content: str
    tool_calls: tuple[PendingToolCall, ...]

    @property
    def tool_invocations(self) -> list[tuple[str, dict[str, Any]]]:
        return [(call.name, call.args) for call in self.tool_calls]


class AgentRunner:
    """Core response/tool policy shared by every user interface."""

    def __init__(
        self,
        tool_executor: ToolExecutor,
        provider: Provider | None = None,
    ) -> None:
        self.tool_executor = tool_executor
        self.provider = provider

    def prepare_response(
        self,
        full_text: str,
        *,
        native_tool_calling: bool,
        native_calls: list[dict[str, str]] | None = None,
    ) -> PreparedResponse:
        if native_tool_calling:
            calls: list[PendingToolCall] = []
            for call in native_calls or []:
                raw_arguments = str(call.get("arguments") or "{}")
                try:
                    args = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append(
                    PendingToolCall(
                        id=str(call.get("id") or ""),
                        name=str(call.get("name") or ""),
                        args=args,
                        raw_arguments=raw_arguments,
                    )
                )
        else:
            calls = [
                PendingToolCall(None, name, args)
                for name, args in extract_tool_invocations(full_text)
            ]
        return PreparedResponse(
            raw_text=full_text,
            content=strip_protocol_lines(full_text).strip(),
            tool_calls=tuple(calls),
        )

    def assistant_metadata(
        self,
        response: PreparedResponse,
        reasoning_items: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": call.id or "",
                    "name": call.name,
                    "arguments": call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.args),
                }
                for call in response.tool_calls
            ]
        }
        if reasoning_items:
            metadata["codex_reasoning"] = list(reasoning_items)
        return metadata

    async def execute_tool(self, call: PendingToolCall) -> dict[str, Any]:
        return await self.tool_executor.execute(call.name, call.args)

    async def run_turn(
        self,
        conversation: list[dict[str, Any]],
        user_content: Any,
        *,
        native_tool_calling: bool,
        max_empty_retries: int = 2,
        max_continuations: int = 10,
    ) -> AsyncIterator[AgentEvent]:
        """Run a complete model/tool loop without importing any UI framework."""
        if self.provider is None:
            raise RuntimeError("AgentRunner.run_turn requires a provider")

        conversation.append({"role": "user", "content": user_content})
        empty_retries = 0
        continuations = 0
        while True:
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            native_calls: list[dict[str, str]] = []
            reasoning_items: list[dict[str, str]] = []
            finish: FinishEvent | None = None

            async for event in self.provider.stream(conversation):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                    yield TurnTextDelta(event.text)
                elif isinstance(event, ReasoningDelta):
                    reasoning_parts.append(event.text)
                    yield TurnReasoningDelta(event.text)
                elif isinstance(event, ToolCallEvent):
                    native_calls.append(
                        {
                            "id": event.id,
                            "name": event.name,
                            "arguments": event.arguments,
                        }
                    )
                elif isinstance(event, UsageEvent):
                    yield TurnUsage(event.input_tokens, event.output_tokens)
                elif isinstance(event, FinishEvent):
                    finish = event
                    reasoning_items.extend(
                        event.provider_metadata.get("reasoning_items", [])
                    )

            full_text = "".join(text_parts)
            reasoning_text = "".join(reasoning_parts)
            prepared = self.prepare_response(
                full_text,
                native_tool_calling=native_tool_calling,
                native_calls=native_calls,
            )

            if not prepared.tool_calls and not prepared.content:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": reasoning_text or "(no output)",
                    }
                )
                if empty_retries < max_empty_retries:
                    empty_retries += 1
                    yield TurnRetrying("empty_response", empty_retries)
                    continue
                yield TurnCompleted("", reasoning_text)
                return

            if (
                finish is not None
                and finish.truncated
                and not prepared.tool_calls
                and continuations < max_continuations
            ):
                conversation.append({"role": "assistant", "content": full_text})
                continuations += 1
                yield TurnRetrying("output_truncated", continuations)
                continue

            if not prepared.tool_calls:
                conversation.append({"role": "assistant", "content": full_text})
                yield TurnCompleted(prepared.content, reasoning_text)
                return

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": full_text,
            }
            if native_tool_calling:
                assistant_message.update(
                    self.assistant_metadata(prepared, reasoning_items)
                )
            conversation.append(assistant_message)

            for call in prepared.tool_calls:
                yield ToolStarted(call)
                result = await self.execute_tool(call)
                result_json = json.dumps(result, default=str)
                if call.id:
                    conversation.append(
                        {
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": call.id,
                            "name": call.name,
                        }
                    )
                else:
                    conversation.append(
                        {"role": "user", "content": f"tool_result({result_json})"}
                    )
                yield ToolCompleted(call, result)

    @staticmethod
    def close_dangling_tool_calls(
        conversation: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        answered = {
            message.get("tool_call_id")
            for message in conversation
            if isinstance(message, dict) and message.get("role") == "tool"
        }
        rebuilt: list[dict[str, Any]] = []
        changed = False
        for message in conversation:
            rebuilt.append(message)
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                if not call_id or call_id in answered:
                    continue
                rebuilt.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {"error": "Tool run was interrupted before it completed."}
                        ),
                        "tool_call_id": call_id,
                        "name": str(call.get("name") or ""),
                    }
                )
                changed = True
        return rebuilt, changed
