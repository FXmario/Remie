"""Events emitted by the headless agent runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from remie.core.runner import PendingToolCall


@dataclass(frozen=True)
class TurnTextDelta:
    text: str


@dataclass(frozen=True)
class TurnReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    call: PendingToolCall


@dataclass(frozen=True)
class ToolCompleted:
    call: PendingToolCall
    result: dict[str, Any]


@dataclass(frozen=True)
class TurnRetrying:
    reason: str
    attempt: int


@dataclass(frozen=True)
class TurnUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TurnCompleted:
    text: str
    reasoning: str


AgentEvent: TypeAlias = (
    TurnTextDelta
    | TurnReasoningDelta
    | ToolStarted
    | ToolCompleted
    | TurnRetrying
    | TurnUsage
    | TurnCompleted
)
