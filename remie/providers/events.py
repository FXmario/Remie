"""Provider-neutral events emitted while a model response is streaming."""

from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class UsageEvent:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FinishEvent:
    reason: str | None
    truncated: bool = False
    complete: bool = True
    provider_metadata: dict[str, Any] = field(default_factory=dict)


ProviderEvent: TypeAlias = (
    TextDelta | ReasoningDelta | ToolCallEvent | UsageEvent | FinishEvent
)
