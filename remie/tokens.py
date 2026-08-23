"""Provider-independent token estimation helpers."""

from typing import Any


def estimate_tokens_from_counts(chars: int, newlines: int) -> int:
    """
    Rough token estimate from pre-computed character and newline counts, used
    when the API does not report exact usage. Based on the ~4 chars/token
    heuristic, with newlines counted separately because streams can cheaply
    track these counters without re-scanning the accumulated text.
    """
    if chars == 0:
        return 0
    return max(1, chars // 4 + newlines // 3)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate for a piece of text, used when the API does not
    report exact usage. Based on the ~4 chars/token heuristic.
    """
    if not text:
        return 0
    return estimate_tokens_from_counts(len(text), text.count("\n"))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """
    Rough token estimate for a single conversation message, summing string
    content (including tool_result(...) messages and multimodal parts).
    """
    content = message.get("content")
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                total += estimate_tokens(part["text"])
            elif hasattr(part, "text") and part.text:
                total += estimate_tokens(part.text)
        return total
    return 0


def estimate_conversation_tokens(
    conversation: list[dict[str, Any]],
) -> int:
    """
    Rough token estimate for the whole conversation, summing string content
    (including tool_result(...) messages).
    """
    return sum(estimate_message_tokens(message) for message in conversation)
