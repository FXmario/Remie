"""Parser for the text-based tool-call protocol used by compatible providers."""

import ast
import json
import re
from typing import Any


def extract_thinking(text: str) -> str:
    """
    Return the content of all 'thinking:' lines joined together.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("thinking:"):
            continue
        lines.append(line[len("thinking:") :].strip())
    return "\n".join(lines)


def _normalize_tool_line(line: str) -> str | None:
    """
    Normalize a tool invocation line to 'tool: name(args)' form, or None if the
    line is not a tool call. Handles plain, angle-bracket-wrapped, and
    self-closing variants like '<tool: name(args)>' and '<tool: name(args) />'.
    """
    if line.startswith("</tool") or line.startswith("tool>"):
        return None
    if line.startswith("<tool:"):
        inner = line[1:]
        if inner.endswith(">"):
            inner = inner[:-1]
        inner = inner.rstrip()
        if inner.endswith("/"):
            inner = inner[:-1].rstrip()
        return inner
    if line.startswith("tool:"):
        return line
    return None


def extract_tool_invocations(text: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Return list of (tool_name, args) requested in 'tool: name({...})' lines.
    Supports plain and angle-bracket-wrapped forms, compact JSON, and
    Python-style keyword arguments. Dashed names are normalized to underscores.
    """
    invocations = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = _normalize_tool_line(line)
        if normalized is None:
            continue
        try:
            after = normalized[len("tool:") :].strip()
            name, rest = after.split("(", 1)
            name = name.strip().replace("-", "_")
            if not rest.endswith(")"):
                continue
            args_text = rest[:-1].strip()
            try:
                args = json.loads(args_text)
            except json.JSONDecodeError:
                call = ast.parse(f"tool({args_text})", mode="eval").body
                if not isinstance(call, ast.Call):
                    continue
                if call.args:
                    args = ast.literal_eval(call.args[0])
                else:
                    args = {
                        keyword.arg: ast.literal_eval(keyword.value)
                        for keyword in call.keywords
                        if keyword.arg is not None
                    }
            if not isinstance(args, dict):
                continue
            invocations.append((name, args))
        except SyntaxError, ValueError, TypeError, json.JSONDecodeError:
            continue
    invocations.extend(extract_dsml_invocations(text))
    return invocations


def extract_dsml_invocations(text: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Parse DSML tool-call markup like:

        <|DSML|>invoke name="list-files">
        <|DSML|>parameter path="." />

    Returns a list of (tool_name, args) with dashed names normalized to
    underscores (list-files -> list_files).
    """
    invocations: list[tuple[str, dict[str, Any]]] = []
    current_name: str | None = None
    current_args: dict[str, Any] = {}

    def flush() -> None:
        nonlocal current_name, current_args
        if current_name is not None:
            invocations.append((current_name, current_args))
        current_name = None
        current_args = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("<|DSML|>"):
            continue
        body = line[len("<|DSML|>") :].strip()
        if body.startswith("invoke name="):
            flush()
            match = re.match(r'invoke name="([^"]+)"', body)
            if not match:
                continue
            current_name = match.group(1).replace("-", "_")
            current_args = {}
        elif body.startswith("parameter") and current_name is not None:
            param_text = body[len("parameter") :].strip()
            param_text = re.sub(r"/?>\s*$", "", param_text).strip()
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", param_text)
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            try:
                parsed = ast.literal_eval(value)
            except ValueError, SyntaxError:
                parsed = value.strip('"').strip("'")
            current_args[key] = parsed
        elif body.startswith("/tool_calls") or body.startswith("tool_calls"):
            flush()
    flush()
    return invocations


def strip_protocol_lines(text: str) -> str:
    """
    Remove 'thinking:', 'tool:', angle-wrapped tool, and DSML protocol lines for
    display. They stay in the conversation history sent to the model.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            line.startswith("thinking:")
            or line.startswith("tool:")
            or line.startswith("<tool:")
            or line.startswith("</tool")
            or line.startswith("<|DSML|>")
        ):
            continue
        lines.append(raw_line)
    return "\n".join(lines)
