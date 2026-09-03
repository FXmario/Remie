"""Construction of Remie's system prompt and project context."""

from pathlib import Path

from remie.tools import (
    TOOL_REGISTRY,
    find_memory_by_id,
    get_active_memory_id,
    get_tool_str_representation,
    memory_file_path,
)

PROJECT_CONTEXT_MAX_CHARS = 8000

MEMORY_MAX_CHARS = 4000

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks.
You have access to a series of tools you can execute. Here are the tools you can execute:

{tool_list_repr}

When you want to use a tool, first provide a short 'thinking:' line explaining your reasoning, then reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.

When multiple valid approaches have meaningful tradeoffs or require a user preference, do not choose silently. Briefly explain the options and ask the user which they prefer. Continue autonomously for routine implementation details or when one option clearly dominates. Do not ask unnecessary confirmation questions.
To ask the user a question, call the 'ask_user' tool and wait for its result instead of ending your turn.

Use the 'memory' tool to persist durable facts, decisions, user preferences, and open tasks that should be remembered across chats. Add a note when you learn something that will matter later; do not log routine progress and do not use memory as a chat transcript. Remie keeps an active project memory, so use memory(action="add", text=...) without a name to append to it. Use memory(action="list") to see older memories (each with an id and a name), and target one by name or id only when needed; memory(action="delete", name=...) removes a memory entirely.
"""


def load_project_context() -> str:
    """
    Load project instructions from AGENTS.md in the launch directory.
    Returns an empty string when there is no AGENTS.md.
    """
    agents_file = Path.cwd() / "AGENTS.md"
    if not agents_file.is_file():
        return ""
    try:
        content = agents_file.read_text(encoding="utf-8")
    except OSError, UnicodeError:
        return ""
    if len(content) > PROJECT_CONTEXT_MAX_CHARS:
        content = (
            content[:PROJECT_CONTEXT_MAX_CHARS].rstrip()
            + "\n\n(AGENTS.md truncated for context.)\n"
        )
    return f"\n\n## Project instructions (from AGENTS.md)\n{content}"


def load_agent_memory() -> str:
    """
    Load the agent's active memory notes from .remie/memory/<uuid>.md in the
    launch directory. Returns an empty string when there is no active memory.
    """
    memory_id = get_active_memory_id()
    if not memory_id:
        return ""
    memory = find_memory_by_id(memory_id)
    if memory is None:
        return ""
    memory_file = memory_file_path(memory_id)
    if not memory_file.is_file():
        return ""
    try:
        content = memory_file.read_text(encoding="utf-8")
    except OSError, UnicodeError:
        return ""
    if not content.strip():
        return ""
    if len(content) > MEMORY_MAX_CHARS:
        content = (
            content[:MEMORY_MAX_CHARS].rstrip()
            + "\n\n(Memory truncated for context.)\n"
        )
    return f'\n\n## Agent memory (from .remie/memory: "{memory["name"]}")\n{content}'


def build_system_prompt(
    native_tools: bool = False, tab_context: dict[str, object] | None = None
) -> str:
    """Build the system prompt.

    With ``native_tools`` (Codex provider) the text-protocol instructions are
    replaced by a note that tools are called natively through the API, and the
    textual tool list is dropped since schemas travel with the request.
    """
    if native_tools:
        protocol = (
            "You have access to the function tools provided with each request. "
            "Call them natively instead of describing tool usage in plain text; "
            "results arrive as tool outputs between your turns.\n"
        )
        tool_list_repr = ""
    else:
        tool_list_repr = ""
        for tool_name in TOOL_REGISTRY:
            tool_list_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
            tool_list_repr += f"\n{'=' * 15}\n"
        protocol = (
            "When you want to use a tool, first provide a short 'thinking:' line "
            "explaining your reasoning, then reply with exactly one line in the "
            "format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.\n"
            "Use compact single-line JSON with double quotes. After receiving a "
            "tool_result(...) message, continue the task.\n"
            "If no tool is needed, respond normally.\n"
        )
    tabs = ""
    if tab_context:
        tabs = (
            "\n\n## Remie tabs\n"
            f"Working directory: {tab_context.get('working_directory', Path.cwd())}\n"
            f"Open tabs in this directory: {tab_context.get('tab_count', 1)}\n"
            f"Current tab: {tab_context.get('active_index', 1)} of "
            f"{tab_context.get('tab_count', 1)}\n"
            f"Current tab title: {tab_context.get('active_title', '')}\n"
            "All tabs belong to this working directory and have independent chat "
            "histories. Tabs do not change the working directory."
        )
    return (
        _compose_system_prompt(tool_list_repr, protocol)
        + load_project_context()
        + load_agent_memory()
        + tabs
    )


# Historical name retained for compatibility.
get_full_system_prompt = build_system_prompt


_ASK_USER_PARAGRAPH = (
    "When multiple valid approaches have meaningful tradeoffs or require a user "
    "preference, do not choose silently. Briefly explain the options and ask the "
    "user which they prefer. Continue autonomously for routine implementation "
    "details or when one option clearly dominates. Do not ask unnecessary "
    "confirmation questions.\nTo ask the user a question, call the 'ask_user' "
    "tool and wait for its result instead of ending your turn."
)

_MEMORY_PARAGRAPH = (
    "Use the 'memory' tool to persist durable facts, decisions, user preferences, "
    "and open tasks that should be remembered across chats. Add a note when you "
    "learn something that will matter later; do not log routine progress and do "
    "not use memory as a chat transcript. Remie keeps an active project memory, "
    'so use memory(action="add", text=...) without a name to append to it. Use '
    'memory(action="list") to see older memories (each with an id and a name), '
    'and target one by name or id only when needed; memory(action="delete", '
    "name=...) removes a memory entirely."
)


def _compose_system_prompt(tool_list_repr: str, protocol: str) -> str:
    return (
        f"You are a coding assistant whose goal it is to help us solve coding tasks. \n"
        f"You have access to a series of tools you can execute. Here are the tools you can execute:\n\n"
        f"{tool_list_repr}\n"
        f"{protocol}\n"
        f"{_ASK_USER_PARAGRAPH}\n"
        f"{_MEMORY_PARAGRAPH}"
    )
