"""Runtime-safe contracts shared by TUI widgets and screens."""

from typing import Any, TypeGuard


class AgentAppContract:
    """Typing marker for the concrete app without importing it circularly."""

    IS_REMIE_AGENT_APP: bool


def is_agent_app(app: Any) -> TypeGuard[AgentAppContract]:
    """Identify Remie's app without injecting its class into module globals."""
    return bool(getattr(app, "IS_REMIE_AGENT_APP", False))
