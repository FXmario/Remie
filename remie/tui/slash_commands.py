"""Registry and parsing helpers for local slash commands."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    """A command handled by the TUI instead of being sent to the model."""

    name: str
    description: str

    @property
    def trigger(self) -> str:
        return f"/{self.name}"


SLASH_COMMANDS = (
    SlashCommand("memories", "Open the memory picker"),
    SlashCommand("chats", "Open saved chats"),
    SlashCommand("connect", "Configure a provider connection"),
    SlashCommand("models", "Switch the active model"),
    SlashCommand("change dir", "Change this tab's working directory"),
)

_COMMANDS_BY_NAME = {command.name: command for command in SLASH_COMMANDS}


def slash_command_matches(text: str) -> tuple[SlashCommand, ...]:
    """Return commands matching a single slash-prefixed token.

    Multiline text and text containing arguments are ordinary prompts and do
    not display command completions.
    """
    if not text.startswith("/") or "\n" in text or "\r" in text:
        return ()
    if any(character.isspace() for character in text) and not "/change dir".startswith(text.casefold()):
        return ()
    query = text[1:].casefold()
    if query.endswith("/"):
        query = query[:-1]
    return tuple(
        command for command in SLASH_COMMANDS if command.name.startswith(query)
    )


def resolve_slash_command(text: str) -> SlashCommand | None:
    """Resolve an exact command, accepting one optional trailing slash."""
    value = text.strip().casefold()
    if not value.startswith("/") or "\n" in value or "\r" in value:
        return None
    if any(character.isspace() for character in value) and value != "/change dir":
        return None
    name = value[1:]
    if name.endswith("/"):
        name = name[:-1]
    return _COMMANDS_BY_NAME.get(name)


def is_slash_command_token(text: str) -> bool:
    """Return whether text is a single slash token (known or unknown)."""
    value = text.strip()
    return (
        value.startswith("/")
        and len(value) > 1
        and not any(character.isspace() for character in value)
    )
