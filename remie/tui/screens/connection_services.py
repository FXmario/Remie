"""Injected external operations used by the connection screen."""

from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from remie.config import ConnectionConfig
from remie.model_names import ModelInfo

ModelRows = list[str | ModelInfo]


@dataclass(frozen=True)
class ConnectionServices:
    """Provider I/O boundary, separated from Textual form rendering."""

    fetch_opencode_models: Callable[[str], Awaitable[ModelRows]]
    fetch_codex_models: Callable[[], Awaitable[ModelRows]]
    fetch_openrouter_models: Callable[[], Awaitable[ModelRows]]
    save_profiles: Callable[[dict[str, ConnectionConfig], str], None]
