"""Provider interface consumed by the agent runner."""

from collections.abc import AsyncIterator
from typing import Any, Protocol

from remie.providers.events import ProviderEvent


class Provider(Protocol):
    """A model backend that exposes one provider-neutral event stream."""

    async def stream(
        self, conversation: list[dict[str, Any]]
    ) -> AsyncIterator[ProviderEvent]: ...
