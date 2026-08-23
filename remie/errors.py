"""Shared exceptions raised by model providers."""


class UnsupportedModelError(RuntimeError):
    """Raised when a configured provider needs an unsupported API format."""


class LLMRequestError(RuntimeError):
    """Raised when the LLM server responds with a non-2xx status."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)
