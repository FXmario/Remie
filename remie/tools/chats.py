"""Backward-compatible chat storage imports.

Chat persistence is application storage rather than a model-callable tool. New
code should import :mod:`remie.storage.chats` directly.
"""

from remie.storage.chats import *  # noqa: F403
