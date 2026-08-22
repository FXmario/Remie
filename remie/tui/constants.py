"""Module-level constants for the Remie TUI."""

import os

from remie.agent import (
    CODEX_BACKEND_BASE,
    OPENCODE_GO_BASE_URL,
    OPENROUTER_BASE_URL,
)

REASONING_EFFORTS = ("off", "low", "medium", "high", "max")
PROMPT_HISTORY_LIMIT = 100
MAX_AUTO_CONTINUATIONS = int(
    os.environ.get("REMIE_MAX_AUTO_CONTINUATIONS", "10")
)
MAX_EMPTY_RESPONSE_RETRIES = int(
    os.environ.get("REMIE_MAX_EMPTY_RESPONSE_RETRIES", "2")
)
COMPACTION_CONTEXT_RATIO = 0.8
COMPACTION_KEEP_MESSAGES = 10

# The streaming preview re-renders the accumulated Markdown, which is
# expensive (parse + Pygments + layout). Two mitigations keep the UI
# responsive: throttling with an interval that grows with the text size, and
# rendering only a bounded tail window of the text so the per-update cost stays
# roughly constant no matter how long the generated answer gets.
#
# Because the per-update render cost is bounded by the preview window, the
# throttle interval is clamped to that same window: it never grows beyond the
# minimum once the text is long enough that the preview is capped, so long
# answers keep streaming at a steady rate instead of slowing down over time.
STREAM_UPDATE_MIN_INTERVAL = 0.1
STREAM_UPDATE_CHARS_PER_SECOND = 50_000
STREAM_PREVIEW_MAX_CHARS = 3000
PROVIDER_BASE_URLS = {
    "opencode-go": OPENCODE_GO_BASE_URL,
    "codex": CODEX_BACKEND_BASE,
    "openrouter": OPENROUTER_BASE_URL,
}

