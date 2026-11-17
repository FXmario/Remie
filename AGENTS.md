# Project instructions for Remie

## Overview

Remie is a Python coding agent with a Textual terminal UI, multiple model
providers, model-callable tools, and independent chat tabs. Python 3.14+ is
required. Read `README.md` for user-facing behavior and `docs/ARCHITECTURE.md`
for request flow, extension points, and persistence contracts.

## Development commands

- Install/sync dependencies: `uv sync`
- Launch the application: `uv run remie` or `uv run main.py`
- Run tests: `uv run pytest`
- Run a focused test file: `uv run pytest tests/test_core.py`

Use the `run_test_shards` tool for discoverable or splittable test suites when
available; use focused commands for targeted checks. Tests live in `tests/`
and use pytest. Prefer isolated temporary directories and mocked provider
responses over real credentials, network requests, or user configuration.

## Code organization

- `remie/core/`: headless, provider-independent agent loop and orchestration.
- `remie/providers/`: provider adapters, routing, stream events, and catalogs.
- `remie/tools/`: tool schemas, execution, command safety, and sandboxing.
- `remie/tui/`: Textual application, widgets, rendering, and modal screens.
- `remie/storage/`: project chat persistence.
- `remie/config.py`: connection profiles and configuration storage.
- `remie/prompts.py`: system prompts and automatic `AGENTS.md` loading.
- `remie/protocol.py`: text-based tool-call parsing.
- `remie/agent.py`: compatibility facade and active connection runtime.
- `remie/codex_client.py` and `remie/openrouter_client.py`: provider clients.

## Implementation conventions

- Follow the surrounding code's style, typing, and test patterns. Keep changes
  focused; avoid unrelated refactors or new dependencies without a clear need.
- Preserve the dependency direction: UI -> core -> provider/tool interfaces.
  Providers and tools must not import Textual. Keep headless execution usable
  without initializing the UI.
- Preserve public compatibility facades and lazy imports unless the task
  explicitly requires changing their contracts.
- Keep text-protocol and native function-calling providers working. Changes to
  tool calls must preserve paired tool results and interrupted-call recovery.
- Respect per-tab working directories and runtime isolation. Do not introduce
  process-global state that leaks one tab's context into another.
- Keep configuration and storage paths injectable for tests. Do not write to
  real user settings, chat history, or authentication files during tests.
- Preserve command safety checks and sandbox boundaries. Never silently retry
  a failed sandboxed command without sandboxing.
- Never commit API keys, OAuth tokens, local chat data, or other credentials.
  Avoid editing generated build output or vendored binaries for ordinary fixes.

## Verification and documentation

Add or update regression tests for behavior changes, including relevant error
paths. Run the affected tests first and broader coverage when practical; report
what ran and any failures or untested areas accurately. For documentation-only
changes, inspect the diff and verify referenced paths and commands.

Update `README.md` for user-visible behavior and `docs/ARCHITECTURE.md` when
architecture or extension contracts change. Keep this file concise: Remie loads
only the current working directory's root `AGENTS.md`, with an 8,000-character
context limit; it does not recursively discover nested instruction files.
