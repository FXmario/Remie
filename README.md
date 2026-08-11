# FuiAgent

A terminal-based coding agent with a [Textual](https://textual.textualize.io/) TUI. It chats with a local LLM (OpenAI-compatible endpoint) and can call tools to read, list, and edit files on your project.

## Features

- Textual TUI with a color-coded conversation log (You / Thinking / Assistant / tool calls / tool results)
- Streaming responses — assistant text appears token-by-token inside the log as the model generates it
- Uses your terminal's native color scheme (ANSI), so the UI matches your system theme — like opencode
- Async agent loop — the UI stays responsive while the LLM responds and tools run
- Tools: `read_file`, `list_files`, `edit_file`
- Thinking step before every tool call; tool calls are announced as `Agent calling <tool>`, kept in the conversation history

## Requirements

- Python >= 3.14
- An OpenAI-compatible local LLM server (e.g. llama.cpp server)

## Setup

Install dependencies:

```bash
uv sync
```

Configure the LLM connection via environment variables (a `.env` file is loaded automatically):

| Variable          | Description                        | Default       |
| ----------------- | ---------------------------------- | ------------- |
| `LLAMA_BASE_URL`  | Base URL of the LLM server         | (required)    |
| `LLAMA_API_KEY`   | API key for the server             | `llama-cpp`   |
| `LLAMA_MODEL`     | Model name                         | `local-model` |

## Usage

Run the agent with the `chaldea` command (installed by `uv sync`), or launch it directly:

```bash
chaldea
# or
uv run main.py
```

Type a message at the bottom input and press Enter. The agent will reason (`Thinking:`), call tools when needed, show the results, and reply — with the response streaming in as it is generated.

### Keybindings

| Key       | Action      |
| --------- | ----------- |
| `Ctrl+C`  | Quit        |
| `Ctrl+L`  | Clear log   |
| `Ctrl+T`  | Toggle dark/light theme |

## How it works

The system prompt instructs the model to emit a short `thinking:` line before a `tool:` line of the form:

```
tool: read_file({"filename": "main.py"})
```

`chaldea/agent.py` parses those lines, runs the requested tool, and feeds the result back as a `tool_result(...)` message until the model responds without a tool call.

## Project layout

- `main.py` — entry point; launches the TUI
- `chaldea/agent.py` — LLM client, system prompt, tool registry, and parsing helpers
- `chaldea/tui.py` — the Textual `AgentApp`, theme detection, and the `chaldea` CLI entry point
