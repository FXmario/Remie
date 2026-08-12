# FuiAgent

A terminal-based coding agent with a [Textual](https://textual.textualize.io/) TUI. It chats with a local LLM (OpenAI-compatible endpoint) and can call tools to read, list, and edit files on your project.

## Features

- Textual TUI with bordered panels per message (user and assistant) and color-coded tool activity
- Streaming responses — assistant text appears token-by-token inside the log as the model generates it
- Syntax highlighting for code blocks the agent generates (fenced code in responses)
- The model name is shown on the top-right of the input box border
- Uses your terminal's native color scheme (ANSI), so the UI matches your system theme — like opencode
- Async agent loop — the UI stays responsive while the LLM responds and tools run
- Tools: `read_file`, `list_files`, `glob_files`, `tree_files`, `edit_file`, `run_command`
- Thinking step before every tool call; tool calls are announced as `Agent calling <tool>`, kept in the conversation history

## Requirements

- Python >= 3.14
- An OpenAI-compatible local LLM server (e.g. llama.cpp server)

## Setup

Install dependencies:

```bash
uv sync
```

To install the agent as a global command and use it from any project:

```bash
uv tool install /your/project/location/FuiAgent
cd ~/Projects/my-other-project
remie
```

The agent operates on the directory where `remie` is launched, so relative tool paths target the current project.

Configure the LLM connection via environment variables (a `.env` file is loaded automatically):

| Variable          | Description                        | Default       |
| ----------------- | ---------------------------------- | ------------- |
| `LLAMA_BASE_URL`  | Base URL of the LLM server         | (required)    |
| `LLAMA_API_KEY`   | API key for the server             | `llama-cpp`   |
| `LLAMA_MODEL`     | Model name                         | `local-model` |
| `REMIE_DEBUG`   | Show raw tool calls (name + params) | (unset)       |

### OpenCode Go

You can also connect to [OpenCode Go](https://opencode.ai/docs/go/) — a low-cost
subscription service for open coding models. It uses the OpenAI-compatible
endpoint `https://opencode.ai/zen/go/v1` (chat completions), so it works with
the same client.

Open the connection picker with `Ctrl+P` or by clicking the model name next to
the input. From there you can:

- Choose **Local (llama.cpp)** to use your environment-configured server
- Choose **OpenCode Go**, paste your API key (from [opencode.ai/auth](https://opencode.ai/auth)), and pick a model — the model list is fetched live, falling back to a bundled list

The active connection (base URL, API key, model) is saved to
`~/.config/fuiagent/config.json` and reused on the next launch.

When the agent calls a tool, the log shows a human-readable line like `Agent calling the read a file`.
Tool results are hidden by default. Set `REMIE_DEBUG=1` to show the raw function name, JSON parameters,
and `tool_result` payloads (e.g. `Agent calling read_file({"filename": "main.py"})`); a `· debug` marker appears in the header.

## Usage

Run the agent with the `remie` command (installed by `uv sync`), or launch it directly:

```bash
remie
# or
uv run main.py
```

Type a message at the bottom input and press Enter. The agent will reason (`Thinking:`), call tools when needed, show the results, and reply — with the response streaming in as it is generated.

### Keybindings

| Key       | Action      |
| --------- | ----------- |
| `Ctrl+C`  | Quit        |
| `Ctrl+L`  | Clear log   |
| `Ctrl+P`  | Open connection/model picker |
| `Ctrl+T`  | Toggle dark/light theme |
| `Esc`     | Stop the agent while it is running |

## How it works

The system prompt instructs the model to emit a short `thinking:` line before a `tool:` line of the form:

```
tool: read_file({"filename": "main.py"})
```

`remie/agent.py` parses those lines, runs the requested tool, and feeds the result back as a `tool_result(...)` message until the model responds without a tool call.

## Project layout

- `main.py` — entry point; launches the TUI
- `remie/agent.py` — LLM client, system prompt, tool registry, and parsing helpers
- `remie/tui.py` — the Textual `AgentApp`, theme detection, and the `remie` CLI entry point
