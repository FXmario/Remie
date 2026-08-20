# Remie

A terminal-based coding agent with a [Textual](https://textual.textualize.io/) TUI. It chats with a local LLM (OpenAI-compatible endpoint) and can call tools to read, list, and edit files on your project.

## Features

- Textual TUI with bordered panels per message (user and assistant) and color-coded tool activity
- Streaming responses — assistant text appears token-by-token inside the log as the model generates it
- Message queue — keep typing while the agent works; queued messages are processed one at a time
- Token counter — running input/output totals plus a live token speed (`tok/s`) meter in the model badge
- Auto-continuation — when a response hits the output token limit (e.g. long reasoning), the agent resumes automatically instead of stopping
- Context compaction — when a long task nears the model's context window, older messages are trimmed automatically so the agent keeps working
- Reasoning display — the agent's thinking process streams in a `Reasoning` panel every turn (native `reasoning_content` when the provider streams it, `thinking:` lines otherwise)
- Syntax highlighting for code blocks the agent generates (fenced code in responses)
- The model name is shown on the top-right of the input box border
- Uses your terminal's native color scheme (ANSI), so the UI matches your system theme — like opencode
- Async agent loop — the UI stays responsive while the LLM responds and tools run
- Tools: `read_file`, `list_files`, `glob_files`, `tree_files`, `edit_file`, `run_command`, `ask_user`, `memory`
- **Agent memory** — every launch automatically creates and activates a blank memory named `session N`, while retaining older named memories under `.remie/memory/`. The agent can append durable facts, decisions, and open tasks to the active memory with the `memory` tool; press `Ctrl+O` to switch to or delete an older memory. When a long session nears the context window, dropped messages are summarized into a compact memory note instead of being silently truncated.
- **Fresh launches** — every launch starts a new conversation and discards the previously saved session; `Ctrl+L` clears the current conversation without creating another memory
- **Command safety** — `run_command` blocks destructive commands before they execute (`rm -rf /`, `rm -rf ~`, disk formatting/partitioning, shutdown/reboot, `chmod -R`/`chown -R` on `/` or `~`, fork bombs, `curl | sh`, `dd` to raw block devices, ...) and shows a `Blocked command` line in the log with the reason
- `REMIE_BLOCKED_COMMANDS` — comma-separated extra substrings (e.g. `git push --force`) that are always blocked, case-insensitive
- `ask_user` — when the agent needs a decision, it pops a modal with predefined choices and a free-text answer field instead of guessing
- Tool calls accepted as `tool: name({...})`, `<tool: name(...)>`, or DSML markup (`<|DSML|>invoke name="..." />`)
- Multiline input — `Shift+Enter` or `Ctrl+J` for a new line, `Enter` to send
- Prompt history — `Up`/`Down` arrows recall previous prompts, like a shell
- Paste images from the clipboard with `Ctrl+V` and send them to vision-capable models
- Thinking step before every tool call; tool calls are announced as `Agent calling <tool>`, kept in the conversation history
- Tool results are shown as readable `Tool result` panels in the log after each call — file contents (`read_file`) are syntax-highlighted by file extension, and `run_command` output is smart-highlighted when it looks like JSON, a unified diff, or a Python traceback (raw JSON still available with `REMIE_DEBUG=1`)

## Requirements

- Python >= 3.14
- An OpenAI-compatible local LLM server (e.g. llama.cpp server)
- Optional: [Codex CLI](https://github.com/openai/codex) for ChatGPT subscription access

## Setup

Install dependencies:

```bash
uv sync
```

To install the agent as a global command and use it from any project:

```bash
uv tool install /home/fuica/Work/agents/Remie --force
cd ~/Projects/my-other-project
remie
```

The agent operates on the directory where `remie` is launched, so relative tool paths target the current project.

### Project context

If the current project has an `AGENTS.md` at its root, its contents are added to the
agent's system prompt so it follows the project's conventions automatically. No
`AGENTS.md` is required — the agent otherwise explores the project with its tools.

Configure the LLM connection via environment variables (a `.env` file is loaded automatically):

| Variable          | Description                        | Default       |
| ----------------- | ---------------------------------- | ------------- |
| `LLAMA_BASE_URL`  | Base URL of the local LLM server   | `http://localhost:7070/v1` |
| `LLAMA_API_KEY`   | API key for the server             | `llama-cpp`   |
| `LLAMA_MODEL`     | Model name                         | `local-model` |
| `OPENAI_API_KEY`  | OpenAI API key; API billing is separate from ChatGPT subscriptions | (unset) |
| `OPENAI_MODEL`    | Initial OpenAI model selection     | `gpt-4o-mini` |
| `REMIE_DEBUG`            | Show raw tool calls (name + params) | (unset)       |
| `REMIE_REASONING_EFFORT` | Reasoning mode: off/low/medium/high/max | `medium`   |
| `REMIE_MAX_OUTPUT_TOKENS` | Max output tokens per response | OpenCode Go `32768`, local `8192` |
| `REMIE_MAX_AUTO_CONTINUATIONS` | Max silent auto-continuations per response | `10` |
| `REMIE_BLOCKED_COMMANDS` | Comma-separated extra command substrings that are always blocked (e.g. `git push --force,aws s3 rm`) | (unset) |

### Codex CLI

Remie can use the Codex CLI app-server with an existing ChatGPT subscription.
Install and authenticate Codex separately:

```bash
npm install -g @openai/codex
codex login
```

Choose **Codex CLI** in the connection picker. Remie starts `codex app-server`
on demand and keeps its JSON-RPC session alive between turns. Remie does not
store or read Codex authentication tokens. Codex runs read-only with approvals
disabled while Remie continues to own and execute its existing tools.

### OpenCode Go

You can also connect to [OpenCode Go](https://opencode.ai/docs/go/) — a low-cost
subscription service for open coding models. It uses the OpenAI-compatible
endpoint `https://opencode.ai/zen/go/v1` (chat completions), so it works with
the same client.

Open the connection picker with `Ctrl+P` or by clicking the model name next to
the input. From there you can:

- Choose **Local (llama.cpp)** to use your environment-configured server
- Choose **OpenAI API**, enter an OpenAI API key, and pick from the models returned by OpenAI
- Choose **OpenCode Go**, paste your API key (from [opencode.ai/auth](https://opencode.ai/auth)), and pick a model — the model list is fetched live, falling back to a bundled list
- Choose **Codex CLI** to use the locally installed and authenticated Codex CLI without an API key
- Choose a reasoning effort (`off`, `low`, `medium`, `high`, or `max`) supported by your provider

The connection form shows provider-specific fields after you choose a provider.
Remote providers offer live model dropdowns; local llama.cpp connections use a
manually entered model name.

Remie remembers each provider's last-used values. Reopening the connection picker
preselects the active provider, and switching providers restores that provider's
URL, API key, model, reasoning setting, and local SSL preference.

The local llama.cpp connection uses the official OpenAI Python SDK pointed at the
configured OpenAI-compatible local endpoint. OpenAI API and OpenCode Go continue
to use the existing HTTP streaming client.

OpenAI API access uses an API key and is billed separately from ChatGPT Plus or Pro
subscriptions. Remie does not use ChatGPT subscription credentials as API keys.

For local llama.cpp connections, the connection picker includes **Verify local SSL
certificates**. Turn it off only when the local server uses a self-signed certificate.
OpenAI and OpenCode Go always verify remote TLS certificates.

The Local provider exposes an editable Base URL field, defaulting to
`http://localhost:7070/v1`. Managed providers use their built-in endpoint.

The active connection (provider, base URL, API key, model, and reasoning effort) is saved to
`~/.config/remie/config.json` and reused on the next launch.

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
| `Ctrl+C`  | Copy selected text, or quit if nothing is selected |
| `Ctrl+L`  | Clear log   |
| `Ctrl+P`  | Open connection/model picker |
| `Ctrl+O`  | Open memory picker (switch/delete active memory) |
| `Ctrl+T`  | Toggle dark/light theme |
| `Esc`     | Stop the agent while it is running |
| `Enter`   | Send the message |
| `Shift+Enter` / `Ctrl+J` | Insert a new line |
| `Ctrl+V`  | Paste clipboard text, or attach an image |
| `Up` / `Down` | Recall previous prompts (history) |

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
