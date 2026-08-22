# Remie

A terminal-based coding agent with a [Textual](https://textual.textualize.io/) TUI. It chats with a local LLM (OpenAI-compatible endpoint) and can call tools to read, list, and edit files on your project.

The Python `remie` command remains available. The Zig preview accepts these commands:

```text
/connect local http://localhost:7070/v1 llama-cpp local-model
/connect opencode-go YOUR_API_KEY deepseek-v4-flash
/models
your normal chat prompt
```

The Local connector targets an OpenAI-compatible `/chat/completions` server. OpenCode
Go model discovery uses `https://opencode.ai/zen/go/v1/models` and saves the active
profile to the shared Remie configuration file.

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
- **Agent memory (durable notes)** — the agent can append durable facts, decisions, user preferences, and open tasks to the active memory with the `memory` tool; press `Ctrl+O` to switch to or delete an older memory. The active memory is remembered across launches and injected into the system prompt. When a long task nears the context window, dropped messages are summarized into a compact note instead of being silently truncated.
- **Chat history** — every conversation is saved as a named chat under `.remie/chats/`. Launching Remie resumes the most recently used chat automatically; `Ctrl+R` opens a chat picker to switch to an older chat, start a new one, or delete one. A chat is auto-titled after its first completed task; `Ctrl+L` starts a new chat while keeping the previous one. Existing `.remie/session.json` files from older versions are imported as a chat on first launch.
- **Codex (ChatGPT Plus/Pro)** — sign in with a ChatGPT subscription via the native OAuth flow and use Codex models (gpt-5.6-sol/terra/luna, gpt-5.5, …) without an API key, the Codex CLI, or npm
- **OpenRouter** — connect with an OpenRouter API key to any model in their catalog; native function calling over plain httpx streaming
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
- An OpenAI-compatible local LLM server (e.g. llama.cpp server), **or**
- An OpenCode Go or OpenRouter API key, **or**
- A ChatGPT Plus/Pro subscription for the Codex provider

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
| `REMIE_DEBUG`            | Show raw tool calls (name + params) | (unset)       |
| `REMIE_REASONING_EFFORT` | Reasoning mode: off/low/medium/high/max | `medium`   |
| `REMIE_MAX_OUTPUT_TOKENS` | Max output tokens per response | OpenCode Go `32768`, local `8192` |
| `REMIE_MAX_AUTO_CONTINUATIONS` | Max silent auto-continuations per response | `10` |
| `REMIE_BLOCKED_COMMANDS` | Comma-separated extra command substrings that are always blocked (e.g. `git push --force,aws s3 rm`) | (unset) |

### OpenCode Go

You can also connect to [OpenCode Go](https://opencode.ai/docs/go/) — a low-cost
subscription service for open coding models. It uses the OpenAI-compatible
endpoint `https://opencode.ai/zen/go/v1` (chat completions), so it works with
the same client.

Open the connection picker with `Ctrl+P` or by clicking the model name next to
the input. From there you can:

- Choose **Local (llama.cpp)** to use your environment-configured server
- Choose **OpenCode Go**, paste your API key (from [opencode.ai/auth](https://opencode.ai/auth)), and pick a model — the model list is fetched live, falling back to a bundled list
- Choose a reasoning effort (`off`, `low`, `medium`, `high`, or `max`) supported by your provider

The connection form shows provider-specific fields after you choose a provider.
OpenCode Go offers a live model dropdown; local llama.cpp connections use a
manually entered model name.

Remie remembers each provider's last-used values. Reopening the connection picker
preselects the active provider, and switching providers restores that provider's
URL, API key, model, reasoning setting, and local SSL preference.

The local llama.cpp connection uses the official OpenAI-compatible Python SDK.
OpenCode Go uses the existing HTTP streaming client.

For local llama.cpp connections, the connection picker includes **Verify local SSL
certificates**. Turn it off only when the local server uses a self-signed certificate.
OpenCode Go always verifies remote TLS certificates.

The Local provider exposes an editable Base URL field, defaulting to
`http://localhost:7070/v1`. Managed providers use their built-in endpoint.

The active connection (provider, base URL, API key, model, and reasoning effort) is saved to
`~/.config/remie/config.json` and reused on the next launch.

### Codex (ChatGPT Plus/Pro)

Remie can also run on a ChatGPT subscription through the same backend the Codex
CLI uses — no API key, no Node.js, no `codex` install. Open the connection
picker with `Ctrl+P`, choose **Codex (ChatGPT Plus/Pro)**, and press **Sign in
with ChatGPT**:

- Remie opens your browser at `auth.openai.com` (PKCE flow) and receives the
  callback on `http://localhost:1455/auth/callback`
- After sign-in the account row shows the signed-in address and plan
  (`you@example.com · Plus`), and **Sign out** deletes the stored tokens
- The model dropdown is fetched live for your account (falling back to a
  bundled list when offline); reasoning effort maps to the Responses API tiers,
  including `xhigh` for Remie's `max`
- Requests stream from `chatgpt.com/backend-api/codex/responses` through the
  official OpenAI SDK with automatic token refresh; expired sessions prompt you
  to sign in again
- Tool calling is native: Remie's tools travel as Responses-API function
  definitions and Codex models call them directly (no text protocol), with
  results replayed as `function_call_output` items between turns

Tokens are stored as `~/.codex/auth.json` in the Codex CLI's own format, so an
existing `codex login` is picked up automatically and signing in from either
tool keeps both authenticated. The file contains bearer credentials — treat it
like a secret.

Note for remote/headless machines: the OAuth callback targets `localhost:1455`,
so the browser must reach the machine running Remie (use
`ssh -L 1455:localhost:1455 user@host` when working over SSH).

### OpenRouter

Pick **OpenRouter** in the connection picker (`Ctrl+P`), paste an API key from
[openrouter.ai/keys](https://openrouter.ai/keys), and choose a model. The model
dropdown loads OpenRouter's live catalog (public endpoint — it works before a
key is entered) with each model's real context window driving context
compaction.

Like Codex, tool calling is native: tools are sent as chat-completions
function definitions and results replay as `tool` messages between turns — no
text protocol. Requests stream over plain httpx from
`openrouter.ai/api/v1/chat/completions`; reasoning effort maps to OpenRouter's
`reasoning.effort` parameter (`max` becomes `high`). Errors surface with
friendly copy, including OpenRouter's 402 "out of credits" state.

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

Press `Ctrl+G` to show or hide the status image beside the prompt. Outside
tmux it is animated; inside tmux it uses a static frame. The preference is
saved in Remie's configuration and restored on the next launch.

Type a message at the bottom input and press Enter. The agent will reason (`Thinking:`), call tools when needed, show the results, and reply — with the response streaming in as it is generated.

### Keybindings

| Key       | Action      |
| --------- | ----------- |
| `Ctrl+C`  | Copy selected text, or quit if nothing is selected |
| `Ctrl+L`  | Start a new chat (the previous one is kept in history) |
| `Ctrl+P`  | Open connection/model picker |
| `Ctrl+O`  | Open memory picker (switch/delete active memory) |
| `Ctrl+R`  | Open chat picker (switch/new/delete saved chats) |
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
- `remie/codex_auth.py` — ChatGPT OAuth (PKCE) sign-in, token storage in `~/.codex/auth.json`, and refresh
- `remie/codex_client.py` — streaming client for the ChatGPT-subscription Codex Responses backend
- `remie/openrouter_client.py` — httpx streaming client for OpenRouter with native function calling
- `remie/tui.py` — the Textual `AgentApp`, theme detection, and the `remie` CLI entry point
