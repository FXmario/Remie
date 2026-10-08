# Updating Remie

Since `remie` is installed as a **uv tool** (a snapshot installed into uv's tool directory), pulling the latest source code alone won't update it — you need to reinstall the tool after getting the new code.

## 1. Update the source code

The project is under git, so pull the latest changes:

```bash
cd /home/fuica/Work/agents/Remie
git pull
```

## 2. Sync dependencies

In case `pyproject.toml` or `uv.lock` changed:

```bash
uv sync
```

## 3. Reinstall the global tool

Because the tool was installed from a **path** (not a registry package), the `remie` command in your PATH runs from uv's tool directory, not from the repo. Reinstalling is required to pick up new code:

```bash
uv tool install /home/fuica/Work/agents/Remie --force
```

The `--force` flag is important, especially since `pyproject.toml` keeps version `0.1.0`. If the version number never bumps, `uv tool upgrade remie` would consider the tool "up to date" and skip the reinstall.

Alternatively, use the upgrade form with `--force` for the same reason:

```bash
uv tool upgrade remie --force
```

## Quick recap

```bash
cd /home/fuica/Work/agents/Remie
git pull && uv sync
uv tool install /home/fuica/Work/agents/Remie --force
```

After that, typing `remie` in any directory will run the updated version.

## Tip for active development

If you're actively developing this project, skip the tool-install step entirely and just launch it from the repo with:

```bash
uv run main.py
```

That always uses the current code. Use the global `remie` install only when you want a stable snapshot available everywhere.


## Upgrading to scoped memory and history search

Remie now adds scoped memory using **Plan A**: existing JSON chats remain the
source of truth, with a disposable SQLite FTS5 search index alongside them.
No chat-database migration is required. New features:

- Global user preferences, shared project memory, and private per-tab Markdown notes.
- User approval for every model-requested memory write.
- Tab-only history search by default, with explicitly requested project-wide recall.
- `/memory reload` to refresh the active tab's frozen prompt snapshot.

Reinstall the global tool as described above and restart running Remie processes.
See [Memory and history search](README.md#memory-and-history-search) for storage,
limits, usage, and recovery. Chat history, tabs, provider settings, `AGENTS.md`,
and context compaction continue to work.

Older unscoped note collections (`memory/`, `memory.md`, `active_memory` in
project state directories) are not imported or erased automatically. Review and
copy useful facts into the new scoped files if desired. Do not delete the entire
Remie state directory, which also contains unrelated chats and settings.
