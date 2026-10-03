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


## Upgrading from a version with persistent agent notes

The persistent memory feature has been removed. The `memory` tool,
`/memories` command, and Memories management tab are no longer available, and
saved notes are no longer injected into the system prompt. Reinstall the global
tool as described above and restart running Remie processes to use the new code.

Chat history, tab layouts, connection settings, and `AGENTS.md` project
instructions remain supported. Context compaction still summarizes older
messages within the current chat; it does not create a separate durable note
collection.

Existing note files are ignored by the new version. Upgrading alone does not
erase them. If you want to remove old notes, delete only `memory/`, `memory.md`,
and `active_memory` inside the relevant Remie state directory:

- `~/.remie/projects/<project-id>/` (or `$REMIE_HOME/projects/<project-id>/`);
- the legacy project-local `.remie/` directory, if it still exists;
- the legacy `~/.remie/` root, if it contains those entries.

Do not delete the entire state directory: its `chats/` and tab-layout files
contain unrelated saved history and workspace settings.
