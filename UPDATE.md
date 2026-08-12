# Updating FuiAgent

Since `remie` is installed as a **uv tool** (a snapshot installed into uv's tool directory), pulling the latest source code alone won't update it — you need to reinstall the tool after getting the new code.

## 1. Update the source code

The project is under git, so pull the latest changes:

```bash
cd /home/mario/Work/FuiAgent
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
uv tool install /home/mario/Work/FuiAgent --force
```

The `--force` flag is important, especially since `pyproject.toml` keeps version `0.1.0`. If the version number never bumps, `uv tool upgrade fuiagent` would consider the tool "up to date" and skip the reinstall.

Alternatively, use the upgrade form with `--force` for the same reason:

```bash
uv tool upgrade fuiagent --force
```

## Quick recap

```bash
cd /home/mario/Work/FuiAgent
git pull && uv sync
uv tool install /home/mario/Work/FuiAgent --force
```

After that, typing `remie` in any directory will run the updated version.

## Tip for active development

If you're actively developing this project, skip the tool-install step entirely and just launch it from the repo with:

```bash
uv run main.py
```

That always uses the current code. Use the global `remie` install only when you want a stable snapshot available everywhere.
