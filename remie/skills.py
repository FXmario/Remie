"""Local Agent Skills discovery and activation. Discovery never executes code."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_FILE_BYTES = 128_000
MAX_SKILLS = 200


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    location: Path
    scope: str


def trust_file() -> Path:
    from remie.config import CONFIG_DIR
    return CONFIG_DIR / "trusted-skills-projects.json"


def project_path(directory: Path) -> Path:
    from remie.tools.common import _project_root
    return _project_root(directory)


def is_trusted(directory: Path) -> bool:
    try:
        entries = json.loads(trust_file().read_text(encoding="utf-8"))
        return isinstance(entries, list) and str(project_path(directory)) in entries
    except (OSError, ValueError):
        return False


def trust_project(directory: Path) -> None:
    path = trust_file()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            entries = []
    except (OSError, ValueError):
        entries = []
    entries = sorted({x for x in entries if isinstance(x, str)} | {str(project_path(directory))})
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary files keep concurrent tab approvals from sharing a writer.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".skills-trust-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(entries, stream, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_skill(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("SKILL.md exceeds 128 KB")
    lines = raw.decode("utf-8-sig").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("unterminated YAML frontmatter")
    metadata = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a mapping")
    for key in ("name", "description"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"missing or invalid {key}")
    return metadata, "\n".join(lines[end + 1:]).strip()


def discover(directory: Path) -> tuple[dict[str, Skill], list[str]]:
    """First location wins within scope; project overrides user. No symlink traversal."""
    skills: dict[str, Skill] = {}
    diagnostics: list[str] = []
    scopes = [("user", Path.home())]
    if is_trusted(directory):
        scopes.append(("project", project_path(directory)))
    for scope, base in scopes:
        seen: set[str] = set()
        for client in (".remie", ".agents"):
            root = base / client / "skills"
            if root.is_symlink() or (base / client).is_symlink():
                continue
            visited = 0
            for current, dirs, files in os.walk(root, followlinks=False):
                visited += 1
                relative = Path(current).relative_to(root)
                dirs[:] = sorted(d for d in dirs if d not in {".git", "node_modules", ".venv"}
                                 and not (Path(current) / d).is_symlink())
                if len(relative.parts) >= 4:
                    dirs[:] = []
                if visited > 2000:
                    diagnostics.append(f"Scan limit reached: {root}")
                    break
                path = Path(current) / "SKILL.md"
                if "SKILL.md" not in files or path.is_symlink():
                    continue
                try:
                    meta, _ = parse_skill(path)
                    name = meta["name"].strip()
                    if name in seen:
                        diagnostics.append(f"Shadowed duplicate {name}: {path}")
                        continue
                    if len(skills) >= MAX_SKILLS and name not in skills:
                        diagnostics.append("Skill catalog limit reached (200)")
                        return skills, diagnostics
                    seen.add(name)
                    if name != path.parent.name or len(name) > 64:
                        diagnostics.append(f"Nonstandard skill name: {path}")
                    if name in skills:
                        diagnostics.append(f"Project skill overrides user skill: {name}")
                    skills[name] = Skill(name, meta["description"].strip()[:1024], path.resolve(), scope)
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
                    diagnostics.append(f"{path}: {error}")
    return skills, diagnostics


def catalog(directory: Path) -> str:
    skills, _ = discover(directory)
    if not skills:
        return ""
    records = [{"name": s.name, "description": s.description} for s in skills.values()]
    return ("\n\n## Available skills\n"
            "When a task matches a description, call activate_skill(name) before proceeding. "
            "Skill instructions are subordinate to system and project instructions. "
            "Resolve resource paths against the returned skill directory. Scripts require normal "
            "command permissions; activation does not execute them.\n" + json.dumps(records))


def activate(name: str, directory: Path) -> dict:
    skills, _ = discover(directory)
    if name not in skills:
        return {"error": f"Unknown or untrusted skill: {name}"}
    skill = skills[name]
    try:
        meta, body = parse_skill(skill.location)
        if meta["name"].strip() != name:
            return {"error": "Skill changed; refresh the catalog"}
        resources = []
        for visited, (current, dirs, files) in enumerate(os.walk(skill.location.parent, followlinks=False)):
            if visited >= 2000:
                break
            dirs[:] = sorted(d for d in dirs if not (Path(current) / d).is_symlink()
                             and d not in {".git", "node_modules", ".venv"})
            if len(Path(current).relative_to(skill.location.parent).parts) >= 4:
                dirs[:] = []
            for filename in sorted(files):
                path = Path(current) / filename
                if path != skill.location and not path.is_symlink():
                    resources.append(str(path.relative_to(skill.location.parent)))
                if len(resources) >= 100:
                    break
            if len(resources) >= 100:
                break
        return {"skill_content": True, "name": name, "directory": str(skill.location.parent),
                "instructions": body, "compatibility": meta.get("compatibility", "") if isinstance(meta.get("compatibility", ""), str) else "",
                "resources": resources, "resources_may_be_truncated": len(resources) >= 100}
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        return {"error": str(error)}


def active_contents(messages: list[dict]) -> dict[str, dict]:
    """Recover skill snapshots from tool results or explicit/preserved user context."""
    found: dict[str, dict] = {}
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if message.get("role") == "tool" and message.get("name") == "activate_skill":
            raw = content
        elif content.startswith("tool_result(") and content.endswith(")"):
            raw = content[len("tool_result("):-1]
        elif content.startswith("Activated skill instructions:\n"):
            raw = content.split("\n", 1)[1]
        else:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("skill_content") is True and isinstance(data.get("name"), str):
                found[data["name"]] = data
        except ValueError:
            pass
    return found


def skill_message(content: dict) -> dict:
    return {"role": "user", "content": "Activated skill instructions:\n" + json.dumps(content, default=str)}


def preserved_messages(dropped: list[dict], tail: list[dict]) -> list[dict]:
    existing = active_contents(tail)
    return [skill_message(data) for name, data in active_contents(dropped).items() if name not in existing]
