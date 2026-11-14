"""Validated model-authored plans and tab-local approval/evidence state."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone
import uuid


def todo_tool(items: list[dict], confirm: bool = False) -> dict:
    """Replace this tab's plan. New/revised steps require user approval.

    Items need title and status (pending, in_progress, completed). Completed
    items also need non-empty evidence. Optionally cite a check_id returned by
    run_test_shards to link a recorded passing check; this does not prove the
    step's semantics. Pass [] to clear the plan and revoke approval.
    """
    if not isinstance(items, list) or not isinstance(confirm, bool):
        return {"error": "items must be a list and confirm must be a boolean"}
    clean = []
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("title"), str)
                or not item["title"].strip()
                or item.get("status") not in ("pending", "in_progress", "completed")):
            return {"error": "Each item needs a title and a valid status"}
        entry = {"title": item["title"].strip(), "status": item["status"]}
        for key in ("evidence", "check_id"):
            value = item.get(key)
            if value is not None:
                if not isinstance(value, str):
                    return {"error": f"{key} must be a string"}
                if value.strip():
                    entry[key] = value.strip()
        if entry["status"] == "completed" and not entry.get("evidence"):
            return {"error": "Completed steps require non-empty evidence"}
        clean.append(entry)
    return {"items": clean, "confirm": confirm}


@dataclass
class TodoState:
    """One tab's plan; approval and eligible checks are never restored from disk."""

    items: list[dict] = field(default_factory=list)
    expanded: bool = True
    user_override: bool = False
    approved: bool = False
    revision: int = 0
    checks: dict[str, dict] = field(default_factory=dict)

    def revoke(self) -> None:
        self.approved = False
        self.revision += 1
        self.checks.clear()

    def replace(self, items: list[dict]) -> dict:
        changed = [i["title"] for i in items] != [i["title"] for i in self.items]
        clean = deepcopy(items)
        for item in clean:
            if item["status"] != "completed":
                continue
            item["verification"] = "agent_reported"
            check_id = item.get("check_id")
            if not check_id:
                continue
            check = self.checks.get(check_id) if not changed else None
            old_matches = [old for old in self.items if all(old.get(k) == item.get(k)
                           for k in ("title", "status", "evidence", "check_id"))]
            old = old_matches[0] if len(old_matches) == 1 else {}
            # Keep historical evidence on unchanged completed steps, including
            # after a restart. It cannot be attached to a different step.
            historical = (old.get("verification") == "check_passed"
                          and all(old.get(k) == item.get(k) for k in
                                  ("title", "status", "evidence", "check_id")))
            if historical:
                check = old.get("check")
            if not check or not check.get("passed"):
                return {"error": "check_id must reference a recorded passing check for this tab and plan"}
            item["verification"] = "check_passed"
            item["check"] = deepcopy(check)
        if changed or not clean:
            self.revoke()
        self.items = clean
        return {"items": deepcopy(clean), "approved": self.approved, "revision": self.revision}

    def record_check(self, args: dict, result: dict) -> str:
        """Record execution, not semantic task correctness, using actual results."""
        outputs = result.get("shards") if result.get("mode") == "parallel" else [result.get("result")]
        passed = (result.get("status") == "passed" and isinstance(outputs, list)
                  and bool(outputs) and all(isinstance(r, dict) and r.get("exit_code") == 0
                                           and not r.get("timed_out") for r in outputs))
        check_id = uuid.uuid4().hex
        self.checks[check_id] = {
            "tool": "run_test_shards", "command": str(args.get("command") or "pytest"),
            "cwd": str(args.get("cwd") or "."), "passed": passed,
            "mode": str(result.get("mode") or "unknown"),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Bound session memory; linked evidence is copied into completed items.
        if len(self.checks) > 100:
            self.checks.pop(next(iter(self.checks)))
        return check_id

    def snapshot(self) -> dict:
        return {"items": deepcopy(self.items), "expanded": self.expanded,
                "user_override": self.user_override}

    @classmethod
    def restore(cls, payload: Any) -> "TodoState":
        state = cls()
        if not isinstance(payload, dict):
            return state
        parsed = todo_tool(payload.get("items", []))
        if "error" not in parsed:
            state.items = parsed["items"]
            for item, saved in zip(state.items, payload.get("items", [])):
                if item["status"] == "completed":
                    item["verification"] = "agent_reported"
                    check = saved.get("check")
                    if (item.get("check_id") and saved.get("verification") == "check_passed"
                            and isinstance(check, dict) and check.get("passed") is True
                            and check.get("tool") == "run_test_shards"):
                        item["verification"] = "check_passed"
                        item["check"] = deepcopy(check)
        if isinstance(payload.get("expanded"), bool):
            state.expanded = payload["expanded"]
        if isinstance(payload.get("user_override"), bool):
            state.user_override = payload["user_override"]
        return state
