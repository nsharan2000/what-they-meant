"""Environment simulator: executes tool calls against a member's state + the scenario's tool rules. Deterministic, no LLM."""
from __future__ import annotations

import json

from .spec import all_hit, norm


def _args_text(args: dict) -> str:
    return " ".join(norm(v) for v in (args or {}).values())


class Simulator:
    def __init__(self, env: dict, member: dict, scenario: dict, allowed_tools: list[str] | None = None,
                 level: str = "L1"):
        self.env, self.member, self.s = env, member, scenario
        self.raw = level != "L0"          # every level except L0-seed uses the raw-field version of rule results
        self.tools = {t["name"]: t for t in env["tools"]}
        self.allowed = set(allowed_tools) if allowed_tools else set(self.tools)
        self.commits: list[dict] = []
        self.approvals: list[dict] = []
        self.calls: dict[str, int] = {}
        self._receipt = 0

    def kind(self, name: str) -> str:
        return self.tools[name]["kind"] if name in self.tools and name in self.allowed else "invalid"

    def execute(self, name: str, args: dict, approval_decider=None):
        if name not in self.tools or name not in self.allowed:
            return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(sorted(self.allowed))}"}
        tool = self.tools[name]
        args = args if isinstance(args, dict) else {}
        # 1) scenario rules (first match wins) — they encode the seed's tool results exactly.
        #    optional "nth": only the n-th call (1-based) of this tool matches; optional "effect": "none" on a commit tool =
        #    a failed attempt (error result, no money moved, not counted as a commit).
        text = _args_text(args)
        self.calls[name] = self.calls.get(name, 0) + 1
        for rule in self.s.get("tool_rules", []):
            if rule["tool"] != name or not all_hit(text, rule.get("match", [])):
                continue
            if rule.get("nth") and int(rule["nth"]) != self.calls[name]:
                continue
            if tool["kind"] == "commit" and rule.get("effect") != "none":
                self._record_commit(name, args)
            if tool["kind"] == "ui":
                decision = approval_decider(args) if approval_decider else "approved"
                self.approvals.append({"args": args, "decision": decision})
            return rule.get("result_raw", rule["result"]) if self.raw else rule["result"]
        h = tool.get("handler", {"type": "rules"})
        t = h.get("type")
        if t == "commit":
            return self._record_commit(name, args)
        if t == "approval":
            decision = approval_decider(args) if approval_decider else "approved"
            self.approvals.append({"args": args, "decision": decision})
            return {"decision": decision}
        if t == "profile":
            return self.member.get(h["field"], {})
        if t == "view":
            rows = self.member.get(h["field"], [])
            q = " ".join(norm(args.get(a, "")) for a in h.get("search_args", []) if args.get(a)).strip()
            if q:
                toks = [w for w in q.split() if len(w) > 1]
                rows = [r for r in rows if all(w in norm(json.dumps(r, ensure_ascii=False)) for w in toks)]
            try:
                limit = int(args.get("limit", h.get("limit_default", 20)))
                offset = int(args.get("offset", 0))
            except (TypeError, ValueError):
                limit, offset = h.get("limit_default", 20), 0
            page = rows[offset: offset + max(1, min(limit, 100))]
            return {"total": len(rows), "offset": offset, "items": page}
        return {"result": "No matching records found."}

    def _record_commit(self, name: str, args: dict) -> dict:
        self._receipt += 1
        c = {"tool": name, "args": args, "receipt": f"TX-{self.s['id']}-{self._receipt:03d}"}
        self.commits.append(c)
        return {"status": "completed", "receipt_id": c["receipt"]}
