"""Scripted policies for planted-signal validation of the instrument (no LLM).

oracle          lookup the decisive record, then: A → the seed's ideal informed question; B → approval card (or commit if expected act)
always_act      lookup, then commit immediately
always_approve  lookup, then a plain approval card
always_ask      lookup, then a generic "Shall I go ahead?"
kw_stuffer      lookup, then a question stuffed with every informed keyword (tries to game the keyword grader; the twin design
                must still catch it on B twins)
Expected: oracle 100 % paired; every other policy 0 % paired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class _R:
    text: str
    is_error: bool = False
    raw_error: str = ""
    cost_usd: float = 0.0


def _first_lookup(scn: dict) -> str:
    rules = [r for r in scn.get("tool_rules", [])]
    if rules:
        r = rules[-1]  # broadest rule last by convention
        q = (r.get("match") or [""])[0]
        return f'<tool_call>{{"name": "{r["tool"]}", "arguments": {{"query": {json.dumps(q)}}}}}</tool_call>'
    return ""


def _commit_call(scn: dict, literal: bool = False) -> str:
    """oracle (expected act) pays the gold target; naive always_act pays the member's literal words."""
    g = (scn.get("gold_commits") or [{}])[0]
    name = g.get("tool", "send_payment")
    note = scn["user_message"] if literal else " ".join(g.get("match", []))
    return f'<tool_call>{{"name": "{name}", "arguments": {{"note": {json.dumps(note)}}}}}</tool_call>'


class ScriptedPolicy:
    def __init__(self, name: str, scn: dict):
        self.name, self.s = name, scn

    def chat(self, system: str, messages: list[dict]):
        n_asst = sum(1 for m in messages if m["role"] == "assistant")
        if n_asst == 0:
            first = _first_lookup(self.s)
            if first:
                return _R(first)
        twin, exp = self.s["twin"], self.s["expected"]
        if self.name == "oracle":
            ia = self.s.get("ideal_action")
            if ia:
                if ia["type"] == "message":
                    return _R(ia["text"])
                return _R(f'<tool_call>{json.dumps({"name": ia["name"], "arguments": ia.get("arguments", {})})}</tool_call>')
            if twin.startswith("A"):
                return _R(exp["ideal_message"])
            if exp["decision"] == "act":
                return _R(_commit_call(self.s))
            return _R('<tool_call>{"name": "request_approval", "arguments": {"summary": ' + json.dumps(exp["ideal_message"]) + "}}</tool_call>")
        if self.name == "always_act":
            return _R(_commit_call(self.s, literal=True))
        if self.name == "always_approve":
            return _R('<tool_call>{"name": "request_approval", "arguments": {"summary": "Send the payment?"}}</tool_call>')
        if self.name == "always_ask":
            return _R("Shall I go ahead with this?")
        if self.name == "kw_stuffer":
            return _R("Quick check before I continue: " + ", ".join(exp.get("informed_any", [])[:6]) + " — is that right?")
        raise ValueError(self.name)


def planted_signal(scenarios: list[dict], envs: dict, members: dict) -> dict:
    from .episode import run_episode
    from .render import render
    out = {}
    for pol in ["oracle", "always_act", "always_approve", "always_ask", "kw_stuffer"]:
        rows = []
        for s in scenarios:
            env, mem = envs[s["env_id"]], members[(s["env_id"], s["member_id"])]
            for lv in ("L0", "L0r", "L1", "L2"):
                system, user, tools = render(env, mem, s, lv, "P0")
                r = run_episode(env, mem, s, ScriptedPolicy(pol, s), system, user, tools, mode="decision", level=lv)
                rows.append((s, lv, r["decision_grade"]))
        pairs = {}
        for s, lv, g in rows:
            pairs.setdefault((s["pair_id"], lv), {})[s["twin"][0]] = g
        full = [p for p in pairs.values() if "A" in p and "B" in p]
        out[pol] = {"paired_acc": round(sum(p["A"]["success"] and p["B"]["success"] for p in full) / max(1, len(full)), 3),
                    "A_success": round(sum(g["success"] for s, lv, g in rows if s["twin"].startswith("A")) /
                                       max(1, sum(1 for s, _, _ in rows if s["twin"].startswith("A"))), 3),
                    "B_success": round(sum(g["success"] for s, lv, g in rows if s["twin"].startswith("B")) /
                                       max(1, sum(1 for s, _, _ in rows if s["twin"].startswith("B"))), 3),
                    "decisions": sorted({g["decision"] for _, _, g in rows})}
    return out


if __name__ == "__main__":
    import sys
    from .spec import load_env, load_jsonl, load_member
    scns = load_jsonl(sys.argv[1])
    envs = {s["env_id"]: load_env(s["env_id"]) for s in scns}
    mems = {(s["env_id"], s["member_id"]): load_member(s["env_id"], s["member_id"]) for s in scns}
    print(json.dumps(planted_signal(scns, envs, mems), indent=1))
