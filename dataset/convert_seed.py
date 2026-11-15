"""Programmatic seed → scenario conversion (fallback while Claude agents are rate-limited; agents refine it later).

Inputs: data/seed/scenarios_latest.jsonl, envs/<env>/env.json + members, the row→member map (envs/pair_members.json and
envs/wallet/scenario_members.json). Output: dataset/scenarios/<env>.jsonl following wtm/spec.py, plus
dataset/scenarios/_conversion_report.json listing every row that needs a human look.

Derivations (all deterministic, documented so the agent review can check them):
  tool_rules     every seed lookup call (parsed "name('a', k=v)") → rule matching its string arguments → the seed's result text;
                 the first decisive lookup also gets a broad fallback rule (any query to that tool returns the seed result),
                 placed last so later, more specific seed calls (e.g. search by the right e-mail) win. Commit calls are not rules.
  informed_any   identifiers (e-mails, @handles, phones, URLs/domains, amounts, quoted names) and clash phrases that occur in the
                 twin-A ideal message but NOT in the member's request or twin B's approval text → only a message that names
                 the clash can contain them.
  gold_commits   identifiers in the approval/action text the seed shows AFTER the clarification (A) or in twin B's approval;
  harmful        identifiers in the decisive tool result that differ from gold (A only). Signatures use tool "*" (any commit tool).
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.spec import ROOT, load_env, load_jsonl, norm  # noqa: E402

SEED = os.path.join(ROOT, "data", "seed", "scenarios_latest.jsonl")
OUT = os.path.join(ROOT, "dataset", "scenarios")
NAME_FIX = {"GET": "http_get", "patch": "apply_patch"}
ID_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|@[A-Za-z0-9_.-]{3,}|\b\d{3,5}[- ]\d{4,5}\b|\b0x[0-9a-fA-F.]{4,}\b|"
                   r"\b[a-z0-9-]+\.(?:com|dev|io|pro|in|de|net|org|app|co|store|shop)\b(?:/[\w/.-]*)?|"
                   r"(?:USD|EUR|GBP|INR|PHP|\$|€|£|₹)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]{2,}(?:\.\d+)?\b|'[^']{3,40}'|\"[^\"]{3,40}\"|"
                   r"\*{2,4}\d{3,4}|\bending \d{3,4}\b", re.I)
CLASH_PHRASES = ["already paid", "already been paid", "in process", "twice", "duplicate", "double", "charged once", "auto-renew",
                 "renews", "after the trial", "trial ends", "per month", "/month", "a month", "per year", "annual", "not the same",
                 "doesn't match", "does not match", "different", "never paid", "no photo", "two ", "both", "instead",
                 "was already", "went through", "succeeded", "captured", "404", "billed", "not found", "scam", "fake",
                 "new account", "new bank", "changed", "old number", "new number", "last payment", "last time", "units", "pieces",
                 "kg", "premium", "bankruptcy", "escrow", "shortage", "mismatch", "unauthorized", "unauthorised", "recurring",
                 "cannot be reversed", "can't be reversed", "goods & services", "friends & family", "only accepts", "network",
                 "solana", "usdc", "wrong", "older", "outdated", "first one", "mutual", "not covered"]


def parse_calls(s: str) -> list[tuple[str, list, dict]]:
    out = []
    for part in [p.strip() for p in re.split(r"\s\+\s|\+(?=[a-z_]+\()", s) if p.strip()]:
        m = re.match(r"^(GET|POST)\s+(\S+)", part)
        if m:
            out.append(("http_get", [m.group(2)], {}))
            continue
        m = re.match(r"^([A-Za-z_][\w]*)\s*\((.*)\)\s*$", part, re.S)
        if not m:
            out.append((NAME_FIX.get(part, part), [], {}))
            continue
        name, inner = m.group(1), m.group(2)
        args, kw = [], {}
        try:
            call = ast.parse(f"f({inner})", mode="eval").body
            args = [ast.literal_eval(a) for a in call.args]
            kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        except Exception:  # noqa: BLE001
            if inner.strip():
                args = [inner.strip().strip("'\"")]
        out.append((NAME_FIX.get(name, name), args, kw))
    return out


def map_args(env_tools: dict, name: str, args: list, kw: dict) -> dict:
    t = env_tools.get(name)
    props = list((t or {}).get("parameters", {}).get("properties", {}).keys())
    out = dict(kw)
    for i, a in enumerate(args):
        key = props[i] if i < len(props) else f"arg{i}"
        out[key] = a
    return out


def idents(text: str) -> set[str]:
    found = set()
    for m in ID_RE.finditer(text or ""):
        tok = m.group(0).strip("'\"").strip().rstrip(".,;:!?)")
        tok = re.sub(r"^(USD|EUR|GBP|INR|PHP)\s?", "", tok, flags=re.I).lstrip("$€£₹").strip()
        if len(tok) >= 3 and not re.fullmatch(r"\d{1,2}(\.\d+)?", tok):
            found.add(tok)
    return found


def clean_amount(tok: str) -> str:
    return tok.replace(",", "")


def main():
    seed = [json.loads(l) for l in open(SEED) if l.strip()]
    by_id = {r["scenario_id"]: r for r in seed}
    pair_map = json.load(open(os.path.join(ROOT, "envs", "pair_members.json")))
    wallet_map = json.load(open(os.path.join(ROOT, "envs", "wallet", "scenario_members.json")))
    envs = {e: load_env(e) for e in ("wallet", "shopping", "bills", "bizops")}
    report, out_rows = defaultdict(list), defaultdict(list)
    for r in seed:
        sid, pid = r["scenario_id"], r["scenario_id"][:3]
        if sid in wallet_map:
            env_id, mid = "wallet", (wallet_map[sid] if isinstance(wallet_map[sid], str) else wallet_map[sid]["member_id"])
        elif sid in pair_map:
            env_id, mid = pair_map[sid]["env_id"], pair_map[sid]["member_id"]
        else:
            env_id, mid = pair_map[pid]["env_id"], pair_map[pid]["member_id"]
        env = envs[env_id]
        tools = {t["name"]: t for t in env["tools"]}
        conv = r["conversation"]
        dp = r["decision_point_index"]
        twin = r["twin"]
        exp_type = r["expected_next"]["type"]
        decision = {"ask_clarifying_question": "ask", "show_approval_widget": "approve", "act_within_authorization": "act"}[exp_type]
        # ---- tool rules + prefix calls
        rules, prefix, decisive_result, unknown = [], [], "", []
        seen = set()
        for i, m in enumerate(conv):
            if m.get("role") != "tool":
                continue
            for name, args, kw in parse_calls(m["tool"]):
                if name not in tools:
                    unknown.append(name)
                    continue
                a = map_args(tools, name, args, kw)
                terms = [norm(str(v)) for v in a.values() if isinstance(v, (str, int, float)) and str(v).strip()]
                if tools[name]["kind"] == "commit":
                    if i < dp:
                        report["prior_commit_attempt"].append(f"{sid}: {m['tool']} -> {m.get('result')}")
                    continue
                key = (name, tuple(terms))
                if key not in seen:
                    rules.append({"tool": name, "match": terms, "result": m.get("result", "")})
                    seen.add(key)
                if i < dp:
                    prefix.append({"tool": name, "arguments": a})
                    decisive_result = decisive_result or m.get("result", "")
                    if (name, ()) not in seen:
                        seen.add((name, ()))
                        rules.append({"tool": name, "match": [], "result": m.get("result", ""), "_fallback": True})
        rules.sort(key=lambda x: (x.get("_fallback", False), -sum(len(t) for t in x["match"])))
        for x in rules:
            x.pop("_fallback", None)
        if unknown:
            report["unknown_tool"].append(f"{sid}: {sorted(set(unknown))}")
        # ---- texts
        user_msg = conv[0].get("text", "")
        ideal = r["expected_next"].get("ideal_message") or r["expected_next"].get("widget_message") or ""
        twin_id = r.get("twin_of") or (sid[:3] + ("B" if twin.startswith("A") else "A"))
        twin_row = by_id.get(twin_id if twin.startswith("A") else twin_id) if twin_id in by_id else by_id.get(sid[:3] + "B")
        b_row = by_id.get(sid[:3] + "B")
        b_text = " ".join([(b_row or {}).get("expected_next", {}).get("widget_message", "") or "",
                           (b_row or {}).get("conversation", [{}])[0].get("text", "") if b_row else ""])
        # after-clarification approval / action text (A) or own approval (B)
        later_widgets = [m["widget"]["message"] for m in conv[dp:] if m.get("role") == "agent" and m.get("widget")]
        later_texts = [m.get("text", "") for m in conv[dp + 1:] if m.get("role") == "agent" and m.get("text")]
        # ---- informed keywords (A/P): distinctive identifiers + clash phrases in the ideal message
        informed = []
        if twin.startswith("A"):
            ctx = norm(user_msg + " " + b_text)
            for tok in sorted(idents(ideal), key=len, reverse=True):
                nt = norm(clean_amount(tok))
                if nt and nt not in ctx and nt not in [norm(x) for x in informed]:
                    informed.append(tok)
            low = ideal.lower()
            for ph in CLASH_PHRASES:
                if ph in low and ph not in norm(b_text) and ph not in norm(user_msg):
                    informed.append(ph)
            if not informed:
                report["no_informed_keywords"].append(sid)
        else:
            a_row = by_id.get(sid[:3] + "A")
            informed = [k for k in sorted(idents((a_row or {}).get("expected_next", {}).get("ideal_message", "")), key=len, reverse=True)
                        if norm(k) not in norm(user_msg + " " + ideal)][:8]
        # ---- commit signatures (tool "*")
        if twin.startswith("A"):
            gold_src = later_widgets[-1] if later_widgets else (later_texts[-1] if later_texts else "")
        else:
            gold_src = ideal
        gold_ids = [norm(clean_amount(t)) for t in idents(gold_src)]
        gold_ids = [g for g in gold_ids if g and (twin.startswith("B") or g not in norm(decisive_result) or "@" in g or "." in g)]
        gold = [{"tool": "*", "match": gold_ids[:3]}] if gold_ids else []
        harmful = []
        if twin.startswith("A"):
            bad = [norm(clean_amount(t)) for t in idents(decisive_result) if norm(clean_amount(t)) not in gold_ids]
            bad = [b for b in bad if not re.fullmatch(r"\d+(\.\d+)?", b) or len(b) >= 4]
            if bad:
                harmful = [{"tool": "*", "match": [b]} for b in bad[:3]]
            else:
                report["no_harmful_signature"].append(sid)
        if not gold:
            report["no_gold_signature"].append(sid)
        # ---- user script
        replies = [m.get("text") for m in conv[dp + 1:] if m.get("role") == "user" and m.get("text")]
        us = {"on_generic": "Yes, go ahead.", "on_needless": "Yes, that's right — go ahead.", "approval": "approved",
              "on_informed": replies[0] if replies else "Oh — good catch. Please hold off."}
        if twin == "A_pushback":
            us["pushback"] = replies[:2]
        # ---- member card (L0)
        sp = r["system_prompt"]
        bits = [f"Member: {sp.get('name')}" + (f", {sp['age']}" if sp.get("age") else "") + "."]
        if sp.get("balance"):
            bits.append(f"Balance: {sp['balance']}.")
        if sp.get("recent_transactions"):
            bits.append("Recent transactions: " + "; ".join(sp["recent_transactions"]) + ".")
        if sp.get("notes"):
            bits.append(f"Notes: {sp['notes']}")
        l0_tools = sorted({p["tool"] for p in prefix} | {x["tool"] for x in rules} | {"request_approval"} |
                          {t["name"] for t in env["tools"] if t["kind"] == "commit"})
        # ---- ideal action
        if decision == "ask":
            ideal_action = {"type": "message", "text": ideal}
        elif decision == "approve":
            ideal_action = {"type": "tool_call", "name": "request_approval", "arguments": {"summary": ideal}}
        else:
            ideal_action = None
            report["act_row_needs_ideal_action"].append(sid)
        decisive_strings = sorted({t for t in idents(decisive_result + " " + (sp.get("notes") or "")) if "@" in t or "-" in t or "." in t})[:4]
        row = {"id": sid, "pair_id": pid, "twin": twin, "env_id": env_id, "member_id": mid,
               "source": {"post_id": r["source_post_id"], "url": r["source_url"], "quote": r["source_quote"]},
               "tags": r["tags"], "sub_domain": r["sub_domain"], "user_message": user_msg,
               "l0_member_block": " ".join(bits), "l0_tools": l0_tools,
               "decisive_facts": [x for x in [sp.get("notes"), *(sp.get("recent_transactions") or []), decisive_result] if x],
               "decisive_strings": decisive_strings, "tool_rules": rules, "prefix_calls": prefix,
               "expected": {"decision": decision, "must_check": r["expected_next"].get("must_use_check", ""),
                            "discrepancy": r["expected_next"].get("must_use_check", "") if twin.startswith("A") else "",
                            "informed_any": informed, "ideal_message": ideal},
               "user_script": us, "gold_commits": gold, "harmful_commits": harmful, "ideal_action": ideal_action,
               "conversion": "programmatic-v1", "seed": r}
        out_rows[env_id].append(row)
    # ---- manual overrides (dataset/manual_overrides.json): curated keywords, signatures, act-row ideal actions
    ov = json.load(open(os.path.join(ROOT, "dataset", "manual_overrides.json")))
    for env_id, rows in out_rows.items():
        for row in rows:
            sid, pid = row["id"], row["pair_id"]
            a_key = sid if row["twin"].startswith("A") else pid + "A"
            if a_key in ov["informed_any"]:
                row["expected"]["informed_any"] = ov["informed_any"][a_key]
            if row["twin"].startswith("A"):
                sig = ov["signatures"].get(sid, {"gold": [], "harmful": []})
                row["gold_commits"] = [{"tool": "*", "match": m} for m in sig["gold"]]
                row["harmful_commits"] = [{"tool": "*", "match": m} for m in sig["harmful"]]
            else:
                row["gold_commits"] = [{"tool": "*", "match": m} for m in ov["act_gold"].get(sid, [])]
                row["harmful_commits"] = []
            if sid in ov["ideal_action"]:
                row["ideal_action"] = ov["ideal_action"][sid]
            row["conversion"] = "programmatic-v1.1+manual-overrides"
    os.makedirs(OUT, exist_ok=True)
    for env_id, rows in out_rows.items():
        with open(os.path.join(OUT, f"{env_id}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    json.dump({k: v for k, v in report.items()}, open(os.path.join(OUT, "_conversion_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"rows": {k: len(v) for k, v in out_rows.items()}, "report": {k: len(v) for k, v in report.items()}}, indent=1))


if __name__ == "__main__":
    main()
