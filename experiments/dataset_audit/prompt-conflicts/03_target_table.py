"""Step 3 — one record per training target (SFT decision + look-up turns, DPO chosen + rejected, GRPO hidden label).

Per record: identity (file, 0-based line index, scenario, twin, env, level, prompt setting, turn), the target's parse under the
protocol (wtm.protocol.parse_action with the tools listed in THAT row's prompt), its decision class and reward under the GRPO
reward (wtm.reward.reward = the evaluation grader + shaping), message features (questions, words, report-path pointers, scam
words, tool names), card-content features, prompt features (look-ups in the prompt, authorisation phrases visible) and the
analyst annotations (annotations.py). GRPO rows have no target: their record carries the hidden label (expected decision,
gold commits) the reward reads.
Output: results/target_table.jsonl (+ results/target_table_checks.json with spec-consistency checks).
    python3 -B experiments/dataset_audit/prompt-conflicts/03_target_table.py
"""
from __future__ import annotations

import collections
import json
import re

from annotations import A as A_ANN, B_ACT_UNCOVERED, B_SA, CARD_FIELDS
from pc_common import (FILES, any_hit, decision_of, dump, dump_jsonl, load_envs, load_rows, load_specs, n_lookups, parse_action,
                       rel, reward, split_system, tool_kinds, tools_in_docs)

AMOUNT_RE = re.compile(r"(?:[$€£₹₱]\s?\d|\b(?:USD|EUR|GBP|INR|PHP)\s?\d|\d[\d,.]*\s?(?:USD|EUR|GBP|INR|PHP|USDC|PYUSD|SOL|ETH)\b)")
METHOD_RE = re.compile(r"PayPal|Venmo|Zelle|Cash App|\bUPI\b|Friends|Goods|purchase|personal|business|Visa|Mastercard|\bcard\b|"
                       r"\*{4}\d|••\d|ending|account|Checking|Savings|\bACH\b|wire|\bbank\b|balance|QR|Pay in 3|debit|"
                       r"saved wallet|deposit address", re.I)
RECIP_ID_RE = re.compile(r"@[\w-]|[\w.+-]+@[\w-]+\.[\w.]+|\b\d{3}-\d{4}\b|0x[0-9a-fA-F]|\.\.\.[0-9a-fA-F]{4}|\(business\)", re.I)
FEE_RE = re.compile(r"\bfee|free\b|no fee|protection|cannot be reversed|can't be reversed", re.I)
WHEN_RE = re.compile(r"\btoday\b|\bnow\b|\bdue\b|tomorrow|\b20\d\d-\d\d-\d\d\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b|"
                     r"/month|/year|monthly|annual|renew|trial|7 days|every", re.I)
ITEMS_RE = re.compile(r"\bx\d|\d+\s?(?:kg|lb|lbs|pcs|L)\b|order|\bitems?\b|OLED|domain|license|basket|\(.*\)", re.I)
COUNT_RE = re.compile(r"\b\d[\d,]*\s+(?:orders|pre-orders|invoices|bills|items|customers|seats)\b", re.I)
SCAM_RE = re.compile(r"\bscam|fraud|phish|impersonat|fake|suspicious|criminal|thief|steal|trick", re.I)
REPORT_RE = re.compile(r"Report a problem|Report fraud|Settings > Security|Help >|Chat with", re.I)
LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s", re.M)


def card_summary(act) -> str:
    a = act.arguments or {}
    return " ".join(str(a.get(k, "")) for k in ("summary", "details")).strip()


def msg_feats(text: str, tool_names: set) -> dict:
    t = text or ""
    return {"n_qmarks": t.count("?"), "words": len(t.split()),
            "sentences": len([s for s in re.split(r"(?<=[.?!])\s+", t.strip()) if s]),
            "has_list": bool(LIST_RE.search(t)), "newlines": t.count("\n"),
            "tool_names_mentioned": sorted(n for n in tool_names if re.search(r"\b" + re.escape(n) + r"\b", t)),
            "scam_words": sorted(set(m.group(0).lower() for m in SCAM_RE.finditer(t))),
            "report_path": bool(REPORT_RE.search(t))}


def card_feats(summary: str) -> dict:
    return {"amount": bool(AMOUNT_RE.search(summary)), "method": bool(METHOD_RE.search(summary)),
            "recipient_id": bool(RECIP_ID_RE.search(summary)), "fee_or_protection": bool(FEE_RE.search(summary)),
            "when_or_terms": bool(WHEN_RE.search(summary)), "items": bool(ITEMS_RE.search(summary)),
            "count": bool(COUNT_RE.search(summary)), "words": len(summary.split())}


def target_record(target: str, scn: dict, kinds: dict, listed: list[str]) -> dict:
    act = parse_action(target, set(listed))
    total, comp = reward(target, scn, kinds)
    rec = {"kind": act.kind, "tool": act.name or None, "tool_kind": kinds.get(act.name) if act.name else None,
           "strict_ok": act.strict_ok, "flags": act.flags, "tool_listed_in_prompt": (act.name in listed) if act.name else None,
           "target_newlines": target.count("\n"), "decision": comp["decision"], "reward": total, "reward_base": comp["base"]}
    if act.kind == "message":
        rec["msg"] = msg_feats(act.text, set(kinds))
        rec["informed"] = any_hit(act.text, scn["expected"].get("informed_any", []))
    if act.kind == "tool_call" and kinds.get(act.name) == "ui":
        s = card_summary(act)
        rec["card_summary"] = s
        rec["card"] = card_feats(s)
    return rec


def main():
    envs = load_envs()
    specs = load_specs()
    kinds_by_env = {e: tool_kinds(envs[e]) for e in envs}
    out, checks = [], collections.Counter()
    mism = []
    for f in FILES:
        for i, r in enumerate(load_rows(f)):
            env = envs[r["env_id"]]
            kinds = kinds_by_env[r["env_id"]]
            scn = json.loads(r["scenario"]) if f == "grpo_train" else specs[r["scenario_id"]]
            if f == "grpo_train" and json.dumps(scn, sort_keys=True) != json.dumps(specs[r["scenario_id"]], sort_keys=True):
                checks["grpo_scenario_differs_from_current_spec"] += 1
            sec = split_system(r["prompt"][0]["content"], env, r["level"])
            listed = tools_in_docs(sec["tools"])
            full_prompt = "\n".join(m["content"] for m in r["prompt"])
            base = {"file": f, "i": i, "scenario_id": r["scenario_id"], "pair_id": r["pair_id"], "twin": r["twin"],
                    "t": r["twin"][0], "env": r["env_id"], "level": r["level"], "prompt_setting": r["prompt_setting"],
                    "turn": r.get("turn", "decision"), "n_lookups_in_prompt": n_lookups(r["prompt"]),
                    "n_tools_listed": len(listed), "request_approval_listed": "request_approval" in listed,
                    "expected": scn["expected"]["decision"], "tags": scn.get("tags", []),
                    "fraud_tag": "new:scam_pressure" in scn.get("tags", [])}
            ann = A_ANN.get(r["scenario_id"])
            if ann:
                base["a_exec"], base["a_clash"] = ann
                base["a_card_states_clash"] = ann[1] in CARD_FIELDS[r["env_id"]]
            if r["scenario_id"] in B_SA:
                base["b_sa_covered"] = True
                base["b_sa_visible"] = any(p in full_prompt for p in B_SA[r["scenario_id"]])
            if r["scenario_id"] in B_ACT_UNCOVERED:
                base["b_act_uncovered"] = True
            if f == "grpo_train":
                out.append({**base, "role": "hidden_label", "gold_commits": scn.get("gold_commits") or []})
                continue
            if f.startswith("sft"):
                tgt = r["completion"][0]["content"]
                rec = target_record(tgt, scn, kinds, listed)
                if base["turn"] == "decision":
                    ideal = scn["ideal_action"]
                    ideal_txt = ideal["text"] if ideal["type"] == "message" else \
                        '<tool_call>' + json.dumps({"name": ideal["name"], "arguments": ideal.get("arguments", {})},
                                                   ensure_ascii=False) + '</tool_call>'
                    if ideal_txt != tgt:
                        mism.append([f, i, r["scenario_id"]])
                out.append({**base, "role": "target", "target": tgt, **rec})
            else:
                for role in ("chosen", "rejected"):
                    tgt = r[role][0]["content"]
                    out.append({**base, "role": role, "target": tgt, **target_record(tgt, scn, kinds, listed)})
    checks["sft_decision_target_differs_from_current_spec_ideal_action"] = len(mism)
    p = dump_jsonl("target_table.jsonl", out)
    dump("target_table_checks.json", {"checks": dict(checks), "spec_mismatch_examples": mism[:10],
                                      "records": collections.Counter(f"{o['file']}:{o['role']}:{o['turn']}" for o in out)})
    print("records", len(out), dict(checks))
    print("wrote", rel(p))


if __name__ == "__main__":
    main()
