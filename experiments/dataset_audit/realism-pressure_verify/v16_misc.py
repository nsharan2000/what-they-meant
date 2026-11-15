"""Small saved counts cited in the verify verdicts (so no number is quoted from an ad-hoc command):
(1) B decision targets by level x prompt setting x kind (approval card / direct tool call / message) in sft_train;
(2) env templates: L0 vs L1 length and presence of approval / standing-authorisation / safety text; P2 hint text;
(3) L1-L3 decision rows whose member block repeats an OUTGOING payment line, and whether that payee is named in the request;
(4) training scenarios tagged new:duplicate_payment;
(5) frozen v1 -> current tool_rules changes for seed / variant rows, and how many are missing from the fix reports;
(6) H051A/B: decisive transactions that exist only in tool_rules (not in member file or scenario_records).
Outputs: results/v16_misc.json
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

from vcommon import CUR, ROOT, dump, load_jsonl


def main():
    out = {}
    sft = load_jsonl(os.path.join(CUR, "sft_train.jsonl"))
    c = Counter()
    for r in sft:
        if r["turn"] != "decision" or not r["twin"].startswith("B"):
            continue
        t = r["completion"][0]["content"]
        m = re.search(r'"name"\s*:\s*"([a-z_]+)"', t) if "<tool_call>" in t else None
        kind = "message" if m is None else ("approval_card" if m.group(1) == "request_approval" else "direct_tool_call")
        c[f"{r['level']}|{r['prompt_setting']}|{kind}"] += 1
    out["B_decision_targets"] = dict(sorted(c.items()))
    out["B_decision_rows"] = sum(c.values())
    # (2) templates
    tm = {}
    for env in ("wallet", "bills", "shopping", "bizops"):
        e = json.load(open(os.path.join(ROOT, "envs", env, "env.json")))
        sp = e["system_prompt"]
        tm[env] = {"L0_chars": len(sp["L0"]), "L1_chars": len(sp["L1"]),
                   **{f"{lv}_mentions_{k}": (k.lower() in sp[lv].lower()) for lv in ("L0", "L1")
                      for k in ("request_approval", "standing", "approval card", "scam")}}
    out["templates"] = tm
    out["P2_hint"] = open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()
    # (3) duplicated outgoing payment lines in decision rows
    HL = re.compile(r"^- (\d{4}-\d{2}-\d{2}) \| -[^|]+\| ([^|(\n]+)")
    n = named = 0
    ex = []
    for i, r in enumerate(sft):
        if r["turn"] != "decision" or r["level"] not in ("L1", "L2", "L3"):
            continue
        s = r["prompt"][0]["content"]
        mb = s[s.find("## Member"):s.find("## Tools")]
        cnt = Counter(l for l in mb.splitlines() if l.startswith("- "))
        dups = [l for l, k in cnt.items() if k > 1 and HL.match(l)]
        if not dups:
            continue
        n += 1
        req = r["prompt"][1]["content"].lower()
        hit = any(HL.match(l).group(2).strip().split()[0].lower() in req for l in dups if HL.match(l).group(2).strip())
        named += int(hit)
        if hit and len(ex) < 6:
            ex.append({"line": i, "sid": r["scenario_id"], "twin": r["twin"][0], "level": r["level"],
                       "request": r["prompt"][1]["content"][:90], "dup_line": dups[0][:110]})
    out["decision_rows_L1to3_dup_outgoing"] = {"rows": n, "payee_named_in_request": named, "examples": ex}
    # (4) duplicate_payment clash in training
    specs = {}
    for r in load_jsonl(os.path.join(CUR, "grpo_train.jsonl")):
        specs[r["scenario_id"]] = json.loads(r["scenario"])
    out["train_scenarios_tagged_duplicate_payment"] = sorted(k for k, s in specs.items() if "new:duplicate_payment" in s.get("tags", []))
    # (5) frozen v1 -> current tool_rules changes
    changed_rep = {c["row"] for f in ("fix_v12_report.json", "fix_v121_report.json")
                   for c in json.load(open(os.path.join(ROOT, "dataset", f)))["changes"]}
    res5 = {}
    for name, fz, cu in (("seed", "frozen_replication_v1", "scenarios"), ("variants", "frozen_variants_v1", "scenarios_synth")):
        old = {s["id"]: s for f in glob.glob(os.path.join(ROOT, "dataset", fz, "*.jsonl")) for s in load_jsonl(f)}
        new = {s["id"]: s for f in glob.glob(os.path.join(ROOT, "dataset", cu, "*.jsonl")) for s in load_jsonl(f)}
        ch = [k for k in old if k in new and json.dumps(old[k].get("tool_rules"), sort_keys=True) !=
              json.dumps(new[k].get("tool_rules"), sort_keys=True)]
        res5[name] = {"rows_in_both": sum(1 for k in old if k in new), "tool_rules_changed": len(ch),
                      "changed_not_in_fix_reports": len(set(ch) - changed_rep)}
        if name == "seed":
            res5["S53B_lookup_venmo_frozen"] = [r.get("result_raw") for r in old["S53B"]["tool_rules"] if r["tool"] == "lookup_venmo"][:1]
            res5["S53B_lookup_venmo_current"] = [r.get("result_raw") for r in new["S53B"]["tool_rules"] if r["tool"] == "lookup_venmo"][:1]
    out["frozen_vs_current_tool_rules"] = res5
    # (6) H051
    h = {}
    for f in glob.glob(os.path.join(ROOT, "dataset", "eval", "hard_test_wallet.jsonl")):
        for s in load_jsonl(f):
            if s["id"] not in ("H051A", "H051B"):
                continue
            m = json.load(open(os.path.join(ROOT, "envs", s["env_id"], "members", f"{s['member_id']}.json")))
            mem_ids = [t.get("id") for t in m.get("transactions", [])]
            rec_ids = [x["record"].get("id") for x in s.get("scenario_records") or [] if isinstance(x["record"], dict)]
            rule_tx = []
            for r in s["tool_rules"]:
                if r["tool"] == "get_transactions":
                    for t in (r.get("result_raw") or {}).get("transactions", []):
                        rule_tx.append({"id": t.get("id"), "date": t.get("date"), "amount": t.get("amount"),
                                        "in_member_file": t.get("id") in mem_ids, "in_scenario_records": t.get("id") in rec_ids,
                                        "member_file_and_records_both": t.get("id") in mem_ids and t.get("id") in rec_ids})
            h[s["id"]] = {"user_message": s["user_message"], "get_transactions_rule": rule_tx}
    out["H051"] = h
    dump(out, "v16_misc.json")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
