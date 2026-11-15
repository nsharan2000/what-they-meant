"""Remaining spot checks.
1. every GRPO prompt contains all of its spec's prefix look-ups (assistant turns == len(prefix_calls))
2. adjudicated scenarios: decision rows per trained file, their gold_commits, and what a CORRECT direct commit (the ideal
   arguments of the B row turned into the natural commit) is classed as under reward / raw eval / adjudicated eval
3. DPO rejected sides on adjudicated rows and their classes
4. judge categories of other models' twin-B ask_informed first decisions (from results/v_judge_rows.jsonl)
5. A_TABLE / B tables as loaded (no retyping)
-> results/v_misc_summary.json
"""
from __future__ import annotations

import json
import os
from collections import Counter

from vcommon import EP, RES, RW, adjudicate, dump, jl, kinds, rows, spec_files
from wtm.labels import adjudications

specs = spec_files()
S = {}
g = rows("grpo_train")
S["grpo_prefix_mismatch"] = [[i, r["scenario_id"], r["level"]] for i, r in enumerate(g)
                             if sum(m["role"] == "assistant" for m in r["prompt"]) != len(json.loads(r["scenario"]).get("prefix_calls") or [])]
adj = adjudications()
S["adjudicated_ids"] = sorted(adj)
S["adjudicated_gold_commits"] = {sid: specs[sid].get("gold_commits") for sid in adj if sid in specs}
cnt = Counter()
for f in ("sft_train", "sft_val", "dpo_train", "grpo_train"):
    for r in rows(f):
        if r["scenario_id"] in adj and r.get("turn", "decision") == "decision":
            cnt[f] += 1
S["adjudicated_decision_rows"] = dict(cnt)
# natural direct commit for V223B (the adjudication's own example) and S43B
direct = {"V223B": ("pay_bill", {"biller": "Lincoln Energy Services", "amount": 142.33}),
          "S43B": ("pay_card_bill", {"amount": 11981, "currency": "INR"})}
S["direct_commit_on_adjudicated"] = {}
for sid, (tool, args) in direct.items():
    s = specs.get(sid)
    if not s:
        continue
    k = kinds(s["env_id"])
    if tool not in k:
        tool = next((t for t, kk in k.items() if kk == "commit" and ("bill" in t or "pay" in t)), tool)
    txt = '<tool_call>' + json.dumps({"name": tool, "arguments": args}) + '</tool_call>'
    rew, parts = RW.reward(txt, s, k)
    cls = parts["decision"]
    S["direct_commit_on_adjudicated"][sid] = {"tool": tool, "class": cls, "reward": rew,
                                              "raw_eval": EP.decision_grade(s, cls), "adjudicated_eval": EP.decision_grade(adjudicate(s), cls)}
d = rows("dpo_train")
S["dpo_rejected_on_adjudicated"] = [[i, r["scenario_id"], r["level"], RW.reward(r["rejected"][0]["content"], specs[r["scenario_id"]],
                                                                               kinds(specs[r["scenario_id"]]["env_id"]))[1]["decision"],
                                     r["rejected"][0]["content"][:160]]
                                    for i, r in enumerate(d) if r["scenario_id"] in adj]
jr = jl(os.path.join(RES, "v_judge_rows.jsonl"))
S["judge_B_ask_informed_categories"] = dict(Counter(r["cat"] for r in jr if r["tw"] == "B" and r["cls"] == "ask_informed"))
S["judge_B_messages_n"] = sum(1 for r in jr if r["tw"] == "B" and r["kind"] == "message")
S["tables"] = {"A_TABLE": RW.A_TABLE, "B_APPROVE": RW.B_APPROVE, "B_ACT": RW.B_ACT}
S["decision_grade_B_act"] = {c: EP.decision_grade({"twin": "B_should_act", "expected": {"decision": "act"}}, c)
                             for c in ("commit_gold", "commit_other", "commit_harmful", "approve_generic")}
S["decision_grade_A"] = {c: EP.decision_grade({"twin": "A_should_ask", "expected": {"decision": "ask"}}, c)
                         for c in ("ask_informed", "tell_informed", "approve_informed", "approve_gold", "approve_generic")}
dump(S, "v_misc_summary.json")
print(json.dumps(S, indent=1, ensure_ascii=False)[:7000])
