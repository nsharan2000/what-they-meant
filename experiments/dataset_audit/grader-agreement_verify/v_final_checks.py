"""Last spot checks: harmful signatures on B-act specs (train + eval); the empty-args B approval card in the judged transcripts;
B-approve reward of a content-free card and of a wrong-amount card on every GRPO B-approve row; training B-act commits with the
right id but a 10x amount.
-> results/v_final_checks.json
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

from vcommon import EP, RES, ROOT, RW, dump, jl, kinds, rows

S = {}
bact = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl", "dataset/eval/*.jsonl"):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        for s in jl(f):
            if s["twin"].startswith("B") and s["expected"]["decision"] == "act":
                bact[s["id"]] = s
S["B_act_specs"] = len(bact)
S["B_act_specs_with_harmful_commits"] = sorted(sid for sid, s in bact.items() if s.get("harmful_commits"))
S["B_act_gold_sigs_with_amount_key"] = sorted(sid for sid, s in bact.items()
                                              if any(any(c.isdigit() for k in g.get("match", []) for c in k) and
                                                     any(k.replace(".", "").isdigit() and len(k) > 3 for k in g.get("match", []))
                                                     for g in s.get("gold_commits") or []))
jr = jl(os.path.join(RES, "v_judge_rows.jsonl"))
S["B_card_empty_args"] = [[r["key"], r["cls"], r["ok"], r["cat"]] for r in jr if r["tw"] == "B" and r["kind"] == "ui" and r["text"] in ("{}", "")]
g = rows("grpo_train")
cnt = Counter()
for r in g:
    s = json.loads(r["scenario"])
    if s["twin"].startswith("B") and s["expected"]["decision"] != "act":
        k = json.loads(r["tool_kinds"])
        cnt["empty_card_reward=" + str(RW.reward('<tool_call>{"name": "request_approval", "arguments": {}}</tool_call>', s, k)[0])] += 1
        cnt["wrong_card_reward=" + str(RW.reward('<tool_call>{"name": "request_approval", "arguments": {"summary": "Send $999,999 to an unknown account?"}}</tool_call>', s, k)[0])] += 1
S["grpo_B_approve_card_rewards"] = dict(cnt)
# training B-act: ideal commit with the amount multiplied by 10 (where the ideal has a numeric amount-like argument)
out = []
for r in g:
    s = json.loads(r["scenario"])
    if s["expected"]["decision"] == "act" and s["id"] not in [x[0] for x in out]:
        a = dict(s["ideal_action"]["arguments"])
        key = next((x for x in ("amount", "price", "total") if isinstance(a.get(x), (int, float))), None)
        if key:
            a[key] = a[key] * 10
            txt = '<tool_call>' + json.dumps({"name": s["ideal_action"]["name"], "arguments": a}) + '</tool_call>'
            out.append([s["id"], key, a[key], RW.reward(txt, s, json.loads(r["tool_kinds"]))[1]["decision"],
                        RW.reward(txt, s, json.loads(r["tool_kinds"]))[0]])
S["train_B_act_10x_amount"] = out
dump(S, "v_final_checks.json")
print(json.dumps(S, indent=1))
