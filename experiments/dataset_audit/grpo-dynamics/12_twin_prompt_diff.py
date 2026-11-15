#!/usr/bin/env python3
"""grpo-dynamics lens, step 12: are the A and B prompts of a twin pair distinguishable at all, and where do they differ?

For every (pair_id, level, prompt_setting) cell of grpo_train.jsonl holding both twins: are the prompts identical (a direct
contradiction: same input, opposite targets)? Which messages differ (system / member request / tool call / tool result)? How
many characters differ (difflib on each differing message)? Then: does M1's discrimination on sampled pairs relate to where
or how much the twins differ (only 13 complete pairs were sampled at one level in M2, so this is descriptive)?
Output: results/twin_prompt_diff.json
"""
import difflib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/grpo_train.jsonl"))]
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
cells = {}
for i, r in enumerate(grpo):
    cells.setdefault((r["pair_id"], r["level"], r["prompt_setting"]), {})[r["twin"][0]] = i
names = {0: "system", 1: "member_request", 2: "lookup_call", 3: "lookup_result", 4: "call2", 5: "result2"}
rows = []
for (pid, lv, ps), v in cells.items():
    if "A" not in v or "B" not in v:
        continue
    a, b = grpo[v["A"]]["prompt"], grpo[v["B"]]["prompt"]
    where, nchar = [], 0
    if len(a) != len(b):
        where.append("n_messages")
    for j in range(min(len(a), len(b))):
        if a[j]["content"] != b[j]["content"]:
            where.append(names.get(j, f"msg{j}"))
            sm = difflib.SequenceMatcher(None, a[j]["content"], b[j]["content"], autojunk=False)
            nchar += sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    rows.append({"pair_id": pid, "level": lv, "prompt_setting": ps, "identical": not where, "where": "+".join(where), "diff_chars": nchar,
                 "A_row": v["A"], "B_row": v["B"]})
D = pd.DataFrame(rows)
R = {"n_cells_both_twins_same_setting": int(len(D)), "identical_prompts": int(D.identical.sum()),
     "where_counts": D["where"].value_counts().to_dict(),
     "diff_chars": {"median": float(D.diff_chars.median()), "p10": float(D.diff_chars.quantile(0.1)), "min": int(D.diff_chars.min()),
                    "max": int(D.diff_chars.max())},
     "system_differs_share": round(float(D["where"].str.contains("system").mean()), 4),
     "only_system_differs": int((D["where"] == "system").sum()),
     "smallest_diffs": D.nsmallest(6, "diff_chars")[["pair_id", "level", "prompt_setting", "where", "diff_chars"]].to_dict("records")}
# M1 on sampled complete pairs (same level; setting may differ)
gp = G.pivot_table(index=["pair_id", "level"], columns="twin", values=["p_ask", "pass_top"], aggfunc="mean").dropna()
R["sampled_pairs_same_level"] = int(len(gp))
if len(gp):
    R["sampled_pairs_mean_ask_gap"] = round(float((gp[("p_ask", "A")] - gp[("p_ask", "B")]).mean()), 4)
    R["sampled_pairs_frac_A_ask_gt_B_ask"] = round(float((gp[("p_ask", "A")] > gp[("p_ask", "B")]).mean()), 4)
D.to_csv(os.path.join(OUT, "twin_prompt_diff.csv"), index=False)
json.dump(R, open(os.path.join(OUT, "twin_prompt_diff.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_12")
