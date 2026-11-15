#!/usr/bin/env python3
"""grpo-dynamics lens, step 10: do null / empty fields in the rendered tool results (L0r/L1/L2/L3 show rule['result_raw'])
push M1 toward asking on should-act (B) prompts?

For each GRPO prompt: parse every <tool_result> JSON body in the prompt, count leaf values that are null, empty lists/dicts or
empty strings ('n_empty') and all leaves ('n_leaves'). Correlate with M1's sampled ask rate on B prompts (needless asks) and
pass rate on A prompts, over the 120 sampled groups (Spearman + a simple binary split). Also the prevalence over all 496 rows.
Output: results/tool_result_nulls.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/grpo_train.jsonl"))]
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
TR = re.compile(r'<tool_result name="[^"]*">\s*(.*?)\s*</tool_result>', re.S)


def leaves(o):
    if isinstance(o, dict):
        if not o:
            yield None
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        if not o:
            yield None
        for v in o:
            yield from leaves(v)
    else:
        yield o


rows = []
for i, r in enumerate(grpo):
    n_e = n_l = n_bad = 0
    for m in r["prompt"]:
        if m["role"] != "user":
            continue
        for body in TR.findall(m["content"]):
            try:
                obj = json.loads(body)
            except ValueError:
                n_bad += 1
                continue
            for v in leaves(obj):
                n_l += 1
                n_e += int(v is None or v == "")
    rows.append({"row_i": i, "n_empty": n_e, "n_leaves": n_l, "n_unparsed": n_bad, "twin": r["twin"][0], "level": r["level"]})
T = pd.DataFrame(rows)
R = {"all_rows": {"rows_with_tool_results": int((T.n_leaves > 0).sum()), "rows_with_any_empty": int((T.n_empty > 0).sum()),
                  "mean_empty_share": round(float((T.n_empty / T.n_leaves.clip(lower=1))[T.n_leaves > 0].mean()), 4),
                  "by_twin_any_empty": T[T.n_leaves > 0].groupby("twin").apply(lambda x: round((x.n_empty > 0).mean(), 4), include_groups=False).to_dict(),
                  "unparsed_bodies": int(T.n_unparsed.sum())}}
M = G.merge(T[["row_i", "n_empty", "n_leaves"]], on="row_i")
M["empty_share"] = M.n_empty / M.n_leaves.clip(lower=1)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)

out = {}
for tw, ycol in (("B", "p_ask"), ("B", "pass_top"), ("A", "pass_top"), ("A", "p_ask")):
    sub = M[(M.twin == tw) & (M.n_leaves > 0)]
    hi = sub[sub.n_empty >= 2]
    lo = sub[sub.n_empty < 2]
    out[f"{tw}|{ycol}"] = {"spearman_vs_n_empty": spearman(sub.n_empty, sub[ycol]), "spearman_vs_empty_share": spearman(sub.empty_share, sub[ycol]),
                           "n": int(len(sub)), "mean_if_n_empty>=2": round(hi[ycol].mean(), 3), "n_hi": int(len(hi)),
                           "mean_if_n_empty<2": round(lo[ycol].mean(), 3), "n_lo": int(len(lo))}
R["sampled_groups"] = out
R["B_examples_high_empty"] = M[(M.twin == "B")].sort_values("n_empty", ascending=False).head(6)[
    ["row_i", "scenario_id", "level", "n_empty", "n_leaves", "p_ask", "pass_top"]].to_dict("records")
json.dump(R, open(os.path.join(OUT, "tool_result_nulls.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_10")
