"""SEA-5 sub-claim: how many tokens differ between twin A and twin B L2 decision prompts (hint stripped)?
difflib opcodes over each message pair; differing characters (max of the two sides per opcode) -> tokens with the v06b model.
Output: results/v05b_diff_size.json
"""
import difflib
import json
import os
import re

import numpy as np

from vcommon import curated, dump, golden_text, specs

w = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "results", "v06b_tokfit_positions.json")))["weights"])
hint = golden_text("P2")


def est(t):
    f = np.array([sum(c.isalpha() and c.isascii() for c in t), sum(c.isdigit() for c in t), sum(c.isspace() for c in t),
                  sum((not c.isalnum()) and (not c.isspace()) and c.isascii() for c in t), sum(not c.isascii() for c in t),
                  len(re.findall(r"[A-Za-z]+", t)), len(re.findall(r"\d+", t)), 0.0], dtype=float)
    return max(0.0, float(f @ w))


by_id = {s["id"]: s for s in specs()}
sft = curated("sft_train")
dec = {(r["scenario_id"], r["level"]): r for r in sft if r["turn"] == "decision"}
tok_stats = {}
for line in open(os.path.join(os.path.dirname(__file__), "..", "results", "token_stats.jsonl")):
    x = json.loads(line)
    if x["file"] == "sft_train.jsonl":
        tok_stats[x["i"]] = x["prompt_tokens"]
idx = {id(r): i for i, r in enumerate(sft)}
out = []
for (sid, lv), ra in dec.items():
    if lv != "L2" or by_id[sid]["twin"] != "A_should_ask":
        continue
    rb = dec.get((sid[:-1] + "B", lv))
    if rb is None:
        continue
    toks = 0.0
    for ma, mb in zip(ra["prompt"], rb["prompt"]):
        a, b = ma["content"].replace(hint, ""), mb["content"].replace(hint, "")
        if a == b:
            continue
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                seg_a, seg_b = a[i1:i2], b[j1:j2]
                toks += max(est(seg_a), est(seg_b), 1.0)
    T = tok_stats[idx[id(ra)]]
    out.append({"pair": sid[:-1], "diff_tokens_est": round(toks, 1), "prompt_tokens": T, "share": toks / T})
d = np.array([x["diff_tokens_est"] for x in out])
sh = np.array([x["share"] for x in out])
res = {"n_pairs": len(out), "diff_tokens_median": float(np.median(d)), "diff_tokens_p10_p90": np.percentile(d, [10, 90]).round(1).tolist(),
       "share_median_pct": round(float(np.median(sh)) * 100, 3)}
dump("v05b_diff_size.json", {"summary": res, "pairs": out})
print(res)
