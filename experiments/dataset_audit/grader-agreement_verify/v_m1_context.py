"""Prompt-side context for M1's samples (for the prompt-interference question, grader lens): top-class rate by prompt length
(<= 4096 vs > 4096 tokens: Tiny Aya's sliding window covers the last 4096 tokens in 3 of 4 layers) and by where a keyword of
the scenario is visible (look-up result at the end of the prompt / system prompt only / nowhere).
Token counts from experiments/dataset_audit/results/token_stats.jsonl (file grpo_train.jsonl, line i).
-> results/v_m1_context_summary.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from vcommon import ROOT, SCORES, SP, dump, jl, rows

g = rows("grpo_train")
tok = {r["i"]: r["prompt_tokens"] for r in jl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))
       if r["file"] == "grpo_train.jsonl"}
idx = {(r["scenario_id"], r["level"]): i for i, r in enumerate(g)}
feat = {}
for i, r in enumerate(g):
    scn = json.loads(r["scenario"])
    K = scn["expected"].get("informed_any", [])
    p = r["prompt"]
    res = "\n".join(m["content"] for m in p[2:] if m["role"] == "user")
    where = "results" if any(SP.kw_hit(res, k) for k in K) else ("system_only" if any(SP.kw_hit(p[0]["content"], k) for k in K) else "nowhere")
    feat[i] = {"long": tok[i] > 4096, "where": where, "tw": "A" if scn["twin"].startswith("A") else "B",
               "act": scn["expected"]["decision"] == "act"}


def top(c, f):
    if f["tw"] == "A":
        return c == "ask_informed"
    return c == "commit_gold" if f["act"] else c.startswith("approve")


agg = defaultdict(lambda: [0, 0])
for s in jl(SCORES):
    m = s["meta"]
    i = idx[(m["scenario_id"], m["level"])]
    f = feat[i]
    t = top(s["components"]["decision"], f)
    for key in (f"{f['tw']}|long={f['long']}", f"{f['tw']}|kw_where={f['where']}", f"{f['tw']}|long={f['long']}|kw_where={f['where']}"):
        agg[key][0] += t
        agg[key][1] += 1
S = {k: v + [round(v[0] / v[1], 3)] for k, v in sorted(agg.items())}
S["grpo_rows_long"] = sum(f["long"] for f in feat.values())
dump(S, "v_m1_context_summary.json")
print(json.dumps(S, indent=1))
