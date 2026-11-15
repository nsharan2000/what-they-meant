#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 12: tests the analyst's prompt-interference inference that 'the long system prompt is shared
context, not the carrier of the decision'.

For every GRPO row: ID/number-like tokens of the scenario's decisive_facts (regex: tokens containing a digit, length >= 3,
or e-mail/handle-like tokens) are located in the rendered prompt. A token is 'system-only' if it occurs in the system
message but in none of the later turns (member request, look-up call, look-up result). For system-only tokens the distance
of their LAST occurrence from the end of the prompt is converted to tokens with the row's own chars/token ratio (exact
token count from token_stats.jsonl). Share of rows whose decisive evidence sits only in the system prompt, and only beyond
the 4,096-token sliding window (visible to the 9 global layers only), by level; M1's pass/ask on the sampled A prompts for
'beyond-window' vs other rows (descriptive; small n).
Outputs: results/v12_reference_position.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl"))]
tok = {t["i"]: t["prompt_tokens"] for t in (json.loads(l) for l in open(os.path.join(
    ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))) if t["file"] == "grpo_train.jsonl"}
G = pd.read_csv(os.path.join(OUT, "v01_groups.csv"))
TOK = re.compile(r"[A-Za-z0-9@._\-]*\d[A-Za-z0-9@._\-]*|[A-Za-z0-9._\-]+@[A-Za-z0-9.\-]+")
rows = []
for i, r in enumerate(grpo):
    scn = json.loads(r["scenario"])
    facts = " ".join(scn.get("decisive_facts") or [])
    cands = sorted({t.strip(".,-") for t in TOK.findall(facts) if len(t.strip(".,-")) >= 3})
    sys_txt = r["prompt"][0]["content"] if r["prompt"][0]["role"] == "system" else ""
    turns = " ".join(m["content"] for m in r["prompt"][1:])
    full = "\x00".join(m["content"] for m in r["prompt"])
    cpt = len(full) / tok[i]
    sys_only, dists = [], []
    for c in cands:
        if c in sys_txt and c not in turns:
            sys_only.append(c)
            dists.append((len(full) - full.rfind(c)) / cpt)
    rows.append({"grpo_i": i, "twin": r["twin"][0], "level": r["level"], "n_cand": len(cands), "n_sys_only": len(sys_only),
                 "n_in_turns": sum(1 for c in cands if c in turns), "min_sysonly_tokens_from_end": min(dists) if dists else np.nan,
                 "tokens": tok[i], "example": sys_only[:3]})
D = pd.DataFrame(rows)
D["has_sys_only"] = D.n_sys_only > 0
D["sys_only_beyond_4096"] = D.min_sysonly_tokens_from_end > 4096
R = {"rows_with_candidates": int((D.n_cand > 0).sum()), "rows": int(len(D))}
R["share_rows_with_system_only_decisive_token"] = {t: round(float(s.has_sys_only.mean()), 4) for t, s in D.groupby("twin")}
R["share_A_rows_system_only_by_level"] = {lv: round(float(s.has_sys_only.mean()), 4)
                                          for lv, s in D[D.twin == "A"].groupby("level")}
R["share_A_rows_system_only_beyond_4096_by_level"] = {lv: round(float(s.sys_only_beyond_4096.mean()), 4)
                                                      for lv, s in D[D.twin == "A"].groupby("level")}
R["A_examples_beyond_4096"] = D[(D.twin == "A") & D.sys_only_beyond_4096].head(5)[
    ["grpo_i", "level", "min_sysonly_tokens_from_end", "example"]].to_dict("records")
M = G.merge(D[["grpo_i", "has_sys_only", "sys_only_beyond_4096"]], on="grpo_i")
A = M[M.twin == "A"]
R["sampled_A_by_beyond_window"] = {str(k): {"n": int(len(s)), "pass": round(float(s.pass_gold.mean()), 4),
                                            "p_ask": round(float(s.p_ask.mean()), 4)} for k, s in A.groupby("sys_only_beyond_4096")}
R["sampled_A_by_has_sys_only"] = {str(k): {"n": int(len(s)), "pass": round(float(s.pass_gold.mean()), 4),
                                           "p_ask": round(float(s.p_ask.mean()), 4)} for k, s in A.groupby("has_sys_only")}
D.to_csv(os.path.join(OUT, "v12_reference_position.csv"), index=False)
json.dump(R, open(os.path.join(OUT, "v12_reference_position.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_v12")
