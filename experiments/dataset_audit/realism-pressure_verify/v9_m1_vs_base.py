"""VERIFY RP-5's matched-scenario 'SFT lift' (M1 T=1 samples on v1.2.1 decision-point views vs base Tiny Aya greedy episodes
from the replication grid on frozen v1 specs, same scenario id + level + prompt setting), and size its confounds:
share of matched rows whose tool_rules changed between frozen v1 and the as-trained spec; base episodes that never reached
the scripted look-up the M1 prompt contains.
Outputs: results/v9_m1_vs_base.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from vcommon import CUR, GRID, OUT, ROOT, boot, dump, ind, load_jsonl


def main():
    pp = pd.read_csv(os.path.join(OUT, "v3_m1_prompts.csv"))
    grpo = load_jsonl(os.path.join(CUR, "grpo_train.jsonl"))
    trained = {r["scenario_id"]: json.loads(r["scenario"]) for r in grpo}
    frozen = {}
    for f in glob.glob(os.path.join(ROOT, "dataset", "frozen_*_v1", "*.jsonl")):
        for s in load_jsonl(f):
            frozen[s["id"]] = s
    base = {}
    for f, model in (("grid_local_tinyaya.jsonl", "vllm:tiny-aya-global-local@http://localhost:8001/v1"),
                     ("grid_open_variants.jsonl", "hf:CohereLabs/tiny-aya-global:cohere")):
        for i, r in enumerate(load_jsonl(os.path.join(GRID, f))):
            if r["model"] == model and r["decision_grade"]["decision"] != "error":
                base[(r["id"], r["level"], r["prompt"])] = (f, i, r)
    rows = []
    for _, p in pp.iterrows():
        k = (p.sid, p.level, p.ps)
        if k not in base:
            continue
        f, i, r = base[k]
        tw = p.twin
        b = ind(r["decision_grade"]["decision"], tw)
        prefix = [c["tool"] for c in trained[p.sid].get("prefix_calls", [])]
        base_tools = [e.get("tool") for e in r["events"] if e.get("kind") == "tool_call"]
        rules_changed = (p.sid in frozen and json.dumps(frozen[p.sid].get("tool_rules"), sort_keys=True) !=
                         json.dumps(trained[p.sid].get("tool_rules"), sort_keys=True))
        rows.append({"sid": p.sid, "level": p.level, "ps": p.ps, "twin": tw, "base_ref": f"{f}:{i}",
                     "base_succ": int(r["decision_grade"]["success"]), "m1_succ": p.succ,
                     "base_needless": b.get("needless", np.nan), "m1_needless": p.needless if tw == "B" else np.nan,
                     "base_called_prefix_tool": int(any(t in base_tools for t in prefix)) if prefix else np.nan,
                     "rules_changed": bool(rules_changed)})
    df = pd.DataFrame(rows)
    out = {"n_matched": int(len(df)), "share_rules_changed": round(float(df.rules_changed.mean()), 4),
           "share_base_called_prefix_tool": round(float(df.base_called_prefix_tool.mean()), 4)}
    res = []
    for (lv, tw), g in list(df.groupby(["level", "twin"])) + [(("all", t), df[df.twin == t]) for t in ("A", "B")]:
        d = (g.m1_succ - g.base_succ).to_numpy(float)
        mu, lo, hi = boot(d, None, seed=91)
        rec = {"level": lv, "twin": tw, "n": int(len(g)), "base_succ": round(g.base_succ.mean(), 4),
               "m1_succ": round(g.m1_succ.mean(), 4), "lift": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)]}
        if tw == "B":
            dn = (g.m1_needless - g.base_needless).to_numpy(float)
            mu2, lo2, hi2 = boot(dn, None, seed=92)
            rec.update({"base_needless": round(g.base_needless.mean(), 4), "m1_needless": round(g.m1_needless.mean(), 4),
                        "needless_diff": round(mu2, 4), "needless_ci": [round(lo2, 4), round(hi2, 4)]})
        res.append(rec)
    out["lift"] = res
    dump(out, "v9_m1_vs_base.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
