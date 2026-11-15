"""Part 3b: base Tiny Aya Global vs the M1 (SFT) policy on the SAME scenario ids at the same level and prompt setting.

M1: per-prompt class rates from results/m1_samples.csv (8 samples at T=1 per training decision prompt).
Base: the replication grid's first decision for the same (scenario id, level, prompt): grid_local_tinyaya (local vLLM, seed
pairs, v1.1 specs) and grid_open_variants (HF provider, variant pairs, open-variant-v1 specs); both greedy (T=0).
Caveats: the base episodes start from the member's message and the model does its own look-ups (decision mode), while the M1
prompts are decision-point views with the look-up already done; specs differ in version (v1.1 / v1 vs v1.2.1).
Matched-prompt bootstrap (10,000 resamples of matched prompts) for the M1 - base lift per level.
Outputs: results/m1_vs_base.csv, results/m1_vs_base.json
Run: python3 -B experiments/dataset_audit/realism-pressure/m1_vs_base.py
"""
from __future__ import annotations

import os

import pandas as pd

from common import RES, boot_mean_ci, dump

MET = {"A": ["ask", "informed_chat", "literal_act", "success"], "B": ["approve", "act", "needless", "success"]}


def main():
    m1 = pd.read_csv(os.path.join(RES, "m1_samples.csv"))
    ep = pd.read_csv(os.path.join(RES, "episode_indicators.csv"))
    base = ep[((ep.family == "seed_v11_local") & (ep.model == "tiny-aya-global-local")) |
              ((ep.family == "variants_v1") & (ep.model == "tiny-aya-global"))]
    base = base.drop_duplicates(["id", "level", "prompt"])
    base_idx = base.set_index(["id", "level", "prompt"])
    pp = m1.groupby(["grpo_line", "scenario_id", "level", "prompt_setting", "twin"]).mean(numeric_only=True).reset_index()
    rows = []
    for r in pp.itertuples():
        k = (r.scenario_id, r.level, r.prompt_setting)
        if k not in base_idx.index:
            continue
        b = base_idx.loc[k]
        rec = {"scenario_id": r.scenario_id, "level": r.level, "prompt": r.prompt_setting, "twin": r.twin,
               "base_family": b.family, "base_decision": b.decision, "base_ref": f"{b.file}:{b.line}",
               "m1_grpo_line": int(r.grpo_line)}
        for mname in MET[r.twin]:
            col = {"informed_chat": "A_informed", "literal_act": "A_literal", "ask": "A_ask", "approve": "B_approve",
                   "act": "B_act", "needless": "B_needless", "success": f"{r.twin}_success"}[mname]
            rec[f"base_{mname}"] = float(b[col])
            rec[f"m1_{mname}"] = float(getattr(r, mname))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "m1_vs_base.csv"), index=False)
    out = []
    for (lv, tw), g in list(df.groupby(["level", "twin"])) + [(("all", t), df[df.twin == t]) for t in ("A", "B")]:
        for mname in MET[tw]:
            bvals, mvals = g[f"base_{mname}"].to_numpy(), g[f"m1_{mname}"].to_numpy()
            d, lo, hi = boot_mean_ci(mvals - bvals, None, seed=21)
            _, blo, bhi = boot_mean_ci(bvals, None, seed=22)
            _, mlo, mhi = boot_mean_ci(mvals, None, seed=23)
            out.append({"level": lv, "twin": tw, "metric": mname, "n_matched_prompts": int(len(g)),
                        "base": round(float(bvals.mean()), 4), "base_ci": [round(blo, 4), round(bhi, 4)],
                        "m1": round(float(mvals.mean()), 4), "m1_ci": [round(mlo, 4), round(mhi, 4)],
                        "lift": round(d, 4), "lift_ci": [round(lo, 4), round(hi, 4)]})
    dump({"n_matched_prompts": int(len(df)), "by_family": df.base_family.value_counts().to_dict(), "rows": out},
         "m1_vs_base.json")
    for r in out:
        print(f"{r['level']:4s} {r['twin']} {r['metric']:14s} n={r['n_matched_prompts']:3d} base {r['base']:.2f} "
              f"m1 {r['m1']:.2f} lift {r['lift']:+.2f} [{r['lift_ci'][0]:+.2f},{r['lift_ci'][1]:+.2f}]")


if __name__ == "__main__":
    main()
