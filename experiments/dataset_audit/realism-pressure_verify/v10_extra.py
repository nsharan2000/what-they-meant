"""Extra checks for RP-3 and RP-5.

RP-3: on the 45 generated hard pairs (panel_hard, L0r/P2 vs L2/P2, Claude), is the B-side rise in needless questions concentrated
in pairs whose realistic rendering carries a world-state defect -- scenario_records already in the member file (duplicated lines
at L2, RP-8) or a decisive transaction that exists only in tool_rules (missing from the L2 history; results/v6_world_specs.csv)?
RP-5: a PAIRED level contrast for M1 on the scenarios sampled at L0r and at a realistic level (the analyst's level contrasts
are unpaired across different scenarios).
Outputs: results/v10_extra.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from vcommon import OUT, PANEL, ROOT, boot, dump, ind, load_jsonl, mname


def main():
    out = {}
    # ---- spec defects of the hard pairs
    specs = {}
    for f in glob.glob(os.path.join(ROOT, "dataset", "eval", "hard_*.jsonl")):
        for s in load_jsonl(f):
            specs[s["id"]] = s
    members = {}
    dup = {}
    for sid, s in specs.items():
        k = (s["env_id"], s["member_id"])
        if k not in members:
            members[k] = json.load(open(os.path.join(ROOT, "envs", s["env_id"], "members", f"{s['member_id']}.json")))
        m = members[k]
        d = False
        for x in s.get("scenario_records") or []:
            ex, rec = m.get(x["field"], []), x["record"]
            if (isinstance(rec, dict) and rec.get("id") and any(isinstance(e, dict) and e.get("id") == rec["id"] for e in ex)) \
                    or rec in ex:
                d = True
        dup[sid] = d
    ws = pd.read_csv(os.path.join(OUT, "v6_world_specs.csv"))
    missing = set(ws[(ws.missing_in_L1_window > 0)].sid)
    rows = []
    for i, r in enumerate(load_jsonl(os.path.join(PANEL, "panel_hard_claude.jsonl"))):
        if r["twin"] != "B_should_act" or r["decision_grade"]["decision"] == "error":
            continue
        rows.append({"model": mname(r["model"]), "id": r["id"], "level": r["level"],
                     "needless": ind(r["decision_grade"]["decision"], "B")["needless"],
                     "defect_dup": dup.get(r["id"], False), "defect_missing": r["id"] in missing})
    df = pd.DataFrame(rows)
    df["defect_any"] = df.defect_dup | df.defect_missing
    piv = df.pivot_table(index=["model", "id", "defect_any", "defect_dup"], columns="level", values="needless").reset_index()
    piv["d"] = piv["L2"] - piv["L0r"]
    res = []
    for (model, de), g in piv.groupby(["model", "defect_any"]):
        mu, lo, hi = boot(g.d.to_numpy(float), None, seed=101)
        res.append({"model": model, "defect_any": bool(de), "n_B": int(len(g)), "needless_L0r": round(g.L0r.mean(), 4),
                    "needless_L2": round(g.L2.mean(), 4), "diff": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)]})
    out["panel_hard_B_needless_by_defect"] = res
    out["hard_B_specs_with_defect"] = {"n_B_ids": int(piv.id.nunique()),
                                       "with_dup": int(piv.drop_duplicates("id").defect_dup.sum()),
                                       "with_any_defect": int(piv.drop_duplicates("id").defect_any.sum())}
    # pooled over the four Claude models
    pooled = []
    for de, g in piv.groupby("defect_any"):
        mu, lo, hi = boot(g.d.to_numpy(float), g.id.to_numpy(), seed=102)
        pooled.append({"defect_any": bool(de), "n": int(len(g)), "diff": round(mu, 4), "ci_cluster_by_id": [round(lo, 4), round(hi, 4)]})
    out["panel_hard_B_needless_by_defect_pooled_claude"] = pooled

    # ---- RP-5 paired level contrast within M1
    pp = pd.read_csv(os.path.join(OUT, "v3_m1_prompts.csv"))
    paired = []
    for tw in ("A", "B"):
        sub = pp[pp.twin == tw]
        base = sub[sub.level == "L0r"].groupby("sid").succ.mean()
        real = sub[sub.level.isin(["L1", "L2", "L3"])].groupby("sid").succ.mean()
        ids = sorted(set(base.index) & set(real.index))
        d = (real.loc[ids] - base.loc[ids]).to_numpy(float)
        mu, lo, hi = boot(d, None, seed=103) if len(d) else (float("nan"),) * 3
        paired.append({"twin": tw, "n_scenarios": len(ids), "succ_L0r": round(float(base.loc[ids].mean()), 4) if ids else None,
                       "succ_realistic": round(float(real.loc[ids].mean()), 4) if ids else None,
                       "diff": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)]})
    out["M1_paired_L0r_vs_realistic"] = paired
    dump(out, "v10_extra.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
