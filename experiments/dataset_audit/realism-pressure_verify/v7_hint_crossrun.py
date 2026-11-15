"""VERIFY RP-6 (big models) + the render-check claim: L2/P0 (replication grid, 2026-11-12, v1.1 specs) vs L2/P2 (hard-set
panel, 2026-11-13, v1.2 specs) for the same model and scenario id.

Checks beyond the analyst's: besides system_chars (P2 must be exactly len('\\n\\n' + hint) longer), compare the first user
message and the n_tools of each paired episode; count pairs whose spec changed in v1.2/v1.2.1 (excluded). Claude runs use
default sampling, so a cross-run difference also contains run-to-run noise: quantify it where a same-cell repeat exists
(grid_claude L2/P0 vs panel? none) -> instead report the share of pairs that flip in each direction.
Outputs: results/v7_hint_crossrun.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from vcommon import GRID, OUT, PANEL, ROOT, boot, cluster, cluster_map, dump, ind, load_jsonl, mname

HINT = "\n\n" + open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()


def main():
    changed = {c["row"] for f in ("fix_v12_report.json", "fix_v121_report.json")
               for c in json.load(open(os.path.join(ROOT, "dataset", f)))["changes"]}
    grid = {}
    for f in ("grid_open_P0.jsonl", "grid_claude.jsonl", "grid_open_variants.jsonl"):
        for i, r in enumerate(load_jsonl(os.path.join(GRID, f))):
            if r["level"] == "L2" and r["prompt"] == "P0":
                grid[(r["model"], r["id"])] = (f, i, r)
    cm = cluster_map()
    recs = []
    checks = {"pairs": 0, "len_diff_eq_hint": 0, "same_user_msg": 0, "same_n_tools": 0, "excluded_changed": 0}
    for f in ("panel_mined_claude.jsonl", "panel_mined_open.jsonl"):
        for i, p in enumerate(load_jsonl(os.path.join(PANEL, f))):
            tw = {"A_should_ask": "A", "B_should_act": "B"}.get(p["twin"])
            if tw is None or p["decision_grade"]["decision"] == "error":
                continue
            g = grid.get((p["model"], p["id"]))
            if g is None or g[2]["decision_grade"]["decision"] == "error":
                continue
            if p["id"] in changed:
                checks["excluded_changed"] += 1
                continue
            gf, gi, gr = g
            checks["pairs"] += 1
            checks["len_diff_eq_hint"] += int(p["system_chars"] - gr["system_chars"] == len(HINT))
            checks["same_user_msg"] += int(p["messages"][0]["content"] == gr["messages"][0]["content"])
            checks["same_n_tools"] += int(p["n_tools"] == gr["n_tools"])
            a, b = ind(gr["decision_grade"]["decision"], tw), ind(p["decision_grade"]["decision"], tw)
            rec = {"model": mname(p["model"]), "id": p["id"], "twin": tw, "cluster": cluster(p["pair_id"], cm),
                   "from_ref": f"experiments/replication/results/{gf}:{gi}", "to_ref": f"{PANEL}/{f}:{i}"}
            for k in a:
                rec[k + "_P0"], rec[k + "_P2"] = a[k], b[k]
            recs.append(rec)
    df = pd.DataFrame(recs)
    res = []
    for (model, tw), g in df.groupby(["model", "twin"]):
        for k in (["ask", "informed", "literal"] if tw == "A" else ["approve", "needless", "act"]):
            d = (g[k + "_P2"] - g[k + "_P0"]).to_numpy(float)
            mu, lo, hi = boot(d, None, seed=71)
            _, clo, chi = boot(d, g.cluster.to_numpy(), seed=72)
            res.append({"model": model, "twin": tw, "metric": k, "n": int(len(g)), "P0": round(g[k + "_P0"].mean(), 4),
                        "P2": round(g[k + "_P2"].mean(), 4), "diff": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)],
                        "cl_ci": [round(clo, 4), round(chi, 4)], "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum())})
    out = {"hint_chars": len(HINT), "checks": checks, "contrasts": res}
    dump(out, "v7_hint_crossrun.json")
    print(json.dumps(checks))
    for r in res:
        print(f"{r['model']:32s} {r['twin']} {r['metric']:9s} n={r['n']:3d} {r['P0']:.2f}->{r['P2']:.2f} {r['diff']:+.2f} "
              f"[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] cl[{r['cl_ci'][0]:+.2f},{r['cl_ci'][1]:+.2f}] up {r['n_up']} down {r['n_down']}")


if __name__ == "__main__":
    main()
