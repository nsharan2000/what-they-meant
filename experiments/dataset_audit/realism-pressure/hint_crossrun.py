"""Part 2 (P2 for the big models): does the one-line P2 hint restore asking at the realistic level L2, and at what cost on B?

The hard-set panel ran the mined pool (93 seed + 178 variant scenarios) at L2 under P2 only; the replication grid ran the same
scenario ids at L2 under P0 (and P1). Cross-run pairing by (model, scenario id). Rendering check: for every paired row the system
prompt is exactly len("\\n\\n" + golden_short) = 192 characters longer under P2 (verified below), i.e. only the hint differs in
the system prompt. Rows whose spec changed in the v1.2 / v1.2.1 fixes (dataset/fix_v12_report.json, fix_v121_report.json) are
excluded. Caveats: different run dates (2026-11-12 grid vs 2026-11-13 panel); Claude runs use default sampling (not greedy), so
part of any difference is run-to-run noise (no repeat runs exist to size it).
Outputs: results/hint_crossrun.csv, results/hint_crossrun.json
Run: python3 -B experiments/dataset_audit/realism-pressure/hint_crossrun.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import (GRID_DIR, PANEL_DIR, RES, ROOT, boot_mean_ci, cluster_of, dump, indicators, pair_cluster_map,
                    short_model, twin_letter)

HINT_CHARS = len("\n\n" + open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip())


def rows(path):
    out = []
    with open(path) as fh:
        for i, l in enumerate(fh):
            r = json.loads(l)
            r["_file"], r["_line"] = os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, i
            out.append(r)
    return out


def main():
    changed = {c["row"] for c in json.load(open(os.path.join(ROOT, "dataset", "fix_v12_report.json")))["changes"]}
    changed |= {c["row"] for c in json.load(open(os.path.join(ROOT, "dataset", "fix_v121_report.json")))["changes"]}
    p2c = pair_cluster_map()
    grid = {}
    for f in ("grid_open_P0.jsonl", "grid_open_P1.jsonl", "grid_claude.jsonl", "grid_open_variants.jsonl"):
        for r in rows(os.path.join(GRID_DIR, f)):
            if r["level"] == "L2":
                grid[(r["model"], r["id"], r["prompt"])] = r
    panel = rows(os.path.join(PANEL_DIR, "panel_mined_open.jsonl")) + rows(os.path.join(PANEL_DIR, "panel_mined_claude.jsonl"))
    recs, char_check = [], {"pairs": 0, "exact_hint_diff": 0}
    for p in panel:
        tw = twin_letter(p["twin"])
        if tw is None or p["id"] in changed or p["decision_grade"]["decision"] == "error":
            continue
        for base_prompt in ("P0", "P1"):
            g = grid.get((p["model"], p["id"], base_prompt))
            if g is None or g["decision_grade"]["decision"] == "error":
                continue
            if base_prompt == "P0":
                char_check["pairs"] += 1
                char_check["exact_hint_diff"] += int(p["system_chars"] - g["system_chars"] == HINT_CHARS)
            ia = indicators(g["decision_grade"]["decision"], tw, g["decision_grade"]["success"])
            ib = indicators(p["decision_grade"]["decision"], tw, p["decision_grade"]["success"])
            rec = {"model": short_model(p["model"]), "id": p["id"], "twin": tw, "cluster": cluster_of(p["pair_id"], p2c),
                   "from_prompt": base_prompt, "dec_from": g["decision_grade"]["decision"],
                   "dec_to": p["decision_grade"]["decision"], "from_ref": f"{g['_file']}:{g['_line']}",
                   "to_ref": f"{p['_file']}:{p['_line']}"}
            keys = ["ask", "informed_chat", "literal_act", "success"] if tw == "A" else ["approve", "act", "needless", "success"]
            for k in keys:
                rec[f"{k}_from"], rec[f"{k}_to"] = ia[k], ib[k]
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RES, "hint_crossrun_rows.csv"), index=False)
    out = []
    for (model, tw, bp), g in df.groupby(["model", "twin", "from_prompt"]):
        keys = ["ask", "informed_chat", "literal_act", "success"] if tw == "A" else ["approve", "act", "needless", "success"]
        for k in keys:
            d = (g[f"{k}_to"] - g[f"{k}_from"]).to_numpy(float)
            mu, lo, hi = boot_mean_ci(d, None, seed=11)
            _, clo, chi = boot_mean_ci(d, g.cluster.to_numpy(), seed=12)
            out.append({"model": model, "twin": tw, "contrast": f"L2: {bp} -> P2", "metric": k, "n": int(len(g)),
                        "rate_from": round(float(g[f"{k}_from"].mean()), 4), "rate_to": round(float(g[f"{k}_to"].mean()), 4),
                        "diff": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)], "cluster_ci": [round(clo, 4), round(chi, 4)],
                        "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum())})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(RES, "hint_crossrun.csv"), index=False)
    dump({"render_check_P0_vs_P2": char_check, "hint_chars": HINT_CHARS, "excluded_changed_rows": sorted(changed),
          "contrasts": out}, "hint_crossrun.json")
    print("render check", char_check)
    for _, r in res.iterrows():
        print(f"{r.model:30s} {r.twin} {r.contrast:12s} {r.metric:14s} n={r.n:3d} {r.rate_from:.2f}->{r.rate_to:.2f} "
              f"{r['diff']:+.2f} [{r.ci[0]:+.2f},{r.ci[1]:+.2f}] cl[{r.cluster_ci[0]:+.2f},{r.cluster_ci[1]:+.2f}]")


if __name__ == "__main__":
    main()
