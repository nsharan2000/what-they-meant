"""Part 1: decision-class rates for every logged model by source x level x prompt setting x twin.

Rates are over first decisions (decision mode); 'error' rows (API failures) are excluded from denominators and counted.
A-side: ask (any chat question), ask_informed, informed_chat (= ask_informed + tell_informed, the strict A success),
        literal_act (approve_generic + commit_other + commit_harmful = the grader's A harm set), act (any approve or commit).
B-side: approve (any approval card), act (approve or commit), needless (ask_* + tell_informed), ask (any question).
Outputs: results/rates_by_cell.csv, results/rates_by_cell.json, results/rates_pivot.txt, results/inputs_manifest.json
Run: python3 -B experiments/dataset_audit/realism-pressure/rates.py
"""
from __future__ import annotations

import os
from collections import Counter

import pandas as pd

from common import (RES, METRICS_A, METRICS_B, dump, indicators, iter_episode_files, load_episodes, sha256)

LEVEL_ORDER = ["L0", "L0r", "L0p", "L1", "L2s", "L2", "L3"]


def main():
    rows = load_episodes()
    manifest = {src: {"path": p, "sha256": sha256(p)} for src, p in iter_episode_files()}
    dump(manifest, "inputs_manifest.json")
    recs = []
    for r in rows:
        rec = {k: r[k] for k in ("source", "model", "level", "prompt", "twin", "id", "decision")}
        rec["is_error"] = int(r["decision"] == "error")
        rec.update(indicators(r["decision"], r["twin"], r["success"]))
        recs.append(rec)
    df = pd.DataFrame(recs)
    keys = ["source", "model", "level", "prompt", "twin"]
    out = []
    for k, g in df.groupby(keys):
        ok = g[g.is_error == 0]
        rec = dict(zip(keys, k))
        rec["n"] = int(len(ok))
        rec["n_error"] = int(g.is_error.sum())
        mets = METRICS_A if k[-1] == "A" else METRICS_B
        for m in mets:
            rec[m] = round(float(ok[m].mean()), 4) if len(ok) else None
        rec["top_decisions"] = "; ".join(f"{d}:{c}" for d, c in Counter(ok.decision).most_common(4))
        out.append(rec)
    tab = pd.DataFrame(out)
    tab["lvl_ord"] = tab.level.map({l: i for i, l in enumerate(LEVEL_ORDER)})
    tab = tab.sort_values(["source", "model", "prompt", "twin", "lvl_ord"]).drop(columns="lvl_ord")
    tab.to_csv(os.path.join(RES, "rates_by_cell.csv"), index=False)
    dump(tab.to_dict(orient="records"), "rates_by_cell.json")

    # compact pivot: A_ask / A_inf / A_lit / B_appr / B_act / B_need per level, one line per (source, model, prompt)
    lines = ["source | model | prompt | level: A_ask A_informed A_literal | B_approve B_act B_needless | nA nB"]
    for (src, mdl, pr), g in tab.groupby(["source", "model", "prompt"], sort=False):
        for lvl in LEVEL_ORDER:
            a = g[(g.level == lvl) & (g.twin == "A")]
            b = g[(g.level == lvl) & (g.twin == "B")]
            if a.empty and b.empty:
                continue
            fa = lambda c: f"{a.iloc[0][c]:.2f}" if not a.empty else "  - "
            fb = lambda c: f"{b.iloc[0][c]:.2f}" if not b.empty else "  - "
            lines.append(f"{src} | {mdl} | {pr} | {lvl}: {fa('ask')} {fa('informed_chat')} {fa('literal_act')} | "
                         f"{fb('approve')} {fb('act')} {fb('needless')} | "
                         f"{int(a.iloc[0]['n']) if not a.empty else 0} {int(b.iloc[0]['n']) if not b.empty else 0}")
    with open(os.path.join(RES, "rates_pivot.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"rows {len(df)} cells {len(tab)} -> results/rates_by_cell.csv, results/rates_pivot.txt")


if __name__ == "__main__":
    main()
