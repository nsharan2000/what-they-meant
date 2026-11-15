"""Parts 2 + 4: paired level / prompt contrasts ("prompt pressure") with bootstrap CIs over scenarios.

A contrast compares two cells (level, prompt) of the SAME model on the SAME scenario ids within one spec family
(rows rendered from the same scenario-spec version). For each metric the per-scenario difference d_i = x_i(to) - x_i(from)
(0/1 indicators) is averaged; CIs: (a) scenario bootstrap (10,000 resamples of scenarios) and (b) source-post cluster bootstrap
(10,000 resamples of clusters; twins and variants of one Reddit post share a cluster). n_up / n_down = scenarios that switch
the indicator on / off. Metrics: A_ask, A_informed (ask_informed + tell_informed), A_literal (approve_generic + commit_other +
commit_harmful), A_success; B_approve, B_act (approve or commit), B_needless (ask_* + tell_informed), B_success.
Outputs: results/contrasts.csv, results/contrasts.json, results/pressure_headline.json
Run: python3 -B experiments/dataset_audit/realism-pressure/pressure.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import RES, boot_mean_ci, dump, indicators, load_episodes

FAMILY = {   # source -> spec family (same scenario specs / tool results within a family)
    "grid_open_P0": "seed_v11_open", "grid_open_P1": "seed_v11_open", "grid_open_controls": "seed_v11_open",
    "grid_local_tinyaya": "seed_v11_local", "grid_spark": "seed_v11_spark", "grid_claude": "seed_v11_claude",
    "grid_open_variants": "variants_v1", "eval_base_v12": "eval_v12",
    "panel_hard_claude": "panel_hard_v12", "panel_hard_open": "panel_hard_v12",
    "panel_mined_claude": "panel_mined_v12", "panel_mined_open": "panel_mined_v12", "panel_mined_fable": "panel_mined_v12",
    # grid_open_seedtools_P0 (seed-text tool results at every level, partial Earth coverage) is left out on purpose
}
METRICS = {"A": ["A_ask", "A_informed", "A_literal", "A_success"], "B": ["B_approve", "B_act", "B_needless", "B_success"]}
COL = {"A_ask": "ask", "A_informed": "informed_chat", "A_literal": "literal_act", "A_success": "success",
       "B_approve": "approve", "B_act": "act", "B_needless": "needless", "B_success": "success"}

CONTRASTS = [  # (name, from_cell, to_cell)
    ("toy->production (L0r->L1)", ("L0r", "P0"), ("L1", "P0")),
    ("toy->buried (L0r->L2)", ("L0r", "P0"), ("L2", "P0")),
    ("production->buried (L1->L2)", ("L1", "P0"), ("L2", "P0")),
    ("salient->buried (L2s->L2)", ("L2s", "P0"), ("L2", "P0")),
    ("toy->production (L0r->L1)", ("L0r", "P1"), ("L1", "P1")),
    ("toy->buried (L0r->L2)", ("L0r", "P1"), ("L2", "P1")),
    ("production->buried (L1->L2)", ("L1", "P1"), ("L2", "P1")),
    ("salient->buried (L2s->L2)", ("L2s", "P1"), ("L2", "P1")),
    ("length only (L0r->L0p)", ("L0r", "P1"), ("L0p", "P1")),
    ("noise (L2->L3)", ("L2", "P1"), ("L3", "P1")),
    ("toy->production (L0r->L1)", ("L0r", "P2"), ("L1", "P2")),
    ("toy->buried (L0r->L2)", ("L0r", "P2"), ("L2", "P2")),
    ("production->buried (L1->L2)", ("L1", "P2"), ("L2", "P2")),
    ("salient->buried (L2s->L2)", ("L2s", "P2"), ("L2", "P2")),
    ("hint P0->P1 @L0r", ("L0r", "P0"), ("L0r", "P1")),
    ("hint P0->P1 @L1", ("L1", "P0"), ("L1", "P1")),
    ("hint P0->P1 @L2", ("L2", "P0"), ("L2", "P1")),
    ("hint P0->P2 @L0r", ("L0r", "P0"), ("L0r", "P2")),
    ("hint P0->P2 @L1", ("L1", "P0"), ("L1", "P2")),
    ("hint P0->P2 @L2", ("L2", "P0"), ("L2", "P2")),
    ("hint P0->P2 @L2s", ("L2s", "P0"), ("L2s", "P2")),
]


def build_frame() -> pd.DataFrame:
    recs = []
    for r in load_episodes():
        fam = FAMILY.get(r["source"])
        if fam is None or r["decision"] == "error" or r.get("rep", 0) != 0:
            continue
        ind = indicators(r["decision"], r["twin"], r["success"])
        rec = {"family": fam, "source": r["source"], "model": r["model"], "level": r["level"], "prompt": r["prompt"],
               "twin": r["twin"], "id": r["id"], "cluster": r["cluster"], "decision": r["decision"],
               "file": r["file"], "line": r["line"]}
        for m in METRICS[r["twin"]]:
            rec[m] = ind[COL[m]]
        recs.append(rec)
    df = pd.DataFrame(recs)
    # one row per (family, model, level, prompt, id): duplicates across files of one family would be a bug
    dup = df.duplicated(["family", "model", "level", "prompt", "id"]).sum()
    assert dup == 0, f"{dup} duplicate episode rows inside a family"
    return df


def run_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (fam, model), g in df.groupby(["family", "model"]):
        cells = {(lv, pr): x.set_index("id") for (lv, pr), x in g.groupby(["level", "prompt"])}
        for name, a, b in CONTRASTS:
            if a not in cells or b not in cells:
                continue
            for tw in ("A", "B"):
                xa = cells[a][cells[a].twin == tw]
                xb = cells[b][cells[b].twin == tw]
                ids = sorted(set(xa.index) & set(xb.index))
                if len(ids) < 10:
                    continue
                for m in METRICS[tw]:
                    va = xa.loc[ids, m].to_numpy(float)
                    vb = xb.loc[ids, m].to_numpy(float)
                    d = vb - va
                    _, lo, hi = boot_mean_ci(d, None, seed=1)
                    _, clo, chi = boot_mean_ci(d, xa.loc[ids, "cluster"].to_numpy(), seed=2)
                    out.append({"family": fam, "model": model, "contrast": name, "from": f"{a[0]}/{a[1]}",
                                "to": f"{b[0]}/{b[1]}", "metric": m, "n": len(ids),
                                "n_clusters": int(xa.loc[ids, "cluster"].nunique()),
                                "rate_from": round(va.mean(), 4), "rate_to": round(vb.mean(), 4),
                                "diff": round(d.mean(), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                "cl_ci_lo": round(clo, 4), "cl_ci_hi": round(chi, 4),
                                "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum()),
                                "sig_scen": bool(lo > 0 or hi < 0), "sig_cluster": bool(clo > 0 or chi < 0)})
    return pd.DataFrame(out)


def headline(ct: pd.DataFrame) -> dict:
    """Item 4: A-ask and B-approve shifts toy (L0r) -> production (L1) / buried (L2), under P0 and P2 (and P1 as reference)."""
    h = {}
    sel = ct[ct.contrast.str.startswith("toy->") & ct.metric.isin(["A_ask", "A_informed", "A_literal", "B_approve",
                                                                    "B_act", "B_needless"])]
    for _, r in sel.iterrows():
        key = f"{r.family}|{r.model}|{r['from']}->{r['to']}"
        h.setdefault(key, {})[r.metric] = {"diff": r["diff"], "ci": [r.ci_lo, r.ci_hi], "cluster_ci": [r.cl_ci_lo, r.cl_ci_hi],
                                           "from": r.rate_from, "to": r.rate_to, "n": int(r.n)}
    return h


def summarize_direction(ct: pd.DataFrame) -> dict:
    """Across models: how many show a significant (scenario CI) shift toward asking vs toward acting, per contrast x prompt."""
    res = {}
    for (name, frm, to), g in ct.groupby(["contrast", "from", "to"]):
        d = {}
        for m in ("A_ask", "A_informed", "A_literal", "B_approve", "B_act", "B_needless"):
            x = g[g.metric == m]
            if x.empty:
                continue
            d[m] = {"n_models": int(len(x)), "sig_up": int(((x.ci_lo > 0)).sum()), "sig_down": int(((x.ci_hi < 0)).sum()),
                    "median_diff": round(float(x["diff"].median()), 4),
                    "models_sig_up": sorted(f"{a}:{b}" for a, b in zip(x[x.ci_lo > 0].family, x[x.ci_lo > 0].model)),
                    "models_sig_down": sorted(f"{a}:{b}" for a, b in zip(x[x.ci_hi < 0].family, x[x.ci_hi < 0].model))}
        res[f"{name} [{frm} -> {to}]"] = d
    return res


def main():
    df = build_frame()
    df.to_csv(os.path.join(RES, "episode_indicators.csv"), index=False)
    ct = run_contrasts(df)
    ct.to_csv(os.path.join(RES, "contrasts.csv"), index=False)
    dump(ct.to_dict(orient="records"), "contrasts.json")
    dump({"headline_toy_to_realistic": headline(ct), "direction_across_models": summarize_direction(ct)},
         "pressure_headline.json")
    print(f"episodes {len(df)} contrasts {len(ct)} -> results/contrasts.csv, results/pressure_headline.json")


if __name__ == "__main__":
    main()
