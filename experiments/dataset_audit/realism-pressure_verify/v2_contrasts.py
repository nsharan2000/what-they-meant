"""VERIFY RP-2/3/4/6/7/9: paired level / hint contrasts of logged model behaviour, recomputed from the raw logs.

Arm = (spec family, model). A contrast pairs the SAME scenario ids of one arm in two cells (level, prompt).
Metrics per twin: A_ask (ask_informed|ask_generic), A_informed (ask_informed|tell_informed), A_literal (approve_generic|
commit_other|commit_harmful); B_approve, B_act (approve*|commit*), B_needless (ask_*|tell_informed).
Significance two ways: scenario bootstrap (the analyst's choice) and source-post cluster bootstrap (variants of one post share a
cluster) -- 10,000 resamples each.  D_ask per pair = ask(A) - ask(B) (chat questions), paired by pair id.
grid_open_seedtools_P0 is left out (different tool-result condition), as the analyst did.
Outputs: results/v2_contrasts.csv, results/v2_dask.csv, results/v2_summary.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from vcommon import OUT, ROOT, PANEL, boot, cluster, cluster_map, dump, episodes, ind

MET = {"A": {"A_ask": "ask", "A_informed": "informed", "A_literal": "literal", "A_success": "succ"},
       "B": {"B_approve": "approve", "B_act": "act", "B_needless": "needless", "B_success": "succ"}}
CON = [(("L0r", "P0"), ("L1", "P0")), (("L0r", "P0"), ("L2", "P0")), (("L0r", "P1"), ("L1", "P1")),
       (("L0r", "P1"), ("L2", "P1")), (("L0r", "P1"), ("L0p", "P1")), (("L2s", "P1"), ("L2", "P1")),
       (("L2s", "P0"), ("L2", "P0")), (("L2s", "P2"), ("L2", "P2")), (("L0r", "P2"), ("L2", "P2")),
       (("L0r", "P2"), ("L1", "P2")), (("L0r", "P0"), ("L0r", "P2")), (("L2", "P0"), ("L2", "P2")),
       (("L1", "P0"), ("L1", "P2")), (("L2s", "P0"), ("L2s", "P2"))]


def frame():
    cm = cluster_map()
    recs = []
    for e in episodes():
        if e["fam"] is None or e["dec"] == "error":
            continue
        d = ind(e["dec"], e["twin"])
        d["succ"] = int(e["succ"])
        recs.append({"fam": e["fam"], "src": e["src"], "model": e["model"], "level": e["level"], "prompt": e["prompt"],
                     "twin": e["twin"], "id": e["id"], "pair": e["pair"], "cluster": cluster(e["pair"], cm),
                     "dec": e["dec"], "line": e["line"], **d})
    df = pd.DataFrame(recs)
    dup = int(df.duplicated(["fam", "model", "level", "prompt", "id"]).sum())
    return df, dup


def contrasts(df):
    out = []
    for (fam, model), g in df.groupby(["fam", "model"]):
        cells = {(lv, pr): x.set_index("id") for (lv, pr), x in g.groupby(["level", "prompt"])}
        for a, b in CON:
            if a not in cells or b not in cells:
                continue
            for tw in ("A", "B"):
                xa, xb = cells[a][cells[a].twin == tw], cells[b][cells[b].twin == tw]
                ids = sorted(set(xa.index) & set(xb.index))
                if len(ids) < 10:
                    continue
                for mname, col in MET[tw].items():
                    va, vb = xa.loc[ids, col].to_numpy(float), xb.loc[ids, col].to_numpy(float)
                    d = vb - va
                    _, lo, hi = boot(d, None, seed=101)
                    _, clo, chi = boot(d, xa.loc[ids, "cluster"].to_numpy(), seed=102)
                    out.append({"fam": fam, "model": model, "from": "/".join(a), "to": "/".join(b), "metric": mname,
                                "n": len(ids), "from_rate": round(va.mean(), 4), "to_rate": round(vb.mean(), 4),
                                "diff": round(d.mean(), 4), "ci": [round(lo, 4), round(hi, 4)],
                                "cl_ci": [round(clo, 4), round(chi, 4)], "sig": bool(lo > 0 or hi < 0),
                                "sig_cl": bool(clo > 0 or chi < 0)})
    return pd.DataFrame(out)


def pair_table(df):
    a = df[df.twin == "A"].set_index(["fam", "model", "level", "prompt", "pair"])
    b = df[df.twin == "B"].set_index(["fam", "model", "level", "prompt", "pair"])
    idx = a.index.intersection(b.index)
    t = pd.DataFrame(index=idx)
    t["cluster"] = a.loc[idx, "cluster"].to_numpy()
    t["D_ask"] = a.loc[idx, "ask"].to_numpy() - b.loc[idx, "ask"].to_numpy()
    t["D_inf"] = a.loc[idx, "informed"].to_numpy() - b.loc[idx, "needless"].to_numpy()
    t["PP"] = a.loc[idx, "succ"].to_numpy() * b.loc[idx, "succ"].to_numpy()
    return t.reset_index()


def dask_contrasts(pt):
    out = []
    for (fam, model), g in pt.groupby(["fam", "model"]):
        cells = {(lv, pr): x.set_index("pair") for (lv, pr), x in g.groupby(["level", "prompt"])}
        for a, b in CON:
            if a not in cells or b not in cells:
                continue
            ids = sorted(set(cells[a].index) & set(cells[b].index))
            if len(ids) < 10:
                continue
            for m in ("D_ask", "D_inf", "PP"):
                d = cells[b].loc[ids, m].to_numpy(float) - cells[a].loc[ids, m].to_numpy(float)
                _, lo, hi = boot(d, None, seed=103)
                _, clo, chi = boot(d, cells[a].loc[ids, "cluster"].to_numpy(), seed=104)
                out.append({"fam": fam, "model": model, "from": "/".join(a), "to": "/".join(b), "metric": m, "n_pairs": len(ids),
                            "from_val": round(float(cells[a].loc[ids, m].mean()), 4),
                            "to_val": round(float(cells[b].loc[ids, m].mean()), 4), "diff": round(float(d.mean()), 4),
                            "ci": [round(lo, 4), round(hi, 4)], "cl_ci": [round(clo, 4), round(chi, 4)],
                            "sig": bool(lo > 0 or hi < 0), "sig_cl": bool(clo > 0 or chi < 0)})
    return pd.DataFrame(out)


def direction(ct, frm, to, metrics=("A_literal", "A_ask", "A_informed", "B_act", "B_approve", "B_needless")):
    res = {}
    for m in metrics:
        x = ct[(ct["from"] == frm) & (ct["to"] == to) & (ct.metric == m)]
        if x.empty:
            continue
        up = x["ci"].apply(lambda c: c[0] > 0)
        dn = x["ci"].apply(lambda c: c[1] < 0)
        upc = x["cl_ci"].apply(lambda c: c[0] > 0)
        dnc = x["cl_ci"].apply(lambda c: c[1] < 0)
        res[m] = {"n_arms": int(len(x)), "sig_up": int(up.sum()), "sig_down": int(dn.sum()),
                  "sig_up_cluster": int(upc.sum()), "sig_down_cluster": int(dnc.sum()),
                  "median_diff_all_arms": round(float(x["diff"].median()), 4),
                  "median_diff_sig_arms": round(float(x[up | dn]["diff"].median()), 4) if (up | dn).any() else None,
                  "raw_up": int((x["diff"] > 0).sum()), "raw_down": int((x["diff"] < 0).sum())}
    return res


def pick(ct, fam, model, frm, to, metric):
    x = ct[(ct.fam == fam) & (ct.model == model) & (ct["from"] == frm) & (ct["to"] == to) & (ct.metric == metric)]
    return x.iloc[0].to_dict() if len(x) else None


def main():
    df, dup = frame()
    ct = contrasts(df)
    ct.to_csv(os.path.join(OUT, "v2_contrasts.csv"), index=False)
    pt = pair_table(df)
    dc = dask_contrasts(pt)
    dc.to_csv(os.path.join(OUT, "v2_dask.csv"), index=False)
    s = {"n_episodes": int(len(df)), "duplicate_rows_within_family": dup}
    # ---- RP-2 / RP-7
    s["RP2_L0rP0_to_L1P0"] = direction(ct, "L0r/P0", "L1/P0")
    s["RP2_L0rP0_to_L2P0"] = direction(ct, "L0r/P0", "L2/P0")
    s["RP2_L0rP1_to_L1P1"] = direction(ct, "L0r/P1", "L1/P1")
    s["RP2_L0rP1_to_L2P1"] = direction(ct, "L0r/P1", "L2/P1")
    s["RP2_length_only_L0rP1_to_L0pP1"] = direction(ct, "L0r/P1", "L0p/P1")
    s["RP2_arms_L0rP0_L1P0"] = sorted(set(ct[(ct["from"] == "L0r/P0") & (ct["to"] == "L1/P0")].apply(lambda r: f"{r.fam}:{r.model}", axis=1)))
    s["RP2_examples"] = {
        "aya32b_seed_L1P0_A_literal": pick(ct, "seed_open", "aya-expanse-32b", "L0r/P0", "L1/P0", "A_literal"),
        "aya32b_seed_L1P0_B_approve": pick(ct, "seed_open", "aya-expanse-32b", "L0r/P0", "L1/P0", "B_approve"),
        "deepseek_variants_L2P0_A_ask": pick(ct, "variants", "DeepSeek-V3.2", "L0r/P0", "L2/P0", "A_ask"),
        "deepseek_variants_L2P0_B_needless": pick(ct, "variants", "DeepSeek-V3.2", "L0r/P0", "L2/P0", "B_needless"),
        "deepseek_seed_L1P0_B_approve": pick(ct, "seed_open", "DeepSeek-V3.2", "L0r/P0", "L1/P0", "B_approve")}
    # ---- RP-3: panel_hard (45 hard pairs), L0r/P2 -> L2/P2
    rp3 = {}
    for model in ("claude-opus", "claude-sonnet", "claude-haiku", "claude-fable", "DeepSeek-V3.2", "Qwen3-235B-A22B-Instruct-2507"):
        x = dc[(dc.fam == "panel_hard") & (dc.model == model) & (dc["from"] == "L0r/P2") & (dc["to"] == "L2/P2")]
        rec = {r.metric: {"from": r.from_val, "to": r.to_val, "diff": r["diff"], "ci": r.ci, "cl_ci": r.cl_ci, "n_pairs": r.n_pairs}
               for _, r in x.iterrows()}
        for m in ("B_approve", "B_needless", "A_informed", "A_ask"):
            p = pick(ct, "panel_hard", model, "L0r/P2", "L2/P2", m)
            if p:
                rec[m] = {"from": p["from_rate"], "to": p["to_rate"], "diff": p["diff"], "ci": p["ci"], "cl_ci": p["cl_ci"]}
        rp3[model] = rec
    s["RP3_panel_hard_L0rP2_to_L2P2"] = rp3
    s["RP3_P1_dask"] = dc[(dc.metric == "D_ask") & (dc["from"] == "L0r/P1") & dc["to"].isin(["L1/P1", "L2/P1"]) & dc.sig]\
        [["fam", "model", "from", "to", "from_val", "to_val", "diff", "ci", "cl_ci", "sig_cl"]].to_dict(orient="records")
    s["RP3_opus_seed_P0_L2"] = dc[(dc.fam == "seed_claude") & (dc.model == "claude-opus") & (dc.metric == "D_ask") &
                                  (dc["from"] == "L0r/P0") & (dc["to"] == "L2/P0")].to_dict(orient="records")
    # ---- RP-4: base Tiny Aya Global across every cell (HF, local vLLM, eval_v12)
    ta = df[df.model.isin(["tiny-aya-global", "tiny-aya-global-local"])]
    cells = []
    for (fam, model, lv, pr), g in ta.groupby(["fam", "model", "level", "prompt"]):
        A, B = g[g.twin == "A"], g[g.twin == "B"]
        if len(A) < 10 or len(B) < 10:
            continue
        cells.append({"fam": fam, "src": ",".join(sorted(g.src.unique())), "model": model, "level": lv, "prompt": pr,
                      "nA": len(A), "nB": len(B), "A_literal": A.literal.mean(), "A_informed": A.informed.mean(),
                      "A_ask": A.ask.mean(), "B_approve": B.approve.mean(), "B_act": B.act.mean(), "B_needless": B.needless.mean()})
    tc = pd.DataFrame(cells)
    tpt = pt[pt.model.isin(["tiny-aya-global", "tiny-aya-global-local"])].groupby(["fam", "model", "level", "prompt"]).agg(
        D_ask=("D_ask", "mean"), PP=("PP", "mean"), n=("D_ask", "size")).reset_index()
    s["RP4_tinyaya_cells"] = {"n_cells": int(len(tc)), "sources": sorted(set(",".join(tc.src).split(","))),
                              **{k: [round(float(tc[k].min()), 4), round(float(tc[k].max()), 4)]
                                 for k in ("A_literal", "A_informed", "B_approve", "B_act", "B_needless")},
                              "D_ask_range": [round(float(tpt.D_ask.min()), 4), round(float(tpt.D_ask.max()), 4)],
                              "PP_max": round(float(tpt.PP.max()), 4), "n_pair_cells": int(len(tpt))}
    tc.round(4).to_csv(os.path.join(OUT, "v2_tinyaya_cells.csv"), index=False)
    ev = ct[(ct.fam == "eval_v12")]
    s["RP4_eval_v12_L0r_to_L2"] = ev[(ev["from"].isin(["L0r/P0", "L0r/P2"])) & ev["to"].isin(["L2/P0", "L2/P2"]) &
                                     ev.metric.isin(["A_ask", "B_approve", "A_informed", "A_literal"])]\
        [["from", "to", "metric", "n", "from_rate", "to_rate", "diff", "ci", "cl_ci"]].to_dict(orient="records")
    # ---- RP-6: hint P0 -> P2 for Tiny Aya (eval_v12) at L0r and L2
    s["RP6_eval_v12_hint"] = ev[ev["from"].isin(["L0r/P0", "L2/P0"]) & ev["to"].isin(["L0r/P2", "L2/P2"])]\
        [["from", "to", "metric", "n", "from_rate", "to_rate", "diff", "ci", "cl_ci"]].to_dict(orient="records")
    s["RP6_eval_v12_hint_dask"] = dc[(dc.fam == "eval_v12") & dc["from"].isin(["L0r/P0", "L2/P0"]) &
                                     dc["to"].isin(["L0r/P2", "L2/P2"])].to_dict(orient="records")
    # ---- RP-9: L2s -> L2
    s["RP9_L2s_to_L2_sig"] = ct[(ct["from"].str.startswith("L2s")) & (ct["to"].str.startswith("L2/")) & ct.sig &
                                ct.metric.isin(["A_informed", "A_literal", "A_ask", "B_needless"])]\
        [["fam", "model", "from", "to", "metric", "n", "from_rate", "to_rate", "diff", "ci", "cl_ci", "sig_cl"]].to_dict(orient="records")
    s["RP9_L2s_to_L2_dask"] = dc[(dc["from"].str.startswith("L2s")) & (dc.metric == "D_ask")]\
        [["fam", "model", "from", "to", "n_pairs", "from_val", "to_val", "diff", "ci", "cl_ci", "sig", "sig_cl"]].to_dict(orient="records")
    s["RP9_L0p_sig"] = ct[(ct["to"] == "L0p/P1") & (ct.sig | ct.sig_cl)][["fam", "model", "metric", "n", "diff", "ci", "cl_ci"]]\
        .to_dict(orient="records")
    s["RP9_L0p_n_arms"] = int(ct[ct["to"] == "L0p/P1"][["fam", "model"]].drop_duplicates().shape[0])
    dump(s, "v2_summary.json")
    print(json.dumps({k: s[k] for k in ("n_episodes", "duplicate_rows_within_family", "RP2_L0rP0_to_L1P0", "RP2_L0rP0_to_L2P0",
                                        "RP2_L0rP1_to_L1P1", "RP2_length_only_L0rP1_to_L0pP1", "RP2_examples")}, indent=1, default=float))


if __name__ == "__main__":
    main()
