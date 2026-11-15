#!/usr/bin/env python3
"""grpo-dynamics lens, step 7: an independent difficulty measure for the 120 GRPO prompts, to separate prompt mix from drift.

Other models' decision-mode results on the same scenario ids (replication grids of 2026-11-12/13; single greedy-or-default
runs, older renderings and labels -> a noisy but policy-independent difficulty signal):
  strong   = Qwen3-235B-A22B, DeepSeek-V3.2, Aya Expanse 32B, Claude Opus/Sonnet/Haiku (grid_open_P0/P1/variants/controls, grid_claude)
  base_aya = Tiny Aya Global (HF provider + local vLLM: grid_open_P0/P1/variants/controls/seedtools, grid_local_tinyaya)
Per scenario: mean decision_grade.success over every available model x level x prompt row ('ext_all'), and at the same level
when present ('ext_level'). Then:
  - Spearman correlation of M1's per-prompt pass rate with the external success (is M1's failure prompt-intrinsic?);
  - external success of the A / B prompts sampled early (steps 0-14) vs late (15-29): did the late half draw harder prompts?
  - OLS of M1 group pass rate on external success + late, per twin (does 'late' survive the difficulty control?).
Output: results/external_difficulty.json, results/groups_ext.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
G["late"] = (G.step >= 15).astype(int)
sids = set(G.scenario_id)
STRONG = ("Qwen3-235B", "DeepSeek-V3.2", "aya-expanse-32b", "claude:")
recs = []
for f in sorted(glob.glob(os.path.join(ROOT, "experiments/replication/results/grid_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r["id"] not in sids or "decision_grade" not in r:
            continue
        m = r["model"]
        grp = "strong" if any(s in m for s in STRONG) else "base_aya" if "tiny-aya-global" in m else "other"
        recs.append({"file": os.path.basename(f), "scenario_id": r["id"], "level": r["level"], "prompt": r["prompt"], "model": m,
                     "grp": grp, "success": float(bool(r["decision_grade"]["success"]))})
E = pd.DataFrame(recs)
R = {"n_records": int(len(E)), "by_grp": E.groupby("grp").agg(n=("success", "size"), scen=("scenario_id", "nunique"),
                                                             success=("success", "mean")).round(3).to_dict("index")}


def rank(x):
    return pd.Series(x).rank().to_numpy()


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1]), int(ok.sum())


for grp in ("strong", "base_aya"):
    sub = E[E.grp == grp]
    allm = sub.groupby("scenario_id")["success"].mean()
    lvl = sub.groupby(["scenario_id", "level"])["success"].mean()
    G[f"ext_{grp}_all"] = G.scenario_id.map(allm)
    G[f"ext_{grp}_level"] = [lvl.get((s, l), np.nan) for s, l in zip(G.scenario_id, G.level)]
G.to_csv(os.path.join(OUT, "groups_ext.csv"), index=False)

R["coverage"] = {c: int(G[c].notna().sum()) for c in G.columns if c.startswith("ext_")}
R["spearman_M1pass_vs_ext"] = {}
for c in [c for c in G.columns if c.startswith("ext_")]:
    for tw in ("A", "B"):
        sub = G[G.twin == tw]
        rho, n = spearman(sub.pass_top, sub[c])
        R["spearman_M1pass_vs_ext"][f"{c}|{tw}"] = {"rho": round(rho, 3), "n": n}

R["early_late_ext"] = {}
for c in ("ext_strong_all", "ext_base_aya_all", "ext_strong_level", "ext_base_aya_level"):
    for tw in ("A", "B"):
        sub = G[(G.twin == tw) & G[c].notna()]
        e, l = sub[sub.late == 0][c].mean(), sub[sub.late == 1][c].mean()
        R["early_late_ext"][f"{c}|{tw}"] = {"early": round(e, 3), "late": round(l, 3), "late_minus_early": round(l - e, 3),
                                            "n_early": int((sub.late == 0).sum()), "n_late": int((sub.late == 1).sum())}

rng = np.random.default_rng(0)


def ols_late(sub, xcol, ycol="pass_top", nboot=4000):
    sub = sub[sub[xcol].notna()]
    X = np.c_[np.ones(len(sub)), sub[xcol].to_numpy(float), sub.late.to_numpy(float)]
    y = sub[ycol].to_numpy(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = []
    for _ in range(nboot):
        i = rng.choice(len(sub), len(sub))
        if np.linalg.matrix_rank(X[i]) == 3:
            bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    bs = np.array(bs)
    raw = y[sub.late.to_numpy() == 1].mean() - y[sub.late.to_numpy() == 0].mean()
    return {"n": int(len(sub)), "raw_late_minus_early": round(raw, 4), "coef_ext": round(b[1], 4),
            "coef_ext_ci95": [round(np.quantile(bs[:, 1], q), 4) for q in (0.025, 0.975)],
            "coef_late_adjusted": round(b[2], 4), "coef_late_ci95": [round(np.quantile(bs[:, 2], q), 4) for q in (0.025, 0.975)]}

R["ols_pass_on_ext_and_late"] = {}
for tw in ("A", "B"):
    for c in ("ext_strong_all", "ext_base_aya_all"):
        R["ols_pass_on_ext_and_late"][f"{tw}|{c}"] = ols_late(G[G.twin == tw], c)
# the late-A failures: their external success
A_late_bad = G[(G.twin == "A") & (G.late == 1)].nsmallest(10, "pass_top")
R["late_A_worst_with_ext"] = A_late_bad[["step", "scenario_id", "level", "prompt_setting", "pass_top", "ext_strong_all",
                                         "ext_base_aya_all", "ext_strong_level"]].round(3).to_dict("records")
R["A_ext_strong_all_mean"] = round(G[G.twin == "A"].ext_strong_all.mean(), 3)
json.dump(R, open(os.path.join(OUT, "external_difficulty.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_07")
