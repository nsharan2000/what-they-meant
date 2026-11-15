#!/usr/bin/env python3
"""grpo-dynamics lens, step 9: (a) are the level / prompt-setting / env differences in M1's pass rate real? (b) can a rerun
build pair-balanced, length-matched steps from grpo_train.jsonl? (c) the candidate pool for a rerun.

(a) OLS per twin of the group pass rate on level + prompt_setting + env dummies, group-bootstrap 95% CIs; and the A-minus-B
    ask-rate gap (M1's twin discrimination) per level with bootstrap CIs.
(b) (pair_id, level, prompt_setting) cells holding both twins; length difference of the two twins (exact Tiny Aya tokens).
(c) pool sizes after removing (i) zero-prefix rows, (ii) the 120 prompts already used (if a rerun wants unseen prompts), and
    the expected share of 'mid' prompts (0 < pass < 1) from the sampled ones.
Output: results/cells_and_pairs.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/grpo_train.jsonl"))]
toks = {r["i"]: r["prompt_tokens"] for r in (json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))
        if r["file"] == "grpo_train.jsonl"}
rng = np.random.default_rng(2)
R = {}


def boot_ols(sub, ycol="pass_top", nboot=3000):
    X = pd.get_dummies(sub[["level", "prompt_setting", "env"]], drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    y = sub[ycol].to_numpy(float)
    b = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    bs = []
    for _ in range(nboot):
        i = rng.choice(len(sub), len(sub))
        Xi = X.to_numpy()[i]
        if np.linalg.matrix_rank(Xi) == Xi.shape[1]:
            bs.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0])
    bs = np.array(bs)
    return {c: {"coef": round(b[j], 3), "ci95": [round(np.quantile(bs[:, j], 0.025), 3), round(np.quantile(bs[:, j], 0.975), 3)]}
            for j, c in enumerate(X.columns)}

R["ols_pass"] = {tw: boot_ols(G[G.twin == tw]) for tw in ("A", "B")}


def r2(sub, cols, ycol="pass_top"):
    X = pd.get_dummies(sub[cols], drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    y = sub[ycol].to_numpy(float)
    b = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    res = y - X.to_numpy() @ b
    n, k = X.shape
    r2_ = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"r2": round(r2_, 4), "adj_r2": round(1 - (1 - r2_) * (n - 1) / (n - k), 4), "n": int(n), "k": int(k)}

R["variance_explained"] = {}
for tw in ("A", "B"):
    sub = G[G.twin == tw]
    p = sub["pass_top"].to_numpy()
    between = p.var(ddof=1)
    binom = p.mean() * (1 - p.mean()) / 8
    R["variance_explained"][tw] = {"level+setting+env": r2(sub, ["level", "prompt_setting", "env"]),
                                   "level_only": r2(sub, ["level"]), "setting_only": r2(sub, ["prompt_setting"]),
                                   "var_between_prompts": round(between, 4), "binomial_noise_var": round(binom, 4),
                                   "prompt_specific_share": round((between - binom) / between, 4)}
R["baseline_note"] = "const = L0r, P0, bills; coefficients are differences in the per-prompt pass rate"

gap = {}
for lv in ("L0r", "L1", "L2", "L3", "all"):
    sub = G if lv == "all" else G[G.level == lv]
    a = sub[sub.twin == "A"].p_ask.to_numpy()
    b = sub[sub.twin == "B"].p_ask.to_numpy()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(4000)]
    gap[lv] = {"gap": round(a.mean() - b.mean(), 3), "ci95": [round(np.quantile(bs, 0.025), 3), round(np.quantile(bs, 0.975), 3)],
               "n_A": int(len(a)), "n_B": int(len(b))}
R["ask_gap_A_minus_B"] = gap
# the same gap as a sensitivity index d' from pooled sample rates (z(hit) - z(false alarm)), via a numpy erf-free probit
def probit(p):
    # Acklam-style rational approximation is overkill; use bisection on the logistic-normal CDF via np.erf substitute
    from math import erf, sqrt
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
S = pd.read_csv(os.path.join(OUT, "samples.csv"))
hit = (S[S.twin == "A"].dclass == "ask").mean()
fa = (S[S.twin == "B"].dclass == "ask").mean()
R["d_prime_ask"] = {"hit_rate_A": round(hit, 4), "false_alarm_rate_B": round(fa, 4), "d_prime": round(probit(hit) - probit(fa), 3)}

# ---- (b) pair cells ----
cells = {}
for i, r in enumerate(grpo):
    cells.setdefault((r["pair_id"], r["level"], r["prompt_setting"]), {})[r["twin"][0]] = i
both = {k: v for k, v in cells.items() if "A" in v and "B" in v}
dl = np.array([abs(toks[v["A"]] - toks[v["B"]]) for v in both.values()])
R["pair_cells"] = {"n_cells": len(cells), "n_cells_with_both_twins": len(both), "rows_in_complete_cells": 2 * len(both),
                   "twin_token_diff_mean": round(dl.mean(), 1), "twin_token_diff_p95": float(np.quantile(dl, 0.95)),
                   "twin_token_diff_max": int(dl.max()),
                   "complete_cells_by_level": pd.Series([k[1] for k in both]).value_counts().to_dict()}
lv_both = pd.Series([k[1] for k in both]).value_counts()
R["pair_cells"]["two_pairs_per_step_steps_available"] = int(len(both) // 2)

cells2 = {}
for i, r in enumerate(grpo):
    cells2.setdefault((r["pair_id"], r["level"]), {}).setdefault(r["twin"][0], []).append(r["prompt_setting"])
both2 = [k for k, v in cells2.items() if "A" in v and "B" in v]
R["pair_level_cells"] = {"n_cells": len(cells2), "n_with_both_twins": len(both2),
                         "n_both_twins_same_prompt_setting": int(sum(1 for k in both2 if set(cells2[k]["A"]) & set(cells2[k]["B"]))),
                         "twin_x_setting_counts": {f"{a}|{b}": int(v) for (a, b), v in
                                                   pd.Series([(r["twin"][0], r["prompt_setting"]) for r in grpo]).value_counts().items()}}

# ---- (c) pools ----
zero = {i for i, r in enumerate(grpo) if len(r["prompt"]) == 2}
used = set(G.row_i)
R["pools"] = {"all": len(grpo), "minus_zero_prefix": len(grpo) - len(zero), "minus_used_120": len(grpo) - len(used),
              "minus_both": len(set(range(len(grpo))) - zero - used),
              "complete_pair_cells_minus_zero_prefix": int(sum(1 for k, v in both.items() if v["A"] not in zero)),
              "mid_share_sampled": round(((G.pass_top > 0) & (G.pass_top < 1)).mean(), 4),
              "pass0_share_sampled": round((G.pass_top == 0).mean(), 4), "pass1_share_sampled": round((G.pass_top == 1).mean(), 4)}
json.dump(R, open(os.path.join(OUT, "cells_and_pairs.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_09")
