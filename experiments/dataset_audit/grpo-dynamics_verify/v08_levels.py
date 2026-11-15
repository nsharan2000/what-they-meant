#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 8: GD-8 (does realism level / the P2 hint / env explain M1's per-prompt pass on its
training prompts?) and the statistical power behind 'no realism gap'.

Per twin: OLS of the per-prompt gold rate on level + prompt setting + env dummies (numpy), adjusted R^2, bootstrap CIs
(resampling prompts) for the L2-vs-L0r, L3-vs-L0r and P2-vs-P0 coefficients; A-B ask-rate gap by level with bootstrap CIs;
over-dispersion (share of between-prompt variance beyond binomial noise of 8 samples); and the smallest level effect the
design could detect (2 x bootstrap SE).
Outputs: results/v08_levels.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
G = pd.read_csv(os.path.join(OUT, "v01_groups.csv"))
R = {}
rng = np.random.default_rng(0)


def design(df):
    X = pd.get_dummies(df[["level", "ps", "env"]].astype(str), drop_first=False)
    keep = [c for c in X.columns if c not in ("level_L0r", "ps_P0") and not c.startswith("env_")]
    envs = sorted(c for c in X.columns if c.startswith("env_"))
    keep += envs[1:]   # first env is the reference
    return X[keep].astype(float), envs[0]


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ b
    n, k = X1.shape
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return b, r2, adj


for t in ("A", "B"):
    df = G[G.twin == t].reset_index(drop=True)
    X, ref_env = design(df)
    y = df["pass_gold"].to_numpy()
    b, r2, adj = ols(X.to_numpy(), y)
    names = ["const"] + list(X.columns)
    coefs = dict(zip(names, b))
    boots = {n: [] for n in ("level_L2", "level_L3", "level_L1", "ps_P2")}
    for _ in range(4000):
        idx = rng.integers(0, len(df), len(df))
        Xb, yb = X.to_numpy()[idx], y[idx]
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(yb)), Xb])) < Xb.shape[1] + 1:
            continue
        bb, *_ = ols(Xb, yb)
        for n in boots:
            if n in names:
                boots[n].append(bb[names.index(n)])
    R[t] = {"n_prompts": int(len(df)), "R2": round(float(r2), 4), "adj_R2": round(float(adj), 4), "ref_env": ref_env,
            "coef": {n: round(float(coefs[n]), 4) for n in boots if n in coefs},
            "ci95": {n: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)] for n, v in boots.items() if v},
            "boot_se": {n: round(float(np.std(v, ddof=1)), 3) for n, v in boots.items() if v},
            "n_by_level": df["level"].value_counts().to_dict(), "n_by_ps": df["ps"].value_counts().to_dict()}
    # over-dispersion: between-prompt variance of pass vs binomial expectation
    v_obs = float(np.var(y, ddof=1))
    v_bin = float(np.mean(y * (1 - y)) * 8 / 7 / 8)   # unbiased p(1-p)/n per prompt, averaged
    R[t]["share_var_beyond_binomial"] = round((v_obs - v_bin) / v_obs, 4)

# A - B ask-rate gap by level
gaps = {}
for lv in ("L0r", "L1", "L2", "L3"):
    a = G[(G.twin == "A") & (G.level == lv)]["p_ask"].to_numpy()
    bb = G[(G.twin == "B") & (G.level == lv)]["p_ask"].to_numpy()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(bb, len(bb)).mean() for _ in range(4000)]
    gaps[lv] = {"gap": round(float(a.mean() - bb.mean()), 4), "ci95": [round(float(np.quantile(bs, .025)), 3),
                                                                    round(float(np.quantile(bs, .975)), 3)],
                "nA": int(len(a)), "nB": int(len(bb))}
R["ask_gap_by_level"] = gaps
for ps in ("P0", "P2"):
    a = G[(G.twin == "A") & (G.ps == ps)]["p_ask"].to_numpy()
    bb = G[(G.twin == "B") & (G.ps == ps)]["p_ask"].to_numpy()
    R[f"ask_gap_{ps}"] = {"gap": round(float(a.mean() - bb.mean()), 4), "nA": int(len(a)), "nB": int(len(bb))}
R["min_detectable_level_effect_A_2se"] = {n: round(2 * v, 3) for n, v in R["A"]["boot_se"].items()}
json.dump(R, open(os.path.join(OUT, "v08_levels.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
print("OK_v08")
