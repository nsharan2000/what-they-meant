#!/usr/bin/env python3
"""grpo-dynamics lens, step 2: is the early -> late reward shift (steps 0-14 vs 15-29) policy drift or prompt mix?

Each of the 120 prompts was sampled in exactly one step (01_join_tables checks), so a per-prompt fixed effect absorbs the step
completely. We therefore test drift four ways:
  (1) composition of the two halves (twin x level x setting x env, B_act share, prompt length);
  (2) permutation test: shuffle the half labels of a twin's groups (sizes kept) -> null distribution of the early-late gap that
      prompt heterogeneity + sampling noise alone produce;
  (3) OLS of the group mean reward / P(approve) on step + prompt features (level, setting, env, B_act), group-bootstrap CI;
  (4) within-scenario contrasts: scenarios sampled in both halves (at different levels), level-adjusted.
Also the direction test: a drift toward approving would raise P(approve) on BOTH twins in the late half.
Input: results/samples.csv, results/groups.csv (from 01_join_tables.py). Output: results/early_late.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
S = pd.read_csv(os.path.join(OUT, "samples.csv"))
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
S["late"] = (S["step"] >= 15).astype(int)
G["late"] = (G["step"] >= 15).astype(int)
rng = np.random.default_rng(0)
R = {}

# ---- reproduce the reported shift (sample means by twin and half) ----
rep = S.groupby(["twin", "late"])["score"].mean().round(4)
R["reported_shift_sample_means"] = {f"{t}|{'late' if l else 'early'}": v for (t, l), v in rep.items()}
R["by_half"] = {}
for (t, l), sub in S.groupby(["twin", "late"]):
    R["by_half"][f"{t}|{'late' if l else 'early'}"] = {
        "n_samples": int(len(sub)), "n_prompts": int(sub["row_i"].nunique()), "reward": round(sub["score"].mean(), 4),
        "pass_top": round(sub["pass_top"].mean(), 4), "p_ask": round((sub["dclass"] == "ask").mean(), 4),
        "p_approve": round((sub["dclass"] == "approve").mean(), 4), "p_lookup": round((sub["dclass"] == "lookup").mean(), 4),
        "decisions": {k: int(v) for k, v in sub["decision"].value_counts().items()}}

# ---- (1) composition ----
comp = {}
for col in ["level", "prompt_setting", "env", "cell"]:
    t = G.groupby(["twin", "late", col]).size().unstack(col, fill_value=0)
    comp[col] = {f"{a}|{'late' if b else 'early'}": {k: int(v) for k, v in r.items()} for (a, b), r in t.iterrows()}
comp["mean_prompt_tokens"] = {f"{a}|{'late' if b else 'early'}": round(v, 1)
                              for (a, b), v in G.groupby(["twin", "late"])["prompt_tokens"].mean().items()}
R["composition"] = comp

# ---- (2) permutation test per twin: group mean rewards, sizes of halves kept ----
def perm(vals, late, n=20000):
    vals, late = np.asarray(vals, float), np.asarray(late, int)
    obs = vals[late == 1].mean() - vals[late == 0].mean()
    k = late.sum()
    d = np.empty(n)
    for i in range(n):
        p = rng.permutation(len(vals))
        d[i] = vals[p[:k]].mean() - vals[p[k:]].mean()
    return {"obs_late_minus_early": round(obs, 4), "null_sd": round(d.std(), 4), "p_two_sided": round(float((np.abs(d) >= abs(obs) - 1e-12).mean()), 4),
            "n_early": int(len(vals) - k), "n_late": int(k)}

R["permutation"] = {}
for t, sub in G.groupby("twin"):
    R["permutation"][t] = {"reward_mean": perm(sub["reward_mean"], sub["late"]), "pass_top": perm(sub["pass_top"], sub["late"]),
                           "p_approve": perm(sub["p_approve"], sub["late"]), "p_ask": perm(sub["p_ask"], sub["late"])}
# one combined test for "policy drifted toward approving": late-early change in P(approve), pooled over both twins
R["permutation"]["both_twins_p_approve"] = perm(G["p_approve"], G["late"])

# ---- (3) OLS with prompt features + step, cluster(group)-bootstrap CI ----
def design(sub, with_step=True):
    X = pd.get_dummies(sub[["level", "prompt_setting", "env"]], drop_first=True).astype(float)
    if sub["twin"].nunique() > 1:
        X["twinB"] = (sub["twin"] == "B").astype(float)
    if (sub["cell"] == "B_act").any():
        X["B_act"] = (sub["cell"] == "B_act").astype(float)
    if with_step:
        X["step"] = sub["step"].astype(float)
    X.insert(0, "const", 1.0)
    return X

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def ols_step(sub, ycol, nboot=4000):
    X = design(sub)
    y = sub[ycol].to_numpy(float)
    b = ols(X.to_numpy(), y)
    j = list(X.columns).index("step")
    bs = []
    idx = np.arange(len(sub))
    for _ in range(nboot):
        s = rng.choice(idx, len(idx), replace=True)
        Xs = X.to_numpy()[s]
        if np.linalg.matrix_rank(Xs) < Xs.shape[1]:
            continue
        bs.append(ols(Xs, y[s])[j])
    bs = np.array(bs)
    # raw (unadjusted) slope for comparison
    Xr = np.c_[np.ones(len(sub)), sub["step"].to_numpy(float)]
    braw = ols(Xr, y)[1]
    return {"slope_per_step_adjusted": round(b[j], 5), "ci95": [round(np.quantile(bs, 0.025), 5), round(np.quantile(bs, 0.975), 5)],
            "slope_x30_adjusted": round(30 * b[j], 4), "slope_per_step_raw": round(braw, 5), "slope_x30_raw": round(30 * braw, 4),
            "n_groups": int(len(sub)), "n_boot_ok": int(len(bs)), "features": list(X.columns)}

R["ols_step"] = {}
for t, sub in G.groupby("twin"):
    R["ols_step"][t] = {y: ols_step(sub, y) for y in ("reward_mean", "pass_top", "p_approve")}
R["ols_step"]["pooled_p_approve"] = ols_step(G, "p_approve")

# mix-only expectation: fit features WITHOUT step on all groups, then average the fitted values per half
R["mix_only_prediction"] = {}
for t, sub in G.groupby("twin"):
    X = design(sub, with_step=False)
    for ycol in ("reward_mean", "p_approve"):
        y = sub[ycol].to_numpy(float)
        fit = X.to_numpy() @ ols(X.to_numpy(), y)
        e = fit[sub["late"].to_numpy() == 0].mean()
        l = fit[sub["late"].to_numpy() == 1].mean()
        oe = y[sub["late"].to_numpy() == 0].mean()
        ol = y[sub["late"].to_numpy() == 1].mean()
        R["mix_only_prediction"][f"{t}|{ycol}"] = {"observed_early": round(oe, 4), "observed_late": round(ol, 4),
                                                    "observed_gap": round(ol - oe, 4), "mix_predicted_gap": round(l - e, 4),
                                                    "share_explained_by_mix": round((l - e) / (ol - oe), 3) if abs(ol - oe) > 1e-9 else None}

# ---- (4) within-scenario contrasts (same scenario sampled early and late, at different levels) ----
lvl_eff = G.groupby(["twin", "level"])["reward_mean"].mean()
G["reward_adj"] = G["reward_mean"] - G.apply(lambda r: lvl_eff[(r["twin"], r["level"])], axis=1)
w = []
for sid, sub in G.groupby("scenario_id"):
    if sub["late"].nunique() == 2:
        e = sub[sub.late == 0]
        l = sub[sub.late == 1]
        w.append({"scenario_id": sid, "twin": sub["twin"].iloc[0], "early_levels": "+".join(e["level"]),
                  "late_levels": "+".join(l["level"]), "early_steps": "+".join(map(str, e["step"])), "late_steps": "+".join(map(str, l["step"])),
                  "early_reward": round(e["reward_mean"].mean(), 3), "late_reward": round(l["reward_mean"].mean(), 3),
                  "diff_adj": round(l["reward_adj"].mean() - e["reward_adj"].mean(), 3),
                  "early_p_approve": round(e["p_approve"].mean(), 3), "late_p_approve": round(l["p_approve"].mean(), 3)})
W = pd.DataFrame(w)
R["within_scenario"] = {"n_scenarios": int(len(W)), "rows": w}
if len(W):
    d = W["diff_adj"].to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    R["within_scenario"]["mean_diff_adj"] = round(d.mean(), 4)
    R["within_scenario"]["ci95"] = [round(np.quantile(bs, 0.025), 4), round(np.quantile(bs, 0.975), 4)]
    for t, sub in W.groupby("twin"):
        R["within_scenario"][f"mean_diff_adj_{t}"] = round(sub["diff_adj"].mean(), 4)
        R["within_scenario"][f"n_{t}"] = int(len(sub))

# ---- the steps with the largest drop: which prompts? ----
step_tab = G.groupby("step").apply(lambda g: pd.Series({
    "reward": round(g["reward_mean"].mean(), 3),
    "prompts": "; ".join(f"{r.scenario_id}/{r.level}/{r.prompt_setting}/{r.env}: pass {r.pass_top:.3f}" for r in g.itertuples())}),
    include_groups=False)
R["per_step"] = step_tab.reset_index().to_dict("records")
# the lowest-reward A groups and which half they fell in
R["lowest_A_groups"] = G[G.twin == "A"].nsmallest(12, "reward_mean")[["step", "scenario_id", "level", "prompt_setting", "env",
                                                                          "reward_mean", "pass_top", "classes"]].to_dict("records")
R["lowest_A_late_share"] = round(float((G[G.twin == "A"].nsmallest(12, "reward_mean")["late"]).mean()), 3)
# ---- the per-step reward curve: is it more variable than random 4-prompt draws from a fixed policy would make it? ----
step_r = G.groupby("step")["reward_mean"].mean().to_numpy()
gm = G["reward_mean"].to_numpy()
boot = np.array([rng.choice(gm, 4, replace=False).mean() for _ in range(20000)])
steps = np.arange(len(step_r))
def slope(y):
    return np.polyfit(steps, y, 1)[0]
obs_slope = slope(step_r)
null = np.array([slope(rng.permutation(step_r)) for _ in range(20000)])
R["step_curve"] = {"sd_observed_step_means": round(float(step_r.std(ddof=1)), 4),
                   "sd_random_4_group_draws": round(float(boot.std(ddof=1)), 4),
                   "slope_per_step": round(float(obs_slope), 5),
                   "slope_perm_p_two_sided": round(float((np.abs(null) >= abs(obs_slope)).mean()), 4),
                   "first10_last10_mean": [round(float(step_r[:10].mean()), 4), round(float(step_r[-10:].mean()), 4)]}
json.dump(R, open(os.path.join(OUT, "early_late.json"), "w"), indent=1, default=str)
print("step_curve", R["step_curve"])
for k in ("reported_shift_sample_means", "by_half", "composition", "permutation", "ols_step", "mix_only_prediction"):
    print(k, json.dumps(R[k], indent=0, default=str)[:3500])
print("within", json.dumps({k: v for k, v in R["within_scenario"].items() if k != "rows"}, default=str))
print(pd.DataFrame(R["within_scenario"]["rows"]).to_string())
print(pd.DataFrame(R["lowest_A_groups"]).to_string())
print("OK_02")
