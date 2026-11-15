#!/usr/bin/env python3
"""grpo-dynamics lens, step 8: decompose the twin-A early -> late drop into identifiable prompt properties vs residual drift.

M1-specific prompt property found in step 2/3: 16 GRPO rows (S02A/B, S03A/B x 4 levels) are decision prompts with NO prior
lookup ([system, user] only). In sft_train, the same [system, user] context is a 'lookup1' row 401 times and a 'decision' row 16
times, so M1's first-turn prior is 'look something up' -> on these prompts M1 looks up instead of deciding (external models
do not share this prior, so 07's difficulty score cannot see it).
Tests (twin A groups):
  - permutation test of the late-early gap with and without the zero-prefix prompts;
  - OLS: pass_top ~ ext_strong_all + zero_prefix + late (group bootstrap CI);
  - decision mix early vs late after removing zero-prefix prompts;
  - lookup counts by prompt (is the late lookup surge concentrated in a few prompts, or broad as a drift would be?).
Output: results/drift_decomposition.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
G = pd.read_csv(os.path.join(OUT, "groups_ext.csv"))
S = pd.read_csv(os.path.join(OUT, "samples.csv"))
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/grpo_train.jsonl"))]
sft = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/sft_train.jsonl"))]
G["late"] = (G.step >= 15).astype(int)
G["zero_prefix"] = G.row_i.map(lambda i: int(len(grpo[i]["prompt"]) == 2))
S["late"] = (S.step >= 15).astype(int)
S["zero_prefix"] = S.row_i.map(lambda i: int(len(grpo[i]["prompt"]) == 2))
rng = np.random.default_rng(1)
R = {}
R["sft_first_turn_targets"] = pd.Series([r.get("turn") for r in sft if len(r["prompt"]) == 2]).value_counts().to_dict()
R["grpo_zero_prefix_rows"] = int(sum(len(r["prompt"]) == 2 for r in grpo))
R["zero_prefix_sampled"] = G[G.zero_prefix == 1][["step", "scenario_id", "level", "prompt_setting", "pass_top", "p_lookup",
                                                  "classes"]].to_dict("records")


def perm(vals, late, n=20000):
    vals, late = np.asarray(vals, float), np.asarray(late, int)
    obs = vals[late == 1].mean() - vals[late == 0].mean()
    k = late.sum()
    d = np.array([(lambda p: vals[p[:k]].mean() - vals[p[k:]].mean())(rng.permutation(len(vals))) for _ in range(n)])
    return {"obs": round(obs, 4), "p_two_sided": round(float((np.abs(d) >= abs(obs) - 1e-12).mean()), 4), "n": int(len(vals))}

A = G[G.twin == "A"]
R["perm_A"] = {"all": {"reward": perm(A.reward_mean, A.late), "pass": perm(A.pass_top, A.late)},
               "without_zero_prefix": {"reward": perm(A[A.zero_prefix == 0].reward_mean, A[A.zero_prefix == 0].late),
                                       "pass": perm(A[A.zero_prefix == 0].pass_top, A[A.zero_prefix == 0].late)}}
# lookup-prone prompts more generally (>= 4 of 8 lookups), whatever the cause
lk = A.p_lookup >= 0.5
R["perm_A"]["without_lookup_majority_prompts"] = {"reward": perm(A[~lk].reward_mean, A[~lk].late), "pass": perm(A[~lk].pass_top, A[~lk].late),
                                                  "removed": A[lk][["scenario_id", "level", "step"]].to_dict("records")}


def ols(sub, cols, ycol="pass_top", nboot=4000):
    X = np.c_[np.ones(len(sub)), sub[cols].to_numpy(float)]
    y = sub[ycol].to_numpy(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = []
    for _ in range(nboot):
        i = rng.choice(len(sub), len(sub))
        if np.linalg.matrix_rank(X[i]) == X.shape[1]:
            bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    bs = np.array(bs)
    return {c: {"coef": round(b[j + 1], 4), "ci95": [round(np.quantile(bs[:, j + 1], 0.025), 4), round(np.quantile(bs[:, j + 1], 0.975), 4)]}
            for j, c in enumerate(cols)}

R["ols_A_pass"] = {"late_only": ols(A, ["late"]), "ext+late": ols(A, ["ext_strong_all", "late"]),
                   "ext+zero_prefix+late": ols(A, ["ext_strong_all", "zero_prefix", "late"])}
R["ols_A_reward"] = {"ext+zero_prefix+late": ols(A, ["ext_strong_all", "zero_prefix", "late"], ycol="reward_mean")}
SA = S[(S.twin == "A") & (S.zero_prefix == 0)]
R["A_decisions_without_zero_prefix"] = {("late" if l else "early"): {"n": int(len(sub)), "prompts": int(sub.row_i.nunique()),
                                        "pass_top": round(sub.pass_top.mean(), 4), "reward": round(sub.score.mean(), 4),
                                        **{k: int(v) for k, v in sub.decision.value_counts().items()}}
                                        for l, sub in SA.groupby("late")}
lk_by_prompt = S[S.decision == "lookup"].groupby(["twin", "scenario_id", "level", "step"]).size().sort_values(ascending=False)
R["lookups_by_prompt"] = [{"twin": a, "scenario_id": b, "level": c, "step": int(d), "n": int(v)} for (a, b, c, d), v in lk_by_prompt.items()]
R["lookup_concentration"] = {"total": int((S.decision == "lookup").sum()),
                             "in_prompts_with_ge4": int(lk_by_prompt[lk_by_prompt >= 4].sum()),
                             "n_prompts_with_any": int(len(lk_by_prompt)), "n_prompts_with_ge4": int((lk_by_prompt >= 4).sum())}
# ext difficulty of A prompts by half: distribution, not just the mean
R["A_ext_strong_by_half"] = {("late" if l else "early"): {"mean": round(sub.ext_strong_all.mean(), 3),
                                                          "n_le_0.15": int((sub.ext_strong_all <= 0.15).sum()),
                                                          "n": int(len(sub))} for l, sub in A.groupby("late")}
json.dump(R, open(os.path.join(OUT, "drift_decomposition.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_08")
