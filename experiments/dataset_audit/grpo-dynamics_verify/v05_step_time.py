#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 5: GD-7 (step time vs the longest prompt in the step).

Step k+1 of progress/result.json step_times (StepClock: generation + scoring + fwd/bwd) <-> scores meta.step k.
Prompt tokens per GRPO row from experiments/dataset_audit/results/token_stats.jsonl (Tiny Aya chat template, exact).
Fits seconds ~ a + b * max_prompt_tokens and seconds ~ a + b * sum_prompt_tokens (and both); padding share; then simulates
random 4-prompt steps vs length-sorted (bucketed) steps over the 496-row pool, with the fitted model floored at the
observed minimum step time (no extrapolation below the data).
Outputs: results/v05_step_time.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs")
S = pd.read_csv(os.path.join(OUT, "v01_samples.csv"))
res = json.load(open(os.path.join(TL, "m2_grpo", "result.json")))
tok = [json.loads(l) for l in open(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))]
gt = {t["i"]: t["prompt_tokens"] for t in tok if t["file"] == "grpo_train.jsonl"}
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl"))]
for i, r in enumerate(grpo):
    t = [x for x in tok if x["file"] == "grpo_train.jsonl" and x["i"] == i][0]
    assert t["scenario_id"] == r["scenario_id"] and t["level"] == r["level"]
R = {}
secs = res["step_times_s"]
trl_secs = [e["step_time"] for e in res["log_history"] if "step_time" in e]
rows = []
for k in range(30):
    gis = sorted(S.loc[S.step == k, "grpo_i"].unique())
    pt = [gt[g] for g in gis]
    comp = S.loc[S.step == k, "n_tokens"].to_numpy()
    rows.append({"log_step": k + 1, "secs": secs[k], "trl_step_time": trl_secs[k], "max_pt": max(pt), "sum_pt": sum(pt),
                 "levels": "+".join(sorted(S.loc[S.step == k].drop_duplicates("grpo_i")["level"])),
                 "pad_share": 1 - sum(pt) / (4 * max(pt)), "max_comp": int(comp.max()), "mean_comp": float(comp.mean())})
T = pd.DataFrame(rows)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2, pred


y = T["secs"].to_numpy()
b1, r2_1, _ = ols(T[["max_pt"]].to_numpy() / 1000, y)
b2, r2_2, _ = ols(T[["sum_pt"]].to_numpy() / 1000, y)
b3, r2_3, _ = ols(T[["max_pt", "sum_pt"]].to_numpy() / 1000, y)
b4, r2_4, _ = ols(T[["max_pt"]].to_numpy() / 1000, T["trl_step_time"].to_numpy())
R["fit_secs_on_max_pt_per1k"] = {"a": round(b1[0], 2), "b": round(b1[1], 3), "R2": round(r2_1, 4)}
R["fit_secs_on_sum_pt_per1k"] = {"a": round(b2[0], 2), "b": round(b2[1], 3), "R2": round(r2_2, 4)}
R["fit_secs_on_max_and_sum"] = {"a": round(b3[0], 2), "b_max": round(b3[1], 3), "b_sum": round(b3[2], 3),
                                "R2": round(r2_3, 4)}
R["fit_trl_step_time_on_max_pt"] = {"a": round(b4[0], 2), "b": round(b4[1], 3), "R2": round(r2_4, 4)}
R["max_pt_range"] = [int(T.max_pt.min()), int(T.max_pt.max())]
R["mean_pad_share"] = round(float(T.pad_share.mean()), 4)
R["steps_max_pt_le_4400"] = {"n": int((T.max_pt <= 4400).sum()), "mean_secs": round(float(T.loc[T.max_pt <= 4400, "secs"].mean()), 1)}
R["steps_max_pt_gt_4400"] = {"n": int((T.max_pt > 4400).sum()), "mean_secs": round(float(T.loc[T.max_pt > 4400, "secs"].mean()), 1)}
R["mean_secs_all"] = round(float(np.mean(secs)), 1)
R["examples"] = T[T.log_step.isin([20, 23, 25])].to_dict("records")
floor = float(T["secs"].min())


def step_secs(maxpt):
    return max(floor, b1[0] + b1[1] * maxpt / 1000)


pool = np.array([gt[i] for i in range(len(grpo))])
rng = np.random.default_rng(0)
rand = [step_secs(rng.choice(pool, 4, replace=False).max()) for _ in range(20000)]
srt = np.sort(pool)
blocks = [step_secs(srt[i:i + 4].max()) for i in range(0, len(srt) - 3, 4)]
R["sim_random_step_secs"] = round(float(np.mean(rand)), 1)
R["sim_bucketed_step_secs"] = round(float(np.mean(blocks)), 1)
R["steps_in_6h_random"] = round(6 * 3600 / np.mean(rand), 1)
R["steps_in_6h_bucketed"] = round(6 * 3600 / np.mean(blocks), 1)
R["bucketed_gain"] = round(float(np.mean(rand) / np.mean(blocks)), 3)
R["floor_secs_used"] = floor
R["share_of_bucketed_steps_at_floor"] = round(float(np.mean([b == floor for b in blocks])), 3)
T.to_csv(os.path.join(OUT, "v05_steps.csv"), index=False)
json.dump(R, open(os.path.join(OUT, "v05_step_time.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_v05")
