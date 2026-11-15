#!/usr/bin/env python3
"""grpo-dynamics lens, step 5: what sets the GRPO step time, and what fits in ~6 GPU hours?

Step k's wall time (progress.json / result.json step_times: generation + scoring + forward/backward) is matched with the
4 prompts generated at step k-1 (scores meta.step = optimizer steps already applied). Fits
  time = a + b * sum(prompt_tokens of the 4 prompts)          (8 completions per prompt; completion lengths ~50 tokens)
  time = a + b * sum(prompt_tokens) + c * sum(prompt_tokens^2)/1e4   (attention term)
and predicts the per-step time of candidate prompt pools (all levels; L1-L3 only; pair-balanced 2 A + 2 B).
Output: results/step_time.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
res = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/result.json")))
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
toks = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))])
toks = toks[toks.file == "grpo_train.jsonl"]
t = {d["step"]: d["seconds"] for d in res["step_times"]}
rows = []
for s, sub in G.groupby("step"):
    k = s + 1
    if k in t:
        pt = sub["prompt_tokens"].to_numpy(float)
        rows.append({"log_step": k, "seconds": t[k], "sum_tokens": pt.sum(), "max_tokens": pt.max(), "sum_sq": (pt ** 2).sum() / 1e4,
                     "n_L0r": int((sub["level"] == "L0r").sum()), "levels": "+".join(sorted(sub["level"])),
                     "n_over_4096": int((pt > 4096).sum())})
D = pd.DataFrame(rows)
R = {"n_steps": int(len(D)), "seconds_mean": round(D.seconds.mean(), 1), "seconds_min": D.seconds.min(), "seconds_max": D.seconds.max()}
X1 = np.c_[np.ones(len(D)), D.sum_tokens]
b1, *_ = np.linalg.lstsq(X1, D.seconds, rcond=None)
pred1 = X1 @ b1
X2 = np.c_[np.ones(len(D)), D.sum_tokens, D.sum_sq]
b2, *_ = np.linalg.lstsq(X2, D.seconds, rcond=None)
pred2 = X2 @ b2
ss = ((D.seconds - D.seconds.mean()) ** 2).sum()
R["fit_linear"] = {"a_s": round(b1[0], 2), "b_s_per_1k_tokens": round(1000 * b1[1], 3), "r2": round(1 - ((D.seconds - pred1) ** 2).sum() / ss, 4)}
R["fit_quadratic"] = {"coef": [round(x, 6) for x in b2], "r2": round(1 - ((D.seconds - pred2) ** 2).sum() / ss, 4)}
X3 = np.c_[np.ones(len(D)), D.max_tokens]
b3, *_ = np.linalg.lstsq(X3, D.seconds, rcond=None)
pred3 = X3 @ b3
R["fit_max_prompt"] = {"a_s": round(b3[0], 2), "b_s_per_1k_max_tokens": round(1000 * b3[1], 3),
                       "r2": round(1 - ((D.seconds - pred3) ** 2).sum() / ss, 4),
                       "note": "TRL pads every prompt of the 32-completion generation batch to the step's longest prompt"}
D["padding_waste"] = 1 - D.sum_tokens / (4 * D.max_tokens)
R["padding_waste_mean"] = round(D.padding_waste.mean(), 4)
R["two_regimes"] = {"max_tokens_le_4400": D[D.max_tokens <= 4400].seconds.describe().round(1).to_dict(),
                    "max_tokens_ge_6400": D[D.max_tokens >= 6400].seconds.describe().round(1).to_dict()}
R["corr_seconds_vs_n_L0r"] = round(float(np.corrcoef(D.seconds, D.n_L0r)[0, 1]), 4)
R["by_n_L0r"] = D.groupby("n_L0r").agg(n=("seconds", "size"), mean_s=("seconds", "mean"), mean_tokens=("sum_tokens", "mean")).round(1).to_dict("index")
R["fast_steps"] = D[D.seconds < 500][["log_step", "seconds", "sum_tokens", "levels"]].to_dict("records")

# per-level token means over the whole GRPO pool -> predicted step time for candidate pools (4 prompts per step)
lv = toks.groupby("level")["prompt_tokens"].mean()
R["pool_mean_tokens_by_level"] = lv.round(1).to_dict()
SHORT_S = D[D.max_tokens <= 4400].seconds.mean()     # observed regime means (the line fit is valid only for 3.8K-7.4K max)
LONG_S = D[D.max_tokens >= 6400].seconds.mean()


def step_time(max_tok):
    """observed regimes: max prompt <= 4.4K tokens -> SHORT_S (an upper bound for shorter steps); else the max-prompt line."""
    max_tok = np.asarray(max_tok, float)
    return np.where(max_tok <= 4400, SHORT_S, b3[0] + b3[1] * max_tok)


# a random 4-prompt step from a pool vs a length-matched step (e.g. twin pairs at one level/setting: max ~ the pair's length)
rng = np.random.default_rng(0)
pools = {"all_levels": toks, "L1_L2_L3": toks[toks.level.isin(["L1", "L2", "L3"])], "L2_L3": toks[toks.level.isin(["L2", "L3"])],
         "L0r_L1": toks[toks.level.isin(["L0r", "L1"])]}
R["regime_seconds"] = {"short(max<=4.4K)": round(SHORT_S, 1), "long(max>=6.4K)": round(LONG_S, 1)}
R["pool_pred"] = {}
for name, pool in pools.items():
    pt = pool["prompt_tokens"].to_numpy(float)
    mx = np.array([rng.choice(pt, 4, replace=False).max() for _ in range(4000)])
    st_random = float(step_time(mx).mean())
    st_matched = float(step_time(pt).mean())     # each step's max = one row's length (length-bucketed steps)
    R["pool_pred"][name] = {"n_rows": int(len(pt)), "mean_prompt_tokens": round(pt.mean(), 1),
                            "random_step_mean_max_tokens": round(mx.mean(), 1),
                            "pred_seconds_random_step": round(st_random, 1), "steps_in_6h_random": int(6 * 3600 // st_random),
                            "pred_seconds_length_bucketed": round(st_matched, 1),
                            "steps_in_6h_length_bucketed": int(6 * 3600 // st_matched)}
D.to_csv(os.path.join(OUT, "step_time.csv"), index=False)
json.dump(R, open(os.path.join(OUT, "step_time.json"), "w"), indent=1, default=float)
print(D.sort_values("seconds").to_string())
print(json.dumps(R, indent=1, default=float))
print("OK_05")
