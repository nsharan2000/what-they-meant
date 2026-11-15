"""CM-2 / CM-3: M1 SFT log history re-read independently (experiments/training_eval/results/train_logs/m1_sft/result.json).

Checks: epoch-mean losses, last-step loss, per-step SD, epoch-boundary behaviour, eval loss / token accuracy / entropy after each
epoch, and how much of the eval-loss rise the decision tokens could explain at most (arithmetic bound using v1_tokens.json).
Output: results/v4_sftlog.json
"""
import json
import os

import numpy as np

from vcommon import ROOT, RES, dump

r = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m1_sft/result.json")))
lh = r["log_history"]
train = [e for e in lh if "loss" in e and "eval_loss" not in e]
ev = [e for e in lh if "eval_loss" in e]
steps = np.array([e["step"] for e in train])
loss = np.array([e["loss"] for e in train])
acc = np.array([e["mean_token_accuracy"] for e in train])
ent = np.array([e["entropy"] for e in train])
ep = np.array([e["epoch"] for e in train])
e1 = ep <= 1.0 + 1e-9
# the epoch boundary: first step whose epoch > 1
b = int(np.argmax(ep > 1.0))
out = {"n_train_logs": len(train), "max_step": int(steps.max()), "steps_epoch1": int(e1.sum()),
       "boundary_step": int(steps[b]), "epoch_at_boundary": [float(ep[b - 1]), float(ep[b])],
       "loss_mean_epoch1": round(float(loss[e1].mean()), 4), "loss_mean_epoch2": round(float(loss[~e1].mean()), 4),
       "loss_last_step": round(float(loss[-1]), 4), "loss_sd_epoch2": round(float(loss[~e1].std(ddof=1)), 4),
       "loss_sd_epoch1_second_half": round(float(loss[(ep > 0.5) & e1].std(ddof=1)), 4),
       "acc_mean_epoch1": round(float(acc[e1].mean()), 4), "acc_mean_epoch2": round(float(acc[~e1].mean()), 4),
       "entropy_mean_epoch1": round(float(ent[e1].mean()), 4), "entropy_mean_epoch2": round(float(ent[~e1].mean()), 4),
       "last10_epoch1": round(float(loss[b - 10:b].mean()), 4), "first10_epoch2": round(float(loss[b:b + 10].mean()), 4),
       "last10_epoch2": round(float(loss[-10:].mean()), 4),
       "first10_epoch1": round(float(loss[:10].mean()), 4)}
out["ratio_first10_ep2_vs_last10_ep1"] = round(out["first10_epoch2"] / out["last10_epoch1"], 4)
# log-linear trend over epoch 1's second half, extrapolated to the first 10 steps of epoch 2
m = (ep > 0.5) & e1
coef = np.polyfit(steps[m], np.log(loss[m]), 1)
pred = np.exp(np.polyval(coef, steps[b:b + 10]))
out["trend_slope_per_step_log"] = round(float(coef[0]), 5)
out["first10_epoch2_vs_trend"] = round(float(loss[b:b + 10].mean() / pred.mean()), 3)
# alternative: linear trend in raw loss
coef2 = np.polyfit(steps[m], loss[m], 1)
pred2 = np.polyval(coef2, steps[b:b + 10])
out["first10_epoch2_vs_linear_trend"] = round(float(loss[b:b + 10].mean() / pred2.mean()), 3)
# Welch t for last 10 of epoch1 vs first 10 of epoch 2 (is any boundary effect detectable at this noise?)
x, y = loss[b - 10:b], loss[b:b + 10]
t = (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / 10 + y.var(ddof=1) / 10)
out["boundary_welch_t"] = round(float(t), 3)
# a boundary drop detectable at 2 SE?
out["boundary_2se_detectable_drop"] = round(float(2 * np.sqrt(x.var(ddof=1) / 10 + y.var(ddof=1) / 10)), 4)
out["evals"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k in
                 ("step", "epoch", "eval_loss", "eval_mean_token_accuracy", "eval_entropy")} for e in ev]
if len(ev) >= 2:
    a, c = ev[0], ev[-1]
    out["eval_loss_change_pct"] = round(100 * (c["eval_loss"] / a["eval_loss"] - 1), 2)
    out["eval_entropy_change_pct"] = round(100 * (c["eval_entropy"] / a["eval_entropy"] - 1), 2)
    out["eval_acc_change_pts"] = round(100 * (c["eval_mean_token_accuracy"] - a["eval_mean_token_accuracy"]), 3)
    # decision tokens' maximum share of the eval-loss rise: 112 first tokens + 56 tool-name tokens of 9,173 val trained tokens.
    tok = json.load(open(os.path.join(RES, "v1_tokens.json")))["sft_val"]
    dP = tok["decision_rows"] + tok["decision_rows_tool_call"]
    T = tok["trained_tokens_per_epoch"]
    rise = c["eval_loss"] - a["eval_loss"]
    out["eval_loss_rise"] = round(rise, 4)
    out["decision_pivot_nll_rise_needed_to_explain_whole_rise_nats"] = round(rise * T / dP, 2)
    out["eval_loss_rise_if_every_pivot_token_rose_by_1_nat"] = round(dP / T, 4)
# per-step loss by epoch around boundary (for the report)
out["loss_steps_50_to_64"] = {int(s): round(float(l), 3) for s, l in zip(steps, loss) if 50 <= s <= 64}
# learning-rate schedule
out["lr_at"] = {int(e["step"]): e["learning_rate"] for e in train if e["step"] in (1, 5, 10, 57, 58, 114)}
dump(out, "v4_sftlog.json")
print(json.dumps(out, indent=1))
