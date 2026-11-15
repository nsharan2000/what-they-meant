#!/usr/bin/env python3
"""grpo-dynamics lens, step 4: how far does a fresh LoRA move at lr 5e-6 vs 2e-5 vs 5e-5 (30 GRPO steps)?

Model: a LoRA's B starts at 0 and Adam moves each coordinate by ~lr x |m_hat/sqrt(v_hat)| per step, a factor that is ~1 for
the first steps whatever the gradient's scale. So the displacement is set by the summed learning rate (sum_lr), not by the
reward. Calibration from the two adapters we have (adapter_effect_m2_ckpt25.json):
  efficiency = total ||dW||_F / sum_lr   (SFT: 114 steps, lr 1e-4 warm-up + cosine; GRPO: 25 steps at 5e-6, 3 warm-up steps)
and the per-coordinate check max|B| / sum_lr (1.0 = every step moved the largest coordinate the full lr in the same direction).
A numpy Adam simulation (scalar coordinates, gradient = signal + noise at several signal-to-noise ratios, the same schedules)
shows how |theta| / sum_lr depends on gradient coherence over 25-114 steps.
The stack-check numbers (mean |d logprob| of the M2 adapter and of the bf16 merge residual) are quoted from log.md (line
'm2_stack FAILED'), parsed here so the arithmetic is reproducible; the rerun prediction scales the adapter effect linearly in
sum_lr (first order).
Output: results/update_size.json
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments/training_eval/results/train_logs")
eff = json.load(open(os.path.join(TL, "adapter_effect_m2_ckpt25.json")))["adapters"]
sft_lh = json.load(open(os.path.join(TL, "m1_sft/result.json")))["log_history"]
grpo_lh = json.load(open(os.path.join(TL, "m2_grpo/result.json")))["log_history"]
merge = json.load(open(os.path.join(TL, "m1_sft/merge_info.json")))
R = {}

sft_lrs = np.array([e["learning_rate"] for e in sft_lh if "loss" in e and "learning_rate" in e])
grpo_lrs = np.array([e["learning_rate"] for e in grpo_lh if "reward" in e])
R["sft"] = {"steps": int(len(sft_lrs)), "sum_lr": float(sft_lrs.sum()), "peak_lr": float(sft_lrs.max()),
            "first_lrs": sft_lrs[:7].tolist()}
R["grpo"] = {"steps": int(len(grpo_lrs)), "sum_lr_all": float(grpo_lrs.sum()), "sum_lr_ckpt25": float(grpo_lrs[:25].sum()),
             "first_lrs": grpo_lrs[:5].tolist(), "note": "the logged lr of step k is the lr its update used (step 1 logs 0.0)"}
s, g = eff["m1_sft_final"], eff["m2_grpo_ckpt25"]
R["efficiency"] = {
    "sft_dW_fro_per_sum_lr": s["total_dW_fro"] / R["sft"]["sum_lr"],
    "grpo_dW_fro_per_sum_lr": g["total_dW_fro"] / R["grpo"]["sum_lr_ckpt25"],
    "sft_maxB_per_sum_lr": s["max_abs_B"] / R["sft"]["sum_lr"],
    "grpo_maxB_per_sum_lr": g["max_abs_B"] / R["grpo"]["sum_lr_ckpt25"],
    "observed_ratio_dW_grpo25_vs_sft": g["ratio_total_vs_m1_sft_final"],
    "sum_lr_ratio_grpo25_vs_sft": R["grpo"]["sum_lr_ckpt25"] / R["sft"]["sum_lr"],
}
R["efficiency"]["grpo_vs_sft_efficiency"] = R["efficiency"]["grpo_dW_fro_per_sum_lr"] / R["efficiency"]["sft_dW_fro_per_sum_lr"]


def sched(lr, n, warm=3):
    """constant_with_warmup as logged: update k (1-based) uses lr * (k-1)/warm for k <= warm, else lr."""
    return np.array([lr * min(1.0, (k - 1) / warm) for k in range(1, n + 1)])


# ---- Adam simulation on scalar coordinates (beta1 0.9, beta2 0.999, eps 1e-8, no weight decay) ----
def adam_disp(lrs, snr, n_coord=4000, seed=0):
    rng = np.random.default_rng(seed)
    th = np.zeros(n_coord)
    m = np.zeros(n_coord)
    v = np.zeros(n_coord)
    sign = rng.choice([-1.0, 1.0], n_coord)
    scale = 10 ** rng.uniform(-4, -1, n_coord)   # gradient scale is irrelevant to Adam (checks that)
    for t, lr in enumerate(lrs, 1):
        gr = scale * (snr * sign + rng.standard_normal(n_coord))
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        th -= lr * mh / (np.sqrt(vh) + 1e-8)
    return float(np.abs(th).mean() / lrs.sum()), float(np.sqrt((th ** 2).mean()) / lrs.sum())

sims = {}
for name, lrs in (("grpo_25_steps", sched(5e-6, 25)), ("grpo_30_steps", sched(5e-6, 30)), ("sft_schedule", sft_lrs)):
    sims[name] = {str(snr): dict(zip(("mean_abs_per_sum_lr", "rms_per_sum_lr"), adam_disp(lrs, snr))) for snr in (0.0, 0.1, 0.3, 1.0, 3.0)}
R["adam_sim"] = sims

# ---- predictions for a fresh LoRA from merged M1, 30 steps, 3 warm-up steps ----
e_g = R["efficiency"]["grpo_dW_fro_per_sum_lr"]
pred = {}
for lr in (5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4):
    for n in (30, 36, 40):
        sl = sched(lr, n).sum()
        dw = e_g * sl
        pred[f"lr{lr:g}_steps{n}"] = {"sum_lr": sl, "pred_dW_fro": round(dw, 4), "pred_ratio_vs_sft": round(dw / s["total_dW_fro"], 4),
                                      "pred_max_abs_B": round(R["efficiency"]["grpo_maxB_per_sum_lr"] * sl, 6)}
R["predictions"] = pred

# ---- stack-check arithmetic (numbers parsed from log.md) ----
log = open(os.path.join(ROOT, "log.md")).read()
m = re.search(r'mean \|Δ\s*\n?\s*logprob\| ([0-9.]+) ≥ 0\.1 × adapter effect ([0-9.]+)', log)
if m:
    resid, effect = float(m.group(1)), float(m.group(2))
    need = resid / 0.1
    R["stack_check"] = {
        "source": "log.md, entry 2026-11-14 18:37 IST ('m2_stack FAILED')",
        "merge_residual_mean_abs_dlogprob": resid, "m2_adapter_effect_mean_abs_dlogprob": effect,
        "m1_adapter_effect_mean_abs_dlogprob(merge_info)": merge["adapter_effect"]["mean_abs_dlogprob"],
        "m2_effect_vs_m1_effect": effect / merge["adapter_effect"]["mean_abs_dlogprob"],
        "effect_needed_to_pass(resid<0.1*effect)": need, "scale_factor_needed": need / effect,
        "pred_effect_lr5e-5_30steps_linear": effect * sched(5e-5, 30).sum() / sched(5e-6, 30).sum(),
        "pred_effect_lr2e-5_30steps_linear": effect * sched(2e-5, 30).sum() / sched(5e-6, 30).sum(),
        "pred_effect_lr1e-4_30steps_linear": effect * sched(1e-4, 30).sum() / sched(5e-6, 30).sum(),
        "pred_effect_grid_linear": {f"lr{lr:g}_steps{n}": round(effect * sched(lr, n).sum() / sched(5e-6, 30).sum(), 4)
                                    for lr in (2e-5, 3e-5, 5e-5, 1e-4) for n in (30, 36, 40)},
        "assumes": "the stack check ran on the final (30-step) adapter; effect scales linearly with sum_lr (first order)",
    }
# decision-logit arithmetic: change of log-odds needed to move a 0.5 decision to 0.8
R["logit_needed"] = {"p0.5_to_0.8": float(np.log(0.8 / 0.2)), "p0.67_to_0.475_(A_ask_early_late)": float(np.log(0.6696 / 0.3304) - np.log(0.475 / 0.525))}
json.dump(R, open(os.path.join(OUT, "update_size.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
print("OK_04")
