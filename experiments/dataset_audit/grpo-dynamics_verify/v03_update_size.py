#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 3: GD-2 (step size vs signal). Independent re-derivation from the logs.

(a) summed learning rate of M1 SFT and M2 GRPO (from the logged learning_rate of every optimizer step; HF Trainer logs the
    lr used by that step), up to checkpoint 25 and to step 30;
(b) ||dW||_F per unit summed lr (adapter_effect_m2_ckpt25.json), max|B| per unit summed lr;
(c) NEW: an upper bound that needs no signal assumption. Adam moves each parameter by at most ~lr per step, so after the
    run every LoRA-B entry satisfies |B_ij| <= sum(lr) (+ a small overshoot from bias-corrected early steps, simulated).
    With PEFT's kaiming-uniform(a=sqrt(5)) A (rows of squared norm ~1/3) and scale alpha/r = 2, ||scale*B*A||_F
    <= 2*sqrt(1/3)*sqrt(#B entries)*sum(lr). Model dims from research/aya-models.md (36 layers, d 2048, FFN 11008,
    16 q heads, 4 kv heads, head_dim 128; LoRA on q,k,v,o,gate,up,down; r 16).
(d) NEW: a small numpy Adam simulation (beta1 .9, beta2 .999, eps 1e-8, bias correction, the run's lr schedule) mapping
    per-parameter gradient SNR to rms|theta| / sum(lr) and max|theta| / sum(lr) over 16.8M parameters, to read the
    observed bulk efficiency (rms) and the observed max|B| statistic.
(e) linear extrapolations for the rerun.
Outputs: results/v03_update_size.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs")
sft = json.load(open(os.path.join(TL, "m1_sft", "result.json")))
grpo = json.load(open(os.path.join(TL, "m2_grpo", "result.json")))
ae = json.load(open(os.path.join(TL, "adapter_effect_m2_ckpt25.json")))["adapters"]
R = {}

s_lr = [e["learning_rate"] for e in sft["log_history"] if "loss" in e and "eval_loss" not in e and "learning_rate" in e]
g_hist = [e for e in grpo["log_history"] if "loss" in e and "learning_rate" in e]
g_lr = [e["learning_rate"] for e in g_hist]
R["sft_steps_logged"] = len(s_lr)
R["grpo_steps_logged"] = len(g_lr)
R["grpo_lr_first5"] = g_lr[:5]
R["sum_lr_sft"] = sum(s_lr)
R["sum_lr_grpo_30"] = sum(g_lr)
R["sum_lr_grpo_25"] = sum(g_lr[:25])
R["ratio_sum_lr_30"] = sum(g_lr) / sum(s_lr)
R["ratio_sum_lr_25"] = sum(g_lr[:25]) / sum(s_lr)
m1, m2 = ae["m1_sft_final"], ae["m2_grpo_ckpt25"]
R["dW_ratio_ckpt25"] = m2["total_dW_fro"] / m1["total_dW_fro"]
R["dW_per_sum_lr"] = {"sft": m1["total_dW_fro"] / sum(s_lr), "grpo_ckpt25": m2["total_dW_fro"] / sum(g_lr[:25])}
R["maxB_per_sum_lr"] = {"sft": m1["max_abs_B"] / sum(s_lr), "grpo_ckpt25": m2["max_abs_B"] / sum(g_lr[:25])}
R["grad_norm_range"] = [min(e["grad_norm"] for e in g_hist), max(e["grad_norm"] for e in g_hist)]

# (c) structural upper bound
L, d, ffn, hq, hkv, hd, r = 36, 2048, 11008, 16, 4, 128, 16
outs = {"q": hq * hd, "k": hkv * hd, "v": hkv * hd, "o": d, "gate": ffn, "up": ffn, "down": d}
ins = {"q": d, "k": d, "v": d, "o": hq * hd, "gate": d, "up": d, "down": ffn}
nB = L * r * sum(outs.values())
nA = L * r * sum(ins.values())
R["lora_modules"] = L * 7
R["lora_B_entries"] = nB
R["lora_A_entries"] = nA
scale = 2.0
cap_per_sum_lr = scale * math.sqrt(1 / 3) * math.sqrt(nB)
R["max_dW_per_sum_lr_if_every_B_entry_moves_sum_lr"] = cap_per_sum_lr
R["grpo_bulk_efficiency_ckpt25"] = R["dW_per_sum_lr"]["grpo_ckpt25"] / cap_per_sum_lr
R["sft_bulk_efficiency_note"] = ("not comparable: SFT also moved A by up to sum(lr)=%.4f vs A entries ~U(+-1/sqrt(fan_in)), "
                                 "|A| ~ 0.022 for fan_in 2048" % sum(s_lr))
R["grpo30_upper_bound_dW"] = cap_per_sum_lr * sum(g_lr)
R["grpo30_upper_bound_vs_sft_dW"] = cap_per_sum_lr * sum(g_lr) / m1["total_dW_fro"]

# (d) Adam simulation
rng = np.random.default_rng(0)
lr_sched = np.array(g_lr[:25])
def adam_sim(snr, n=200_000, steps=25, b1=0.9, b2=0.999, eps=1e-8, gscale=5e-5):
    th = np.zeros(n)
    m = np.zeros(n)
    v = np.zeros(n)
    mu = gscale * snr * np.sign(rng.standard_normal(n))     # fixed per-parameter mean direction
    for t in range(1, steps + 1):
        g = mu + gscale * rng.standard_normal(n)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        th -= lr_sched[t - 1] * mh / (np.sqrt(vh) + eps)
    s = lr_sched[:steps].sum()
    return float(np.sqrt(np.mean(th ** 2)) / s), float(np.abs(th).max() / s), float(np.quantile(np.abs(th), 1 - 1e-5) / s)
sim = {}
for snr in (0.0, 0.1, 0.25, 0.5, 1.0, 3.0):
    rms, mx, q = adam_sim(snr)
    sim[str(snr)] = {"rms_over_sumlr": round(rms, 4), "max_over_sumlr_200k": round(mx, 4), "q99999_over_sumlr": round(q, 4)}
R["adam_sim_25_steps"] = sim
# same simulation on the SFT schedule (114 logged lrs), to read SFT's bulk efficiency against its own noise floor
lr_sched = np.array(s_lr)
sim_sft = {}
for snr in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
    rms, mx, q = adam_sim(snr, steps=len(s_lr))
    sim_sft[str(snr)] = {"rms_over_sumlr": round(rms, 4)}
R["adam_sim_sft_schedule"] = sim_sft
R["sft_rms_B_over_sumlr_est_(A_assumed_~init)"] = m1["total_dW_fro"] / sum(s_lr) / (scale * math.sqrt(1 / 3) * math.sqrt(nB))
# observed rms |B| entry relative to sum lr (from total dW via the A-norm relation)
R["grpo_rms_B_over_sumlr_est"] = R["dW_per_sum_lr"]["grpo_ckpt25"] / (scale * math.sqrt(1 / 3) * math.sqrt(nB))

# robustness: pure-noise floor when the per-step gradient scale follows the logged GRPO grad_norm (steps 1-25)
gn = np.array([e["grad_norm"] for e in g_hist[:25]])
lr_sched = np.array(g_lr[:25])
def adam_sim_scaled(snr, n=200_000, b1=0.9, b2=0.999, eps=1e-8):
    th = np.zeros(n); m = np.zeros(n); v = np.zeros(n)
    mu = np.sign(rng.standard_normal(n))
    for t in range(1, 26):
        g = gn[t - 1] * 1e-4 * (snr * mu + rng.standard_normal(n))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        th -= lr_sched[t - 1] * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return float(np.sqrt(np.mean(th ** 2)) / lr_sched.sum())
R["adam_sim_25_steps_gradnorm_scaled_rms"] = {str(snr): round(adam_sim_scaled(snr), 4) for snr in (0.0, 0.1, 0.25, 0.5)}

# (e) extrapolations (linear in lr; steps via summed lr)
def sum_lr(lr, steps, warm=3):
    return sum(lr * min(1.0, s / warm) for s in range(steps))
base25 = sum(g_lr[:25])
for lr in (2e-5, 5e-5, 1e-4):
    for steps in (30, 36):
        R[f"pred_dW_vs_sft_lr{lr}_steps{steps}"] = round(R["dW_ratio_ckpt25"] * sum_lr(lr, steps) / base25, 4)
R["check_sum_lr_formula_30"] = sum_lr(5e-6, 30)
# log-prob effect: 0.0171 measured on the final (30-step) adapter (log.md 18:37) -> linear
eff30 = 0.0171
for lr in (2e-5, 5e-5, 1e-4):
    for steps in (30, 36):
        R[f"pred_logprob_effect_lr{lr}_steps{steps}"] = round(eff30 * sum_lr(lr, steps) / sum_lr(5e-6, 30), 4)
R["stack_threshold_10x_residual"] = round(0.01916 / 0.1, 4)
json.dump(R, open(os.path.join(OUT, "v03_update_size.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
print("OK_v03")
