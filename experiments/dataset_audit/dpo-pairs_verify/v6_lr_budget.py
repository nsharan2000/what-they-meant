"""Verifier step 6: learning-rate budget of the running M3 DPO vs the measured SFT (M1) and GRPO (M2) adapters.

Inputs (read-only): experiments/training_eval/results/train_logs/{m1_sft,m2_grpo}/result.json (log_history: the logged
learning_rate at step k is the rate used for update k: step 1 logs 0.0 under warmup), adapter_effect_m2_ckpt25.json,
training/train_dpo.py defaults (--lr 5e-6, batch 1 x grad-accum 16, 1 epoch) and AlignTune's DPO setup (linear schedule,
warmup 0: third_party/aligntune/src/aligntune/core/rl/trainer_base.py _setup_optimizer_scheduler + backend_factory
warmup_ratio=0.0 for create_rl_trainer; read from code, not from the M3 run).
Output: results/v6_lr_budget.json
"""
from __future__ import annotations

import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
L = os.path.join(ROOT, "experiments/training_eval/results/train_logs")
sft = json.load(open(os.path.join(L, "m1_sft/result.json")))["log_history"]
grpo = json.load(open(os.path.join(L, "m2_grpo/result.json")))["log_history"]
eff = json.load(open(os.path.join(L, "adapter_effect_m2_ckpt25.json")))["adapters"]

s_steps = [h for h in sft if "loss" in h and "learning_rate" in h]
g_steps = [h for h in grpo if "loss" in h and "learning_rate" in h]
sft_lr = [h["learning_rate"] for h in s_steps]
sft_loss = [h["loss"] for h in s_steps]
g_lr = [h["learning_rate"] for h in g_steps]
cum_sft = [sum(sft_lr[:k + 1]) for k in range(len(sft_lr))]
tot_sft = cum_sft[-1]
tot_g25 = sum(g_lr[:25])

N_DPO = 496
steps_dpo = N_DPO // (1 * 16)        # HF: len(dataloader) // grad_accum, 1 epoch
lr0 = 5e-6
lin = [lr0 * (1 - k / steps_dpo) for k in range(steps_dpo)]    # linear decay, warmup 0: lr at update k+1
tot_dpo_lin = sum(lin)
tot_dpo_const = lr0 * steps_dpo

dw_sft = eff["m1_sft_final"]["total_dW_fro"]
dw_g = eff["m2_grpo_ckpt25"]["total_dW_fro"]
calib_sft = dw_sft / tot_sft
calib_g = dw_g / tot_g25
k_rel = calib_g / calib_sft
step_same = next(k + 1 for k, c in enumerate(cum_sft) if c >= tot_dpo_lin)
g_reward = [h.get("reward") for h in g_steps]

res = {
    "sft_steps_logged": len(s_steps), "sft_total_lr": tot_sft, "sft_lr_first6": sft_lr[:6],
    "grpo_steps_logged": len(g_steps), "grpo_lr_first5": g_lr[:5], "grpo_total_lr_through_step25": tot_g25,
    "dpo_optimizer_steps": steps_dpo, "dpo_total_lr_linear": tot_dpo_lin, "dpo_total_lr_constant": tot_dpo_const,
    "dpo_over_sft_total_lr_linear": tot_dpo_lin / tot_sft, "dpo_over_sft_total_lr_constant": tot_dpo_const / tot_sft,
    "dW_fro_sft": dw_sft, "dW_fro_grpo25": dw_g, "grpo_over_sft_dW": dw_g / dw_sft,
    "grpo_over_sft_total_lr": tot_g25 / tot_sft,
    "dW_per_unit_lr_grpo_relative_to_sft": k_rel,
    "max_abs_B_over_total_lr": {"sft": eff["m1_sft_final"]["max_abs_B"] / tot_sft, "grpo25": eff["m2_grpo_ckpt25"]["max_abs_B"] / tot_g25},
    "predicted_dpo_dW_over_sft_linear(calib=grpo)": k_rel * tot_dpo_lin / tot_sft,
    "predicted_dpo_dW_over_sft_constant(calib=grpo)": k_rel * tot_dpo_const / tot_sft,
    "predicted_dpo_dW_over_sft_linear(calib=sft)": tot_dpo_lin / tot_sft,
    "sft_step_where_cum_lr_reaches_dpo_total": step_same,
    "sft_cum_lr_steps_1_8": cum_sft[:8],
    "sft_loss_steps_1_5": sft_loss[:5], "sft_loss_mean_steps_8_12": sum(sft_loss[7:12]) / 5,
    "grpo_reward_mean_steps_1_12": sum(g_reward[:12]) / 12, "grpo_reward_mean_steps_13_25": sum(g_reward[12:25]) / 13,
    "grpo_reward_mean_steps_26_30": sum(g_reward[25:30]) / len(g_reward[25:30]),
    "lr_needed_for_10pct_of_sft_budget_linear": 0.10 * tot_sft / (tot_dpo_lin / lr0),
}
json.dump(res, open(os.path.join(OUT, "v6_lr_budget.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
