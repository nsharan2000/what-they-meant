#!/usr/bin/env python3
"""grpo-dynamics lens, step 11: the M2 GRPO run's effective configuration and log-history ranges (item 5).

Reads training/train_grpo.py (the trainer kwargs literally passed to create_agentic_trainer), m2_grpo/args.json, guards.json
and result.json log_history. Output: results/run_config.json
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo")
src = open(os.path.join(ROOT, "training/train_grpo.py")).read()
call = src[src.index('trainer = create_agentic_trainer('):src.index('trainer.train()')]
kw = {}
for k in ("temperature", "top_p", "loss_type", "scale_rewards", "beta", "mask_truncated_completions", "lr_scheduler_type",
          "vllm_mode", "vllm_enable_sleep_mode", "gradient_checkpointing", "bf16", "seed", "log_completions"):
    m = re.search(rf"\b{k}=([^,\n]+)", call)
    kw[k] = m.group(1).strip() if m else None
lora = re.search(r"LoraConfig\((.*?\])\)", call, re.S)
kw["LoraConfig"] = " ".join(lora.group(1).split()) if lora else None
for k in ("num_iterations", "epsilon", "max_grad_norm", "importance_sampling", "steps_per_generation"):
    kw[f"{k}_set_in_script"] = bool(re.search(rf"\b{k}", call))
args = json.load(open(os.path.join(TL, "args.json")))
guards = json.load(open(os.path.join(TL, "guards.json")))
res = json.load(open(os.path.join(TL, "result.json")))
lh = [e for e in res["log_history"] if "reward" in e]


def rng(key):
    v = np.array([e[key] for e in lh if key in e], float)
    return {"min": round(float(v.min()), 5), "max": round(float(v.max()), 5), "mean": round(float(v.mean()), 5)}

R = {"trainer_kwargs_from_script": kw, "args_json": args, "guards": guards,
     "derived": {"completions_per_step": args["batch"] * args["num_generations"],
                 "prompts_seen": len(lh) * args["batch"], "dataset_rows": guards["grpo_prompts"]["prompts"],
                 "epochs_covered": round(len(lh) * args["batch"] / guards["grpo_prompts"]["prompts"], 4),
                 "logged_lrs_first5": [e["learning_rate"] for e in lh[:5]]},
     "log_ranges": {k: rng(k) for k in ("loss", "grad_norm", "entropy", "reward", "reward_std", "frac_reward_zero_std",
                                        "completions/mean_length", "completions/max_length", "clip_ratio/region_mean",
                                        "sampling/importance_sampling_ratio/min", "sampling/importance_sampling_ratio/max",
                                        "sampling/sampling_logp_difference/mean", "wtm/truncated_frac", "wtm/end_eos6_frac")},
     "end_ids": res["end_ids"], "trunc_penalty": res["trunc_penalty"], "minutes": res["minutes"],
     "train_loss_final": [e for e in res["log_history"] if "train_loss" in e][0]["train_loss"]}
json.dump(R, open(os.path.join(OUT, "run_config.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
print("OK_11")
