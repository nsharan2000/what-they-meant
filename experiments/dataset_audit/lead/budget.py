"""Lead audit: GPU-hour estimates and a timeline for the remaining experiments (REPORT.md section 5).

Every estimate is derived from measured rates; nothing here was measured anew.
  - SFT:   114 steps in 310.8 min for 2 epochs of 904 rows (log.md 2026-11-14 11:48 IST; brief: ~2.6 h/epoch).
  - GRPO:  measured mean 718.7 s/step at 4 prompts x 8 generations; GD's step-time model predicts 735.1 s for a random
           4-prompt draw and 526.5 s for length-bucketed steps (grpo-dynamics/results/step_time.json, R^2 0.998 fit).
  - DPO:   1 epoch of 496 pairs ~4-5.4 h (brief; M3 run).
  - Eval:  one new arm ~1-1.5 h for test + hard_test x L0r/L1/L2/L2s x P0/P2 (brief).
  - Merge: 2-5 min (brief).  Audits: 2-4 h for 3 arms (base, M1, M3) in the pipeline (brief) -> per arm 2/3-4/3 h.
Single-turn probes (no measured rate) are bounded by the per-call cost of one eval arm:
  864 episodes per arm = (82 test + 26 hard_test rows) x 4 levels x 2 prompt settings (run_eval.sh LEVELS/PROMPTS;
  splits-eval-alignment_verify/results/v07_eval_alignment.json d_clusters), 2-3 model calls per episode for a trained arm
  (canonical look-up(s) + decision) -> 1,728-2,592 calls per 1-1.5 h. An n=8 sampled call counts as 2 call-equivalents
  (shared prefill, 8 short decodes: GRPO completions max 117 tokens, grpo-dynamics_verify/results/v11_checks.json).
Run: python3 experiments/dataset_audit/lead/budget.py   (stdlib only) -> lead/results/budget.json
"""
import json, os
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
step_time = json.load(open(os.path.join(ROOT, "grpo-dynamics/results/step_time.json")))

SFT_H_PER_EPOCH = 310.8 / 2 / 60                      # 2.59 h
GRPO_S_MEASURED = step_time["seconds_mean"]            # 718.7
GRPO_S_RANDOM = step_time["pool_pred"]["all_levels"]["pred_seconds_random_step"]          # 735.1
GRPO_S_BUCKETED = step_time["pool_pred"]["all_levels"]["pred_seconds_length_bucketed"]    # 526.5
DPO_H_496 = (4.0, 5.4)
EVAL_H = (1.0, 1.5)
MERGE_H = (2 / 60, 5 / 60)
AUDIT_H_PER_ARM = (2 / 3, 4 / 3)
EPISODES_PER_ARM = (82 + 26) * 4 * 2                   # 864
CALLS_PER_ARM = (EPISODES_PER_ARM * 2, EPISODES_PER_ARM * 3)
H_PER_CALL = (EVAL_H[0] / CALLS_PER_ARM[1], EVAL_H[1] / CALLS_PER_ARM[0])   # optimistic, pessimistic

N_TRAIN_DEC, N_VAL_DEC = 496, 112                      # grpo_train / sft_train decision prompts; sft_val decision rows


def calls_h(n_calls):
    return (n_calls * H_PER_CALL[0], n_calls * H_PER_CALL[1])


def add(*parts):
    return (sum(p[0] for p in parts), sum(p[1] for p in parts))


jobs = {}
# J1: decision probe + M1 sample bank (greedy M1 and M3 on train+val decision prompts; M1 n=8, T=1 on train prompts)
j1_calls = (N_TRAIN_DEC + N_VAL_DEC) * 2 + N_TRAIN_DEC * 2
jobs["J1_probe"] = {"parts": {"single_turn_calls": j1_calls}, "hours": calls_h(j1_calls)}
# J1b (optional): decision-turn eval on held-out prompts with canonical look-ups pre-filled, base/M1/M3, L0r+L2 x P0+P2
j1b_calls = 3 * (82 + 26) * 2 * 2
jobs["J1b_decision_turn_eval"] = {"parts": {"single_turn_calls": j1b_calls}, "hours": calls_h(j1b_calls)}
# J3: GRPO v2 from merged M1: 36 length-bucketed steps (cap 6 h wall clock) + 3 checkpoint probes + eval
grpo_train = (36 * GRPO_S_BUCKETED / 3600, 6.0)
jobs["J3_grpo_v2"] = {"parts": {"train": grpo_train, "ckpt_probe": calls_h(3 * N_VAL_DEC), "eval": EVAL_H,
                                "train_if_unbucketed_36_steps_h": 36 * GRPO_S_MEASURED / 3600,
                                "train_if_random_30_steps_h": 30 * GRPO_S_RANDOM / 3600}}
jobs["J3_grpo_v2"]["hours"] = add(grpo_train, calls_h(3 * N_VAL_DEC), EVAL_H)
# J2: on-policy twin DPO from merged M1 (<= 496 pairs, 1 epoch) + merge + eval
jobs["J2_onpolicy_dpo_from_m1"] = {"parts": {"train": DPO_H_496, "merge": MERGE_H, "eval": EVAL_H},
                                   "hours": add(DPO_H_496, MERGE_H, EVAL_H)}
# J4: decision-weighted SFT from base, same data and hyperparameters as M1 (2 epochs), probe both epoch ends + eval
sft2 = (2 * SFT_H_PER_EPOCH, 2 * SFT_H_PER_EPOCH)
jobs["J4_weighted_sft"] = {"parts": {"train": sft2, "probe": calls_h(2 * N_VAL_DEC), "eval": EVAL_H},
                           "hours": add(sft2, calls_h(2 * N_VAL_DEC), EVAL_H)}
# J5: plain 1-epoch SFT (epoch question) + probe + eval
sft1 = (SFT_H_PER_EPOCH, SFT_H_PER_EPOCH)
jobs["J5_sft_1epoch"] = {"parts": {"train": sft1, "probe": calls_h(N_VAL_DEC), "eval": EVAL_H},
                         "hours": add(sft1, calls_h(N_VAL_DEC), EVAL_H)}
# J6 (alternative to J2): M3' = DPO from base on the current pairs at lr 5e-5
jobs["J6_dpo_from_base_lr5e5"] = {"parts": {"train": DPO_H_496, "merge": MERGE_H, "eval": EVAL_H},
                                  "hours": add(DPO_H_496, MERGE_H, EVAL_H)}
jobs["AUDIT_final_arm"] = {"parts": {}, "hours": AUDIT_H_PER_ARM}
jobs["M3_adapter_effect"] = {"parts": {"note": "experiments/training_eval/adapter_effect.py is CPU-only"}, "hours": (0.0, 0.0)}

for j in jobs.values():
    j["hours"] = [round(j["hours"][0], 2), round(j["hours"][1], 2)]

CHAINS = {
    "A_if_J2_approved": ["J1_probe", "J3_grpo_v2", "J2_onpolicy_dpo_from_m1", "AUDIT_final_arm"],
    "B_if_J2_not_approved": ["J1_probe", "J3_grpo_v2", "J4_weighted_sft", "AUDIT_final_arm"],
}
T0 = {"best": datetime(2026, 11, 15, 3, 0), "worst": datetime(2026, 11, 15, 6, 0)}   # pipeline end (brief)
CUTOFF = datetime(2026, 11, 16, 7, 29)            # last moment for GPU work (Mon 07:29 IST)
BUDGET_H = 18.0                                   # brief: realistic GPU hours after the pipeline

fmt = lambda d: d.strftime("%a %H:%M")
chains = {}
for name, seq in CHAINS.items():
    lo = sum(jobs[j]["hours"][0] for j in seq)
    hi = sum(jobs[j]["hours"][1] for j in seq)
    timeline = {}
    for case, idx in (("best", 0), ("worst", 1)):
        t, steps = T0[case], []
        for j in seq:
            t = t + timedelta(hours=jobs[j]["hours"][idx])
            steps.append([j, fmt(t)])
        timeline[case] = {"start": fmt(T0[case]), "ends": steps,
                          "slack_to_cutoff_h": round((CUTOFF - t).total_seconds() / 3600, 2)}
    chains[name] = {"jobs": seq, "gpu_h": [round(lo, 2), round(hi, 2)], "within_18h_budget_upper": hi <= BUDGET_H + 0.05,
                    "timeline": timeline}

latest_start = {j: fmt(CUTOFF - timedelta(hours=jobs[j]["hours"][1]))
                for j in ("J4_weighted_sft", "J5_sft_1epoch", "J2_onpolicy_dpo_from_m1", "J3_grpo_v2", "J1b_decision_turn_eval")}

res = {"rates": {"sft_h_per_epoch": round(SFT_H_PER_EPOCH, 3), "grpo_s_per_step_measured": GRPO_S_MEASURED,
                 "grpo_s_per_step_random_pred": GRPO_S_RANDOM, "grpo_s_per_step_bucketed_pred": GRPO_S_BUCKETED,
                 "dpo_h_496_pairs": DPO_H_496, "eval_h_per_arm": EVAL_H, "merge_h": [round(x, 3) for x in MERGE_H],
                 "audit_h_per_arm": [round(x, 2) for x in AUDIT_H_PER_ARM], "episodes_per_eval_arm": EPISODES_PER_ARM,
                 "calls_per_eval_arm": CALLS_PER_ARM, "h_per_single_turn_call": [round(x, 6) for x in H_PER_CALL]},
       "jobs": jobs, "chains": chains, "cutoff": fmt(CUTOFF), "budget_h": BUDGET_H,
       "latest_start_to_finish_by_cutoff_pessimistic": latest_start}
os.makedirs(os.path.join(ROOT, "lead", "results"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "lead", "results", "budget.json"), "w"), indent=1, default=str)
for k, v in jobs.items():
    print(f"{k:28s} {v['hours'][0]:5.2f}-{v['hours'][1]:5.2f} h")
for k, v in chains.items():
    print(k, v["gpu_h"], "best ends", v["timeline"]["best"]["ends"][-1], "slack", v["timeline"]["best"]["slack_to_cutoff_h"],
          "| worst ends", v["timeline"]["worst"]["ends"][-1], "slack", v["timeline"]["worst"]["slack_to_cutoff_h"])
print("latest starts (pessimistic):", latest_start)
