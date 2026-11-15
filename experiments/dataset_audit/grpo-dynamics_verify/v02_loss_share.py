#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 2: GD-1's loss arithmetic ('the decision is ~2 % of the token-averaged loss').

The SFT loss is the token-mean NLL over ALL trained tokens of a batch (completion-only; trained span = target tokens +
<|END_RESPONSE|> + <|END_OF_TURN_TOKEN|>, guards.check_sft). Per epoch the decision tokens of the 497 decision rows sit among
the trained tokens of all 904 rows. Two estimates of the decision NLL per decision row from M1's samples (v01):
  gold-class NLL (smoothed, mean over the 120 sampled prompts) and first-token FAMILY NLL (message vs approve/commit call).
Share of the per-token loss = (n_decision_rows * NLL_dec / total trained tokens) / logged loss.
Outputs: results/v02_loss_share.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
tok = [json.loads(l) for l in open(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))]
sft_t = [t for t in tok if t["file"] == "sft_train.jsonl"]
m1 = json.load(open(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m1_sft", "result.json")))
v01 = json.load(open(os.path.join(OUT, "v01_policy.json")))
R = {}
trained = np.array([t["completion_tokens"] + 2 for t in sft_t])
is_dec = np.array([t["turn"] == "decision" for t in sft_t])
R["rows"] = int(len(sft_t))
R["decision_rows"] = int(is_dec.sum())
R["trained_tokens_mean_all_rows"] = round(float(trained.mean()), 2)
R["trained_tokens_mean_decision_rows"] = round(float(trained[is_dec].mean()), 2)
R["trained_tokens_total_per_epoch"] = int(trained.sum())
lh = [e for e in m1["log_history"] if "loss" in e and "eval_loss" not in e]
ep2 = [e["loss"] for e in lh if e["epoch"] > 1.0]
ep1 = [e["loss"] for e in lh if e["epoch"] <= 1.0]
R["loss_epoch1_mean"] = round(float(np.mean(ep1)), 4)
R["loss_epoch2_mean"] = round(float(np.mean(ep2)), 4)
R["loss_last"] = round(float(lh[-1]["loss"]), 4)
R["loss_last5_mean"] = round(float(np.mean([e["loss"] for e in lh[-5:]])), 4)
R["token_acc_epoch2_mean"] = round(float(np.mean([e["mean_token_accuracy"] for e in lh if e["epoch"] > 1.0])), 4)
for lab, nll in (("gold_class", v01["implied_gold_nll_clip_1_16"]["mean"]),
                 ("first_token_family", v01["implied_family_nll_clip_1_16"]["mean"])):
    per_tok_all = R["decision_rows"] * nll / R["trained_tokens_total_per_epoch"]
    per_tok_decrow = nll / R["trained_tokens_mean_decision_rows"]
    R[lab] = {
        "nll_per_decision_row": nll,
        "analyst_style_nll_over_42.2": round(nll / 42.2, 4),
        "contribution_nats_per_trained_token_all_rows": round(per_tok_all, 4),
        "share_of_epoch2_mean_loss": round(per_tok_all / R["loss_epoch2_mean"], 4),
        "share_of_last5_mean_loss": round(per_tok_all / R["loss_last5_mean"], 4),
        "share_within_a_decision_row_at_epoch2_loss": round(per_tok_decrow / R["loss_epoch2_mean"], 4),
        "loss_if_decision_were_perfect_epoch2": round(R["loss_epoch2_mean"] - per_tok_all, 4),
    }
R["first_token_share_of_trained_tokens"] = round(float(is_dec.sum() / trained.sum()), 4)
json.dump(R, open(os.path.join(OUT, "v02_loss_share.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
print("OK_v02")
