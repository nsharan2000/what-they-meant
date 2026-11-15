#!/usr/bin/env python3
"""grpo-dynamics lens, step 3: were the GRPO prompts literally SFT training prompts, and what does M1's sampled decision
distribution imply about the SFT loss on the decision?

(a) For every GRPO row: is there an SFT (train or val) row with the same scenario_id/level/prompt_setting, turn 'decision',
    and an identical prompt (all messages equal)? Is the SFT target's decision class the GRPO gold class?
(b) Decision-token arithmetic: M1 sampled at temperature 1.0 / top_p 1.0 (train_grpo.py), so the sampled class frequencies
    estimate M1's probability of the gold decision class on each prompt. The class is fixed by the first target tokens
    ('<' + 'tool' for a tool call vs a word for a chat message), so -log P(gold class) is a lower bound on the NLL the SFT
    loss put on those first tokens. Spread over the trained tokens of a decision row (mean 42.2 trained tokens per SFT row,
    m1_sft/guards.json) this is the share of the per-token SFT loss the decision can account for.
Outputs: results/sft_overlap.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
D = os.path.join(ROOT, "dataset/curated_v121")
grpo = [json.loads(l) for l in open(os.path.join(D, "grpo_train.jsonl"))]
sft = [json.loads(l) for l in open(os.path.join(D, "sft_train.jsonl"))]
val = [json.loads(l) for l in open(os.path.join(D, "sft_val.jsonl"))]
G = pd.read_csv(os.path.join(OUT, "groups.csv"))
guards = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m1_sft/guards.json")))
m1 = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m1_sft/result.json")))


def key(r):
    return (r["scenario_id"], r["level"], r["prompt_setting"])


def target_class(r):
    t = r["completion"][-1]["content"].lstrip()
    if t.startswith("<tool_call>"):
        name = json.loads(t[len("<tool_call>"):t.index("</tool_call>")])["name"]
        return "tool:" + name
    return "message"


sft_dec = {}
for src, rows in (("sft_train", sft), ("sft_val", val)):
    for i, r in enumerate(rows):
        if r.get("turn") == "decision":
            sft_dec.setdefault(key(r), []).append((src, i, r))
R = {"n_grpo": len(grpo)}
res = []
for gi, r in enumerate(grpo):
    hits = sft_dec.get(key(r), [])
    same = [(s, i) for s, i, x in hits if x["prompt"] == r["prompt"]]
    tc = [target_class(x) for s, i, x in hits if x["prompt"] == r["prompt"]]
    res.append({"grpo_i": gi, "twin": r["twin"][0], "level": r["level"], "n_key_hits": len(hits), "n_identical": len(same),
                "where": "+".join(sorted({s for s, _ in same})), "sft_target_class": tc[0] if tc else None})
X = pd.DataFrame(res)
R["grpo_rows_with_identical_sft_decision_prompt"] = int((X.n_identical > 0).sum())
R["by_source"] = {k: int(v) for k, v in X["where"].value_counts().items()}
R["sft_target_class_by_twin"] = {t: {k: int(v) for k, v in sub["sft_target_class"].value_counts(dropna=False).items()}
                                 for t, sub in X.groupby("twin")}
# the 120 sampled prompts
Xs = X[X.grpo_i.isin(G.row_i)]
R["sampled_prompts_identical_in_sft_train"] = int((Xs["where"].str.contains("sft_train")).sum())
R["sampled_prompts_n"] = int(len(Xs))
sft_dec_rows = sum(1 for r in sft if r.get("turn") == "decision")
R["sft_train_turn_counts"] = {k: int(v) for k, v in pd.Series([r.get("turn") for r in sft]).value_counts().items()}
R["sft_train_decision_rows"] = sft_dec_rows

# ---- (b) decision-token NLL implied by M1's sampled class frequencies ----
p = G["pass_top"].clip(lower=0.5 / 8, upper=1 - 0.5 / 8).to_numpy()   # add-half smoothing on 8 samples
nll = -np.log(p)
mean_tokens = guards["sft_train"]["trained_tokens_mean"]
R["decision_nll"] = {
    "mean_nll_gold_class_nats": round(float(nll.mean()), 4),
    "median_nll": round(float(np.median(nll)), 4),
    "by_twin": {t: round(float(-np.log(sub["pass_top"].clip(0.5 / 8, 1 - 0.5 / 8)).mean()), 4) for t, sub in G.groupby("twin")},
    "trained_tokens_mean_per_sft_row": mean_tokens,
    "decision_share_of_per_token_loss": round(float(nll.mean() / mean_tokens), 4),
    "note": "lower bound on the first-token loss if the class is decided by one token; smoothing p in [1/16, 15/16]",
}
lh = [e for e in m1["log_history"] if "loss" in e and "eval_loss" not in e]
ep2 = [e["loss"] for e in lh if e["epoch"] > 1.0]
R["m1_train_loss_epoch2_mean"] = round(float(np.mean(ep2)), 4)
R["m1_last_loss"] = round(lh[-1]["loss"], 4)
R["m1_eval_loss"] = [round(e["eval_loss"], 4) for e in m1["log_history"] if "eval_loss" in e]
R["m1_mean_token_accuracy_epoch2"] = round(float(np.mean([e["mean_token_accuracy"] for e in lh if e["epoch"] > 1.0])), 4)
json.dump(R, open(os.path.join(OUT, "sft_overlap.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
print("OK_03")
