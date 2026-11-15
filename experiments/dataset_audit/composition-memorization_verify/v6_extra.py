"""Extra checks: (1) bias-corrected first-token-type NLL (beta-binomial fit to the per-prompt counts) and a greedy proxy
(majority of 8 samples) for M1's decision accuracy, since evaluation decodes at temperature 0 (wtm/run.py default, PLAN.md);
(2) CM-4 look-up composition; (3) act- vs approve-expected twin-B prompts.
Inputs: results/v2_grpo_prompts.csv, scores.jsonl, curated sft_train / grpo_train.
Output: results/v6_extra.json
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vcommon import ROOT, RES, cur, dump, load_jsonl, target, request, is_call, call_name

pp = pd.read_csv(os.path.join(RES, "v2_grpo_prompts.csv"))
out = {}

# ---- (1) beta-binomial fit of the per-prompt count of gold-TYPE samples (k of n=8), by twin and pooled ---------------------
from math import lgamma


def betaln(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def bb_loglik(a, b, k, n):
    return sum(betaln(a + ki, b + n - ki) - betaln(a, b) for ki in k)   # binomial coefficient omitted (constant)


def fit(k, n=8):
    grid = np.exp(np.linspace(np.log(0.05), np.log(50), 120))
    best = (-1e18, None, None)
    for a in grid:
        for b in grid:
            ll = bb_loglik(a, b, k, n)
            if ll > best[0]:
                best = (ll, a, b)
    return best


def e_neglog(a, b, m=20001):
    x = np.linspace(1e-6, 1 - 1e-6, m)
    logpdf = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b)
    w = np.exp(logpdf)
    w /= w.sum()
    return float((-np.log(x) * w).sum()), float((x * w).sum()), float((w * (x > 0.5)).sum())


pp["k_type"] = (pp.p_goldtype * pp.n).round().astype(int)
res = {}
for name, sub in (("all", pp), ("A", pp[pp.twin == "A"]), ("B", pp[pp.twin == "B"])):
    ll, a, b = fit(sub.k_type.tolist())
    enl, ep, p_gt_half = e_neglog(a, b)
    res[name] = {"n_prompts": int(len(sub)), "alpha": round(a, 3), "beta": round(b, 3), "E_p_goldtype": round(ep, 4),
                 "E_neglog_p_goldtype_nats": round(enl, 4), "share_prompts_p_gt_0.5": round(p_gt_half, 4),
                 "plugin_add0.5_nll": round(float((-np.log((sub.k_type + 0.5) / (sub.n + 1))).mean()), 4)}
out["first_token_type_betabinomial"] = res
# same for the strict-correct CLASS
for name, sub in (("all", pp), ("A", pp[pp.twin == "A"]), ("B", pp[pp.twin == "B"])):
    ll, a, b = fit(sub.correct.astype(int).tolist())
    enl, ep, p_gt_half = e_neglog(a, b)
    res2 = out.setdefault("correct_class_betabinomial", {})
    res2[name] = {"alpha": round(a, 3), "beta": round(b, 3), "E_p_correct": round(ep, 4), "E_neglog_nats": round(enl, 4),
                  "share_prompts_p_gt_0.5": round(p_gt_half, 4)}
# greedy proxy: majority of the 8 samples (ties count 0.5)
maj = np.where(pp.correct > 4, 1.0, np.where(pp.correct == 4, 0.5, 0.0))
out["majority_of_8_accuracy"] = round(float(maj.mean()), 4)
out["majority_of_8_accuracy_by_twin"] = {t: round(float(maj[pp.twin == t].mean()), 4) for t in ("A", "B")}
# loss share implied by the bias-corrected type NLL
T, D = 38160, 497
out["implied_nats_per_trained_token_type_nll_corrected"] = round(D * res["all"]["E_neglog_p_goldtype_nats"] / T, 5)

# ---- (2) CM-4 look-up composition ------------------------------------------------------------------------------------------
tr = cur("sft_train")
look = [r for r in tr if r["turn"] != "decision"]
by_t = defaultdict(set)
for r in look:
    by_t[target(r)].add(r["twin"][0])
shared = {t for t, v in by_t.items() if v >= {"A", "B"}}
out["lookup_rows"] = len(look)
out["lookup_rows_share"] = round(len(look) / len(tr), 4)
out["lookup_targets"] = len(by_t)
out["lookup_targets_shared_by_both_twins"] = len(shared)
out["rows_with_twin_shared_lookup_target"] = sum(target(r) in shared for r in look)
two = [r for r in tr if len(r["prompt"]) == 2]
out["system_user_prompts"] = len(two)
out["system_user_prompt_turns"] = dict(Counter(r["turn"] for r in two))
out["system_user_decision_scenarios"] = sorted({r["scenario_id"] for r in two if r["turn"] == "decision"})
out["decode_qr_rows"] = sum('"decode_qr"' in target(r) for r in look)
# ---- (3) act vs approve twin-B prompts (M1 samples) --------------------------------------------------------------------------
S = load_jsonl(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))
grpo = cur("grpo_train")
exp = {}
for g in grpo:
    s = json.loads(g["scenario"])
    exp[(g["scenario_id"], g["level"])] = s["expected"]["decision"]
bb = Counter()
for s in S:
    m = s["meta"]
    if m["twin"].startswith("B"):
        bb[(exp[(m["scenario_id"], m["level"])], s["components"]["decision"])] += 1
out["twinB_class_by_expected"] = {f"{e}:{d}": v for (e, d), v in sorted(bb.items())}
dump(out, "v6_extra.json")
print(json.dumps(out, indent=1))
