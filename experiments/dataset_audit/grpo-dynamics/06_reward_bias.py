#!/usr/bin/env python3
"""grpo-dynamics lens, step 6: does the reward table push the policy toward approving or toward asking?

(1) Reconstruct TRL's advantages (scale_rewards='group': A = (r - mean_g) / (std_g(unbiased) + 1e-4)) and validate them against
    the logged per-step loss (dapo, on-policy ratio 1: loss ~ -sum_i A_i L_i / sum_i L_i, up to the vLLM importance weights).
(2) Net push per decision class = sum of advantages of the samples in that class. For a softmax policy with a class bias b_k
    shared by all prompts, d(objective)/d b_k = sum_i A_i (1[c_i = k] - pi_k) = sum_i A_i 1[c_i = k] because advantages sum to
    zero inside a group: this is the push GRPO puts on 'approve more' vs 'ask more' irrespective of the prompt.
    Also token-weighted (dapo weights a sample by its length) and per twin.
(3) Per step: the shared ask-vs-approve push against the step's A/B mix (4 random prompts -> unbalanced steps).
(4) Counterfactual reward tables re-scored on the same 960 decisions (same samples, same groups).
(5) Raw-reward calculus for a policy that cannot tell the twins apart (P(ask) = q on every prompt), using M1's observed
    informed/generic split inside each family: dE[r]/dq on the GRPO set's A:B mix.
Output: results/reward_bias.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.reward import A_TABLE, B_ACT, B_APPROVE  # noqa: E402

OUT = os.path.join(HERE, "results")
S = pd.read_csv(os.path.join(OUT, "samples.csv"))
lh = [e for e in json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/result.json")))["log_history"]
      if "reward" in e]
R = {}


def base_reward(dec, cell, table):
    if dec == "invalid":
        return table.get("invalid", -1.0)
    if dec == "lookup":
        return table.get("lookup", -0.1)
    t = table["A"] if cell == "A" else table["B_act"] if cell == "B_act" else table["B_approve"]
    return t.get(dec, -0.5)


CURRENT = {"A": A_TABLE, "B_approve": B_APPROVE, "B_act": B_ACT, "lookup": -0.1, "invalid": -1.0}


def advantages(df, rcol):
    g = df.groupby("row_i")[rcol]
    mean = g.transform("mean")
    std = g.transform(lambda x: x.std(ddof=1))
    return (df[rcol] - mean) / (std + 1e-4)


S["adv"] = advantages(S, "score")
S["fam"] = S["dclass"]

# ---- (1) validate against the logged loss ----
rows = []
for e in lh:
    sub = S[S.step == e["step"] - 1]
    pred = -(sub.adv * sub.n_tokens).sum() / sub.n_tokens.sum()
    rows.append({"log_step": e["step"], "logged_loss": e["loss"], "pred_loss": pred})
V = pd.DataFrame(rows)
R["loss_validation"] = {"corr": round(float(np.corrcoef(V.logged_loss, V.pred_loss)[0, 1]), 4),
                        "slope_logged_on_pred": round(float(np.polyfit(V.pred_loss, V.logged_loss, 1)[0]), 4),
                        "mean_abs_diff": round(float((V.logged_loss - V.pred_loss).abs().mean()), 5)}
R["recomputed_scores_match"] = bool(np.allclose(
    S.apply(lambda r: base_reward(r.decision, r.cell, CURRENT), axis=1) + S["format"] + S["verbosity"], S["score"], atol=1e-6))


# ---- (2) net push per class ----
def pushes(df, acol="adv"):
    out = {}
    tok = df[acol] * df["n_tokens"] / df.groupby("step")["n_tokens"].transform("sum")
    for (tw, fam), sub in df.assign(tokw=tok).groupby(["twin", "fam"]):
        out[f"{tw}|{fam}"] = {"n": int(len(sub)), "sum_adv": round(sub[acol].sum(), 3), "mean_adv": round(sub[acol].mean(), 3),
                              "frac_pos_adv": round((sub[acol] > 1e-9).mean(), 3), "token_weighted": round(sub["tokw"].sum(), 4)}
    tot = {}
    for fam, sub in df.assign(tokw=tok).groupby("fam"):
        tot[fam] = {"sum_adv": round(sub[acol].sum(), 3), "token_weighted": round(sub["tokw"].sum(), 4), "n": int(len(sub))}
    return out, tot

R["push_current"], R["push_current_total"] = pushes(S)
R["mean_tokens_by_family"] = S.groupby(["twin", "fam"])["n_tokens"].mean().round(1).to_dict()
R["mean_tokens_by_family"] = {f"{a}|{b}": v for (a, b), v in R["mean_tokens_by_family"].items()}
# decisions with positive advantage that are wrong for the twin
A_ = S[S.twin == "A"]
R["A_wrong_but_reinforced"] = {d: {"n": int((A_.decision == d).sum()), "n_pos_adv": int(((A_.decision == d) & (A_.adv > 1e-9)).sum()),
                                   "sum_pos_adv": round(A_[(A_.decision == d) & (A_.adv > 0)].adv.sum(), 3)}
                               for d in ("approve_informed", "approve_gold", "lookup", "ask_generic", "statement", "approve_generic")}
B_ = S[S.twin == "B"]
R["B_wrong_but_reinforced"] = {d: {"n": int((B_.decision == d).sum()), "n_pos_adv": int(((B_.decision == d) & (B_.adv > 1e-9)).sum())}
                               for d in ("lookup", "ask_generic", "ask_informed", "commit_other", "commit_gold")}

# ---- (3) per-step shared push vs the step's A/B mix ----
st = []
for s, sub in S.groupby("step"):
    nA = sub[sub.twin == "A"].row_i.nunique()
    st.append({"step": s, "n_A_prompts": nA, "push_ask": sub[sub.fam == "ask"].adv.sum(), "push_approve": sub[sub.fam == "approve"].adv.sum()})
P = pd.DataFrame(st)
P["ask_minus_approve"] = P.push_ask - P.push_approve
R["per_step_push"] = {"corr_ask_minus_approve_vs_nA": round(float(np.corrcoef(P.ask_minus_approve, P.n_A_prompts)[0, 1]), 4),
                      "by_nA": P.groupby("n_A_prompts").agg(n_steps=("step", "size"), mean_ask_minus_approve=("ask_minus_approve", "mean")).round(3).to_dict("index"),
                      "n_A_prompts_distribution": {int(k): int(v) for k, v in P.n_A_prompts.value_counts().sort_index().items()},
                      "sum_abs_step_push": round(P.ask_minus_approve.abs().sum(), 3), "abs_sum_step_push": round(abs(P.ask_minus_approve.sum()), 3)}

# ---- (4) counterfactual tables ----
def tbl(a_over=None, bapp_over=None, bact_over=None, lookup=-0.1):
    return {"A": {**A_TABLE, **(a_over or {})}, "B_approve": {**B_APPROVE, **(bapp_over or {})}, "B_act": {**B_ACT, **(bact_over or {})},
            "lookup": lookup, "invalid": -1.0}

TABLES = {
    "current": CURRENT,
    "no_card_credit_on_A(approve_informed/gold -0.5), lookup -0.5": tbl({"approve_informed": -0.5, "approve_gold": -0.5}, lookup=-0.5),
    "symmetric(B asks -1.0)": tbl(bapp_over={"ask_informed": -1.0, "ask_generic": -1.0, "tell_informed": -1.0},
                                  bact_over={"ask_informed": -1.0, "ask_generic": -1.0, "tell_informed": -1.0}),
    "binary(gold +1 else -1)": {"A": {"ask_informed": 1.0, "tell_informed": 1.0}, "B_approve": {"approve_generic": 1.0, "approve_informed": 1.0, "approve_gold": 1.0},
                                "B_act": {"commit_gold": 1.0}, "lookup": -1.0, "invalid": -1.0, "_default": -1.0},
}


def base_reward2(dec, cell, table):
    if "_default" in table:
        if dec in ("lookup", "invalid"):
            return -1.0
        t = table["A"] if cell == "A" else table["B_act"] if cell == "B_act" else table["B_approve"]
        return t.get(dec, -1.0)
    return base_reward(dec, cell, table)


R["counterfactual"] = {}
for name, table in TABLES.items():
    df = S.copy()
    df["r"] = df.apply(lambda r: base_reward2(r.decision, r.cell, table), axis=1) + df["format"]
    df["a"] = advantages(df, "r")
    out = {}
    for tw in ("A", "B"):
        sub = df[df.twin == tw]
        out[tw] = {f: round(sub[sub.fam == f].a.sum(), 3) for f in ("ask", "approve", "commit", "lookup", "statement", "invalid")}
    tot = {f: round(df[df.fam == f].a.sum(), 3) for f in ("ask", "approve", "commit", "lookup", "statement", "invalid")}
    zero = df.groupby("row_i")["r"].std(ddof=1).pipe(lambda x: np.isclose(x, 0)).mean()
    out["total"] = tot
    out["shared_bias_ask_minus_approve"] = round(tot["ask"] - tot["approve"], 3)
    out["discriminative_(askA-askB)-(appA-appB)"] = round((out["A"]["ask"] - out["B"]["ask"]) - (out["A"]["approve"] - out["B"]["approve"]), 3)
    out["zero_var_frac"] = round(float(zero), 4)
    R["counterfactual"][name] = out

# ---- (5) raw-reward calculus for a non-discriminating policy ----
def fam_value(cell, fam, table=CURRENT):
    sub = S[(S.cell == cell) & (S.fam == fam)]
    if not len(sub):
        return None
    return float(sub.decision.map(lambda d: base_reward(d, cell, table)).mean())

vals = {c: {f: fam_value(c, f) for f in ("ask", "approve")} for c in ("A", "B_approve", "B_act")}
share = S.drop_duplicates("row_i").cell.value_counts(normalize=True).to_dict()
grpo_rows = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/grpo_train.jsonl"))]
cells_all = pd.Series(["A" if r["twin"].startswith("A") else ("B_act" if json.loads(r["scenario"])["expected"]["decision"] == "act" else "B_approve")
                       for r in grpo_rows]).value_counts(normalize=True).to_dict()
dEdq = sum(cells_all[c] * (vals[c]["ask"] - vals[c]["approve"]) for c in vals if vals[c]["ask"] is not None)
R["nondiscriminating_policy"] = {"family_value_given_M1_split": vals, "cell_share_grpo_set": {k: round(v, 4) for k, v in cells_all.items()},
                                 "dE_reward_d_P(ask)": round(dEdq, 4),
                                 "reading": "negative = a policy that ignores the twin gains reward by approving more"}
# break-even: what fraction of A asks must be informed for dE/dq = 0 (approve values as observed)
fA_app = vals["A"]["approve"]
for f in np.linspace(0, 1, 101):
    askA = f * 1.0 + (1 - f) * (-0.5)
    d = cells_all["A"] * (askA - fA_app) + cells_all["B_approve"] * (vals["B_approve"]["ask"] - vals["B_approve"]["approve"]) + \
        cells_all.get("B_act", 0) * (vals["B_act"]["ask"] - vals["B_act"]["approve"])
    if d >= 0:
        R["nondiscriminating_policy"]["informed_share_of_A_asks_needed_for_neutral"] = round(float(f), 2)
        break
R["nondiscriminating_policy"]["observed_informed_share_of_A_asks"] = round(float(
    S[(S.cell == "A") & (S.fam == "ask")].decision.isin(["ask_informed", "tell_informed"]).mean()), 4)
json.dump(R, open(os.path.join(OUT, "reward_bias.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_06")
