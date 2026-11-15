#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 4: GRPO advantages, decision 'pushes', per-step composition (GD-2 signal part, GD-3, GD-4,
GD-5) -- independent re-derivation from scores.jsonl + grpo_train.jsonl + the TRL log.

Advantage (TRL 1.13, scale_rewards='group'): A_i = (r_i - mean_g) / (std_g + 1e-4), std unbiased (torch.std default).
Push on a decision family f in a group = sum_{i in f} A_i (= d/dz_f of sum_i A_i log softmax(z)_{f_i}, since sum_i A_i = 0).
Net ask-vs-approve push of a group N_g = push_ask - push_approve (ask = ask_* or tell_*; approve = approve_*).
Twin-aware split over groups: discriminative D = sum_{A} N_g - sum_{B} N_g ; shared (twin-blind) S = sum_{A} N_g + sum_{B} N_g.
Also validates the reconstruction against the logged dapo loss: loss ~ -sum_i A_i n_i / sum_i n_i (token-level mean).
Outputs: results/v04_advantages.json, results/v04_steps.csv
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs")
S = pd.read_csv(os.path.join(OUT, "v01_samples.csv"))
grpo_log = [e for e in json.load(open(os.path.join(TL, "m2_grpo", "result.json")))["log_history"] if "loss" in e]
R = {}


def fam(d):
    if d.startswith(("ask", "tell")):
        return "ask"
    if d.startswith("approve"):
        return "approve"
    if d.startswith("commit"):
        return "commit"
    return d  # lookup / statement / invalid


S["fam"] = S["decision"].map(fam)


def advantages(df, col):
    out = np.zeros(len(df))
    for _, idx in df.groupby(["step", "grpo_i"]).indices.items():
        r = df[col].to_numpy()[idx]
        out[idx] = (r - r.mean()) / (r.std(ddof=1) + 1e-4)
    return out


S["adv"] = advantages(S, "score")


def pushes(df, adv="adv"):
    g = df.groupby(["step", "grpo_i"])
    rows = []
    for (st, gi), s in g:
        pa = s.loc[s.fam == "ask", adv].sum()
        pp = s.loc[s.fam == "approve", adv].sum()
        rows.append({"step": st, "grpo_i": gi, "twin": s.twin.iloc[0], "cell": s.cell.iloc[0], "push_ask": pa,
                     "push_approve": pp, "N": pa - pp,
                     "push_commit": s.loc[s.fam == "commit", adv].sum(), "push_lookup": s.loc[s.fam == "lookup", adv].sum()})
    return pd.DataFrame(rows)


P = pushes(S)
NA, NB = P.loc[P.twin == "A", "N"].sum(), P.loc[P.twin == "B", "N"].sum()
R["sum_N_A_groups"] = round(float(NA), 3)
R["sum_N_B_groups"] = round(float(NB), 3)
R["discriminative_D"] = round(float(NA - NB), 3)
R["shared_S"] = round(float(NA + NB), 3)
R["shared_over_discriminative"] = round(float(abs(NA + NB) / (NA - NB)), 4)
R["sum_push_ask_on_A"] = round(float(P.loc[P.twin == "A", "push_ask"].sum()), 3)
R["sum_push_approve_on_A"] = round(float(P.loc[P.twin == "A", "push_approve"].sum()), 3)
R["sum_push_ask_on_B"] = round(float(P.loc[P.twin == "B", "push_ask"].sum()), 3)
R["sum_push_approve_on_B"] = round(float(P.loc[P.twin == "B", "push_approve"].sum()), 3)
R["sum_push_lookup_on_A"] = round(float(P.loc[P.twin == "A", "push_lookup"].sum()), 3)
R["groups_A"], R["groups_B"] = int((P.twin == "A").sum()), int((P.twin == "B").sum())
# per-group mean N by twin: does the shared term come from composition (more B groups) or per-group asymmetry?
mA, mB = P.loc[P.twin == "A", "N"].mean(), P.loc[P.twin == "B", "N"].mean()
R["mean_N_per_A_group"] = round(float(mA), 4)
R["mean_N_per_B_group"] = round(float(mB), 4)
R["shared_if_balanced_60_60"] = round(float(60 * mA + 60 * mB), 3)
R["shared_from_composition_(nB-nA)*mean|N|"] = round(float((P.twin == "B").sum() - (P.twin == "A").sum()) * float(
    (abs(mA) + abs(mB)) / 2), 3)

# per step
st = P.groupby("step").agg(N=("N", "sum"), nA=("twin", lambda x: int((x == "A").sum())))
st["absN"] = st["N"].abs()
R["per_step_corr_N_vs_nA"] = round(float(np.corrcoef(st["N"], st["nA"])[0, 1]), 4)
R["per_step_nA_counts"] = {str(k): int(v) for k, v in st["nA"].value_counts().sort_index().items()}
R["per_step_mean_N_by_nA"] = {str(k): round(float(v), 3) for k, v in st.groupby("nA")["N"].mean().items()}
R["sum_step_N"] = round(float(st["N"].sum()), 3)
R["sum_abs_step_N"] = round(float(st["absN"].sum()), 3)
# per-step N if each step's groups were re-paired 2A+2B (expected |shared| per step from per-group asymmetry only)
R["per_step_N_explained_by_nA_R2"] = round(float(np.corrcoef(st["N"], st["nA"])[0, 1] ** 2), 4)

# momentum view: Adam's first moment averages ~10 steps; how much of the per-step swing survives an EMA(0.9)?
ema, e = [], 0.0
for t, v in enumerate(st["N"].to_numpy(), 1):
    e = 0.9 * e + 0.1 * v
    ema.append(e / (1 - 0.9 ** t))
st["N_ema"] = ema
R["per_step_N_mean"] = round(float(st["N"].mean()), 3)
R["per_step_N_sd"] = round(float(st["N"].std(ddof=1)), 3)
R["per_step_N_ema_sd"] = round(float(np.std(ema[5:], ddof=1)), 3)
st.to_csv(os.path.join(OUT, "v04_steps.csv"))

# wrong-but-reinforced samples on A
A = S[S.twin == "A"]
R["A_reinforced_(adv>0)_by_class"] = {d: f"{int((s.adv > 0).sum())}/{len(s)}" for d, s in A.groupby("decision")}
B = S[S.twin == "B"]
R["B_reinforced_(adv>0)_by_class"] = {d: f"{int((s.adv > 0).sum())}/{len(s)}" for d, s in B.groupby("decision")}

# loss reconstruction vs the TRL log (log step k+1 <-> score step k)
rec = []
for k in range(30):
    s = S[S.step == k]
    n = s["n_tokens"].to_numpy().astype(float)
    rec.append(-(s["adv"].to_numpy() * n).sum() / n.sum())
logged = np.array([e["loss"] for e in grpo_log])
rec = np.array(rec)
R["loss_recon_corr"] = round(float(np.corrcoef(rec, logged)[0, 1]), 4)
R["loss_recon_slope"] = round(float(np.polyfit(rec, logged, 1)[0]), 4)
R["logged_reward_vs_scores_maxdiff"] = float(np.max(np.abs(np.array([e["reward"] for e in grpo_log]) -
                                                         S.groupby("step")["score"].mean().to_numpy())))
R["logged_zero_std_vs_scores"] = [float(e["frac_reward_zero_std"]) for e in grpo_log] == [
    float((S[S.step == k].groupby("grpo_i")["score"].std(ddof=1) < 1e-9).mean()) for k in range(30)]

# alternative reward tables (re-score from the logged class + logged format/verbosity)
A_T = {"ask_informed": 1.0, "tell_informed": 0.5, "approve_gold": 0.5, "approve_informed": 0.25, "statement": -0.25,
       "ask_generic": -0.5, "approve_generic": -1.0, "commit_gold": -1.0, "commit_harmful": -1.0, "commit_other": -1.0}
BAP = {"approve_generic": 1.0, "approve_informed": 1.0, "approve_gold": 1.0, "commit_gold": -0.5, "commit_harmful": -1.0,
       "commit_other": -0.5, "ask_informed": -0.5, "ask_generic": -0.5, "tell_informed": -0.5, "statement": -0.5}
BAC = {"commit_gold": 1.0, "approve_generic": 0.5, "approve_informed": 0.5, "approve_gold": 0.5, "commit_other": -0.5,
       "commit_harmful": -1.0, "ask_informed": -0.5, "ask_generic": -0.5, "tell_informed": -0.5, "statement": -0.5}


def base_of(row, a_t, bap, bac, lookup=-0.1):
    d = row.decision
    if d == "invalid":
        return -1.0
    if d == "lookup":
        return lookup
    t = a_t if row.twin == "A" else (bac if row.cell == "B_act" else bap)
    return t.get(d, -0.5)


S["base_re"] = [base_of(r, A_T, BAP, BAC) for r in S.itertuples()]
R["rescored_base_matches_logged"] = int((np.abs(S["base_re"] - S["base"]) < 1e-9).sum())
alts = {
    "no_card_credit_on_A": ({**A_T, "approve_gold": -1.0, "approve_informed": -1.0}, BAP, BAC, -0.1),
    "B_asks_minus1": (A_T, {**BAP, "ask_informed": -1.0, "ask_generic": -1.0, "tell_informed": -1.0}, BAC, -0.1),
    "binary_gold": ({k: (1.0 if k in ("ask_informed", "tell_informed") else -1.0) for k in A_T},
                    {k: (1.0 if k.startswith("approve") else -1.0) for k in BAP},
                    {k: (1.0 if k == "commit_gold" else -1.0) for k in BAC}, -1.0),
    "lookup_minus_0.5": (A_T, BAP, BAC, -0.5),
}
R["alt_tables"] = {}
for name, (at, bp, bc, lk) in alts.items():
    S["r_alt"] = [base_of(r, at, bp, bc, lk) for r in S.itertuples()] + S["fmt"] + S["verb"]
    S["adv_alt"] = advantages(S, "r_alt")
    Q = pushes(S, "adv_alt")
    a, b = Q.loc[Q.twin == "A", "N"].sum(), Q.loc[Q.twin == "B", "N"].sum()
    R["alt_tables"][name] = {"discriminative": round(float(a - b), 3), "shared": round(float(a + b), 3),
                             "zero_var_groups": int((S.groupby(["step", "grpo_i"])["r_alt"].std(ddof=1) < 1e-9).sum()),
                             "A_lookup_adv_pos": int(((S.twin == "A") & (S.decision == "lookup") & (S.adv_alt > 0)).sum())}

# twin-blind derivative of E[reward] wrt P(ask) given M1's within-family class mix
def fam_value(sub, f):
    s = sub[sub.fam == f]
    return float(s["base"].mean()) if len(s) else np.nan


vals = {}
for c, s in S.groupby("cell"):
    vals[c] = {"ask": fam_value(s, "ask"), "approve": fam_value(s, "approve"), "commit": fam_value(s, "commit")}
R["family_values_by_cell"] = {c: {k: round(v, 4) for k, v in d.items()} for c, d in vals.items()}
# proceed value = M1's empirical mix of approve/commit within the proceed family
proceed = {}
for c, s in S.groupby("cell"):
    q = s[s.fam.isin(["approve", "commit"])]
    proceed[c] = float(q["base"].mean())
R["proceed_value_by_cell"] = {c: round(v, 4) for c, v in proceed.items()}
grpo_counts = {"A": 247, "B_approve": 231, "B_act": 18}
sample_counts = S.groupby("cell")["grpo_i"].nunique().to_dict()
for lab, w in (("grpo_set_496", grpo_counts), ("sampled_120", sample_counts)):
    tot = sum(w.values())
    R[f"dE_dPask_twin_blind_{lab}"] = round(sum(w[c] / tot * (vals[c]["ask"] - proceed[c]) for c in w), 4)
    R[f"dE_dPask_twin_blind_{lab}_if_all_A_asks_informed"] = round(
        sum(w[c] / tot * ((1.0 if c == "A" else vals[c]["ask"]) - proceed[c]) for c in w), 4)
json.dump(R, open(os.path.join(OUT, "v04_advantages.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
print("OK_v04")
