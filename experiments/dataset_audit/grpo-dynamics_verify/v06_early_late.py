#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 6: GD-6 (is the early -> late reward shift drift or prompt mix?).

Null model = a FIXED policy: the 120 groups (prompt + its 8 samples) are exchangeable across steps, 4 per step.
Permutation tests shuffle whole groups across the 30 step slots (twin labels travel with their groups).
Statistics: SD of per-step mean reward; OLS slope of step mean on step; A / B early(0-14)-late(15-29) mean reward change;
NEW: the twin-blind drift signature -- the early->late change of P(ask) on A AND on B (a shared drift toward approving
lowers both; prompt mix need not), tested jointly as the twin-balanced mean change of P(ask).
Also the TRL log's 'steps 1-12 vs 13-25' means quoted in the session (log steps = score steps + 1), the zero-prefix prompts'
positions, and the logit change the A ask-rate drop implies.
Outputs: results/v06_early_late.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs")
G = pd.read_csv(os.path.join(OUT, "v01_groups.csv"))
log = [e for e in json.load(open(os.path.join(TL, "m2_grpo", "result.json")))["log_history"] if "loss" in e]
R = {}
rw = np.array([e["reward"] for e in log])
R["trl_log_reward_mean_steps_1_12"] = round(float(rw[:12].mean()), 4)
R["trl_log_reward_mean_steps_13_25"] = round(float(rw[12:25].mean()), 4)
R["trl_log_reward_mean_steps_26_30"] = round(float(rw[25:30].mean()), 4)
G["late"] = (G.step >= 15).astype(int)
logit = lambda p: np.log(p / (1 - p))  # noqa: E731


def stats(df):
    sm = df.groupby("step")["r_mean"].mean().reindex(range(30)).to_numpy()
    slope = np.polyfit(np.arange(30), sm, 1)[0]
    out = {"sd_step_means": float(np.std(sm, ddof=1)), "slope": float(slope)}
    for t in ("A", "B"):
        s = df[df.twin == t]
        out[f"{t}_early"] = float(s.loc[s.late == 0, "r_mean"].mean())
        out[f"{t}_late"] = float(s.loc[s.late == 1, "r_mean"].mean())
        out[f"{t}_diff"] = out[f"{t}_late"] - out[f"{t}_early"]
        out[f"{t}_pask_early"] = float(s.loc[s.late == 0, "p_ask"].mean())
        out[f"{t}_pask_late"] = float(s.loc[s.late == 1, "p_ask"].mean())
        out[f"{t}_pask_diff"] = out[f"{t}_pask_late"] - out[f"{t}_pask_early"]
    out["pask_diff_twin_balanced"] = (out["A_pask_diff"] + out["B_pask_diff"]) / 2
    return out


obs = stats(G)
R["observed"] = {k: round(v, 4) for k, v in obs.items()}
R["n_groups_early_late_by_twin"] = {t: [int(((G.twin == t) & (G.late == 0)).sum()), int(((G.twin == t) & (G.late == 1)).sum())]
                                    for t in ("A", "B")}
rng = np.random.default_rng(0)
NPERM = 20000
slots = G["step"].to_numpy().copy()
perm = {k: [] for k in obs}
for _ in range(NPERM):
    H = G.copy()
    H["step"] = rng.permutation(slots)
    H["late"] = (H.step >= 15).astype(int)
    st = stats(H)
    for k, v in st.items():
        perm[k].append(v)
P = {}
for k in ("slope", "A_diff", "B_diff", "A_pask_diff", "B_pask_diff", "pask_diff_twin_balanced"):
    a = np.array(perm[k])
    P[k] = round(float(np.mean(np.abs(a) >= abs(obs[k]) - 1e-12)), 4)
P["sd_step_means_null_mean"] = round(float(np.mean(perm["sd_step_means"])), 4)
P["sd_step_means_p_ge_observed"] = round(float(np.mean(np.array(perm["sd_step_means"]) >= obs["sd_step_means"])), 4)
R["perm_p_two_sided"] = P
# same without the zero-prefix prompts
Gz = G[~G.zero_prefix.astype(bool)].copy()
R["zero_prefix_sampled"] = G[G.zero_prefix.astype(bool)][["grpo_i", "scenario_id", "level", "step", "pass_gold", "p_ask",
                                                           "classes"]].to_dict("records")
obs_z = {}
for t in ("A", "B"):
    s = Gz[Gz.twin == t]
    obs_z[f"{t}_diff"] = float(s.loc[s.late == 1, "r_mean"].mean() - s.loc[s.late == 0, "r_mean"].mean())
    obs_z[f"{t}_pask_diff"] = float(s.loc[s.late == 1, "p_ask"].mean() - s.loc[s.late == 0, "p_ask"].mean())
obs_z["pask_diff_twin_balanced"] = (obs_z["A_pask_diff"] + obs_z["B_pask_diff"]) / 2
pz = {k: [] for k in obs_z}
for _ in range(NPERM):
    lt = rng.permutation(Gz["late"].to_numpy())
    for t in ("A", "B"):
        m = (Gz.twin == t).to_numpy()
        r = Gz["r_mean"].to_numpy()[m]
        a = Gz["p_ask"].to_numpy()[m]
        l = lt[m]
        pz[f"{t}_diff"].append(r[l == 1].mean() - r[l == 0].mean())
        pz[f"{t}_pask_diff"].append(a[l == 1].mean() - a[l == 0].mean())
    pz["pask_diff_twin_balanced"].append((pz["A_pask_diff"][-1] + pz["B_pask_diff"][-1]) / 2)
R["no_zero_prefix"] = {k: {"obs": round(v, 4), "p": round(float(np.mean(np.abs(pz[k]) >= abs(v) - 1e-12)), 4)}
                       for k, v in obs_z.items()}
# logit change implied by the A ask-rate drop (sample level)
pa_e, pa_l = obs["A_pask_early"], obs["A_pask_late"]
R["A_pask_logit_change"] = round(float(logit(pa_l) - logit(pa_e)), 4)
pb_e, pb_l = obs["B_pask_early"], obs["B_pask_late"]
R["B_pask_logit_change"] = round(float(logit(pb_l) - logit(pb_e)), 4)
json.dump(R, open(os.path.join(OUT, "v06_early_late.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_v06")
