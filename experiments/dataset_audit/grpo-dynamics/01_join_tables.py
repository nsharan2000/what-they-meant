#!/usr/bin/env python3
"""grpo-dynamics lens, step 1: join the M2 GRPO reward log with the GRPO training rows and describe the M1 policy.

Inputs (read-only):
  experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl   960 scored completions (30 steps x 4 prompts x 8)
  experiments/training_eval/results/train_logs/m2_grpo/result.json    TRL log_history of the same run
  dataset/curated_v121/grpo_train.jsonl                               496 GRPO prompts (scenario spec, env, level, setting)
  experiments/dataset_audit/results/token_stats.jsonl                 exact Tiny Aya prompt token counts per row
Outputs (experiments/dataset_audit/grpo-dynamics/results/):
  samples.csv          one row per completion, joined
  groups.csv           one row per prompt group (8 completions): pass rates, reward mean/std, class counts
  tables_01.json       decision-class distributions, pass-rate tables, zero-variance fractions, log consistency checks
Pass definitions: top = reward base is the table maximum (+1.0: A ask_informed, B-approve approve_*, B-act commit_gold);
strict / lenient = wtm.episode.decision_grade (the evaluation grader's success / success_lenient).
"""
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.episode import decision_grade  # noqa: E402  (stdlib-only package)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl")
RESULT = os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/result.json")
GRPO = os.path.join(ROOT, "dataset/curated_v121/grpo_train.jsonl")
TOKS = os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")

scores = [json.loads(l) for l in open(SCORES) if l.strip()]
rows = [json.loads(l) for l in open(GRPO) if l.strip()]
toks = {r["i"]: r for r in (json.loads(l) for l in open(TOKS)) if r["file"] == "grpo_train.jsonl"}

checks = {}
key_count = collections.Counter((r["scenario_id"], r["level"]) for r in rows)
checks["grpo_rows"] = len(rows)
checks["grpo_unique_sid_level"] = len(key_count)
checks["grpo_dup_sid_level"] = sum(v > 1 for v in key_count.values())
idx = {(r["scenario_id"], r["level"]): i for i, r in enumerate(rows)}
for i, r in enumerate(rows):
    t = toks[i]
    assert t["scenario_id"] == r["scenario_id"] and t["level"] == r["level"] and t["prompt_setting"] == r["prompt_setting"], i

recs = []
for k, s in enumerate(scores):
    m = s["meta"]
    i = idx[(m["scenario_id"], m["level"])]
    r = rows[i]
    scn = json.loads(r["scenario"])
    assert scn["twin"] == m["twin"] == r["twin"]
    d = s["components"]["decision"]
    exp = scn["expected"]["decision"]
    g = decision_grade(scn, d) if d not in ("lookup", "invalid", "truncated") else {"success": False, "success_lenient": False}
    recs.append({
        "k": k, "step": m["step"], "row_i": i, "scenario_id": m["scenario_id"], "pair_id": r["pair_id"], "level": m["level"],
        "prompt_setting": r["prompt_setting"], "env": r["env_id"], "twin": r["twin"][0], "expected": exp,
        "cell": "A" if r["twin"].startswith("A") else ("B_act" if exp == "act" else "B_approve"),
        "decision": d, "score": s["score"], "base": s["components"]["base"], "format": s["components"].get("format", 0.0),
        "verbosity": s["components"].get("verbosity", 0.0), "n_tokens": m["n_tokens"], "end_id": m["end_id"],
        "truncated": m["truncated"], "prompt_tokens": toks[i]["prompt_tokens"],
        "pass_top": float(s["components"]["base"] >= 1.0 - 1e-9), "pass_strict": float(bool(g["success"])),
        "pass_lenient": float(bool(g["success_lenient"])),
        "synthetic": bool(scn.get("synthetic")),
    })
df = pd.DataFrame(recs)
df["dclass"] = df["decision"].map(lambda d: "ask" if d.startswith(("ask", "tell")) else
                                  "approve" if d.startswith("approve") else "commit" if d.startswith("commit") else
                                  "lookup" if d == "lookup" else "statement" if d == "statement" else "invalid")
df.to_csv(os.path.join(OUT, "samples.csv"), index=False)

# ---- structure checks: each step = 4 distinct prompts x 8 completions; each prompt sampled in exactly one step ----
per_step = df.groupby("step").agg(n=("k", "size"), n_prompts=("row_i", "nunique"))
checks["steps"] = int(df["step"].nunique())
checks["step_min_max"] = [int(df["step"].min()), int(df["step"].max())]
checks["all_steps_4x8"] = bool(((per_step["n"] == 32) & (per_step["n_prompts"] == 4)).all())
gsize = df.groupby(["step", "row_i"]).size()
checks["all_groups_size_8"] = bool((gsize == 8).all())
checks["distinct_prompts"] = int(df["row_i"].nunique())
checks["prompt_in_more_than_one_step"] = int((df.groupby("row_i")["step"].nunique() > 1).sum())
checks["distinct_scenarios"] = int(df["scenario_id"].nunique())
checks["distinct_pairs"] = int(df["pair_id"].nunique())
checks["truncated"] = int(df["truncated"].sum())
checks["end_id_counts"] = {str(k): int(v) for k, v in df["end_id"].value_counts().items()}
checks["format_bonus_counts"] = {str(k): int(v) for k, v in df["format"].value_counts().items()}
checks["verbosity_nonzero"] = int((df["verbosity"] != 0).sum())
checks["score_minus_base_values"] = {str(round(k, 4)): int(v) for k, v in (df["score"] - df["base"]).round(4).value_counts().items()}

# ---- group table ----
def grp(g):
    sc = g["score"].to_numpy()
    cnt = g["decision"].value_counts()
    maj = cnt.index[0]
    return pd.Series({
        "step": int(g["step"].iloc[0]), "scenario_id": g["scenario_id"].iloc[0], "pair_id": g["pair_id"].iloc[0],
        "level": g["level"].iloc[0], "prompt_setting": g["prompt_setting"].iloc[0], "env": g["env"].iloc[0],
        "twin": g["twin"].iloc[0], "cell": g["cell"].iloc[0], "prompt_tokens": int(g["prompt_tokens"].iloc[0]),
        "pass_top": g["pass_top"].mean(), "pass_strict": g["pass_strict"].mean(), "pass_lenient": g["pass_lenient"].mean(),
        "reward_mean": sc.mean(), "reward_std1": sc.std(ddof=1), "reward_std0": sc.std(ddof=0),
        "zero_var": bool(np.isclose(sc.std(ddof=1), 0.0)), "n_classes": int(cnt.size),
        "majority": maj, "majority_share": cnt.iloc[0] / len(g),
        "p_ask": (g["dclass"] == "ask").mean(), "p_approve": (g["dclass"] == "approve").mean(),
        "p_commit": (g["dclass"] == "commit").mean(), "p_lookup": (g["dclass"] == "lookup").mean(),
        "classes": json.dumps(cnt.to_dict()),
    })

G = df.groupby("row_i").apply(grp, include_groups=False).reset_index()
G["pass_bin"] = np.where(G["pass_top"] == 0, "0", np.where(G["pass_top"] == 1, "1", "mid"))
G["majority_is_gold"] = G["pass_top"] > 0.5
G.to_csv(os.path.join(OUT, "groups.csv"), index=False)

T = {"checks": checks}
# decision-class distribution by cell
T["decision_by_cell"] = {c: {k: int(v) for k, v in sub["decision"].value_counts().items()} for c, sub in df.groupby("cell")}
T["decision_by_twin"] = {c: {k: int(v) for k, v in sub["decision"].value_counts().items()} for c, sub in df.groupby("twin")}
T["dclass_by_twin"] = {c: {k: round(v, 4) for k, v in sub["dclass"].value_counts(normalize=True).items()}
                       for c, sub in df.groupby("twin")}
T["sample_pass_by_cell"] = df.groupby("cell")[["pass_top", "pass_strict", "pass_lenient"]].mean().round(4).to_dict("index")
T["sample_pass_by_twin"] = df.groupby("twin")[["pass_top", "pass_strict", "pass_lenient"]].mean().round(4).to_dict("index")

def ptab(by):
    t = G.groupby(by).agg(n_prompts=("row_i", "size"), pass_mean=("pass_top", "mean"), pass_strict=("pass_strict", "mean"),
                          frac_pass0=("pass_top", lambda x: (x == 0).mean()), frac_pass1=("pass_top", lambda x: (x == 1).mean()),
                          frac_mid=("pass_top", lambda x: ((x > 0) & (x < 1)).mean()), majority_gold=("majority_is_gold", "mean"),
                          reward_mean=("reward_mean", "mean"), p_ask=("p_ask", "mean"), p_approve=("p_approve", "mean"),
                          zero_var=("zero_var", "mean"), tokens=("prompt_tokens", "mean")).round(3)
    t.index = ["|".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index]
    return t.to_dict("index")

T["prompt_pass_by_twin"] = ptab(["twin"])
T["prompt_pass_by_cell"] = ptab(["cell"])
T["prompt_pass_by_twin_level"] = ptab(["twin", "level"])
T["prompt_pass_by_twin_ps"] = ptab(["twin", "prompt_setting"])
T["prompt_pass_by_twin_env"] = ptab(["twin", "env"])
T["prompt_pass_by_level"] = ptab(["level"])
T["prompt_pass_by_ps"] = ptab(["prompt_setting"])
T["prompt_pass_by_env"] = ptab(["env"])
T["prompt_pass_by_twin_level_ps"] = ptab(["twin", "level", "prompt_setting"])
full = ptab(["twin", "level", "prompt_setting", "env"])
T["prompt_pass_by_twin_level_ps_env"] = full
pd.DataFrame(full).T.to_csv(os.path.join(OUT, "pass_by_twin_level_ps_env.csv"))

# discrimination: P(ask) on A minus P(ask) on B, per level (a policy that ignores the twin has 0)
disc = {}
for lv, sub in G.groupby("level"):
    a = sub[sub.twin == "A"]["p_ask"].mean()
    b = sub[sub.twin == "B"]["p_ask"].mean()
    disc[lv] = {"p_ask_A": round(a, 3), "p_ask_B": round(b, 3), "gap": round(a - b, 3),
                "n_A": int((sub.twin == "A").sum()), "n_B": int((sub.twin == "B").sum())}
a, b = G[G.twin == "A"]["p_ask"].mean(), G[G.twin == "B"]["p_ask"].mean()
disc["all"] = {"p_ask_A": round(a, 3), "p_ask_B": round(b, 3), "gap": round(a - b, 3)}
T["ask_rate_gap_A_minus_B"] = disc

# pairs where both twins were sampled at the same level: does M1 separate them?
both = G.pivot_table(index=["pair_id", "level"], columns="twin", values=["pass_top", "p_ask"], aggfunc="mean").dropna()
T["pairs_both_twins_same_level"] = {
    "n": int(len(both)),
    "rows": [{"pair_id": p, "level": l, "A_pass": round(r[("pass_top", "A")], 3), "B_pass": round(r[("pass_top", "B")], 3),
              "A_p_ask": round(r[("p_ask", "A")], 3), "B_p_ask": round(r[("p_ask", "B")], 3)}
             for (p, l), r in both.iterrows()]}

# zero-variance and reward spread
T["groups"] = {
    "n": int(len(G)), "zero_var_groups": int(G["zero_var"].sum()), "zero_var_frac": round(G["zero_var"].mean(), 4),
    "zero_var_rows": G[G["zero_var"]][["row_i", "step", "scenario_id", "level", "prompt_setting", "twin", "majority",
                                        "reward_mean"]].to_dict("records"),
    "reward_std1_mean": round(G["reward_std1"].mean(), 4), "reward_std1_median": round(G["reward_std1"].median(), 4),
    "reward_std1_quantiles": {str(q): round(G["reward_std1"].quantile(q), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    "n_classes_dist": {str(k): int(v) for k, v in G["n_classes"].value_counts().sort_index().items()},
    "pass_bin_by_twin": {t: {k: int(v) for k, v in sub["pass_bin"].value_counts().items()} for t, sub in G.groupby("twin")},
    "pass_bin_by_cell": {t: {k: int(v) for k, v in sub["pass_bin"].value_counts().items()} for t, sub in G.groupby("cell")},
    "majority_gold_frac": round(G["majority_is_gold"].mean(), 4),
    "majority_gold_by_twin": G.groupby("twin")["majority_is_gold"].mean().round(4).to_dict(),
    "pass_hist_by_twin": {t: {str(round(k, 3)): int(v) for k, v in sub["pass_top"].value_counts().sort_index().items()}
                          for t, sub in G.groupby("twin")},
}
# per-prompt pass-rate dispersion vs a homogeneous-binomial null (is failure prompt-specific or uniform noise?)
disp = {}
for t, sub in G.groupby("twin"):
    p = sub["pass_top"].to_numpy()
    pbar = p.mean()
    obs_var = p.var(ddof=1)
    binom_var = pbar * (1 - pbar) / 8
    disp[t] = {"mean_pass": round(pbar, 4), "var_between_prompts": round(obs_var, 4), "binomial_var_if_homogeneous": round(binom_var, 4),
               "ratio": round(obs_var / binom_var, 3)}
T["pass_dispersion"] = disp

# ---- consistency with the TRL log (log step s+1 is the update made from the samples generated at step s) ----
lh = [e for e in json.load(open(RESULT))["log_history"] if "reward" in e]
cons = []
for e in lh:
    s = e["step"] - 1
    sub = df[df.step == s]
    gs = G[G.step == s]
    cons.append({"log_step": e["step"], "logged_reward": e["reward"], "scores_reward": sub["score"].mean(),
                 "logged_reward_std": e["reward_std"], "scores_group_std_mean": gs["reward_std1"].mean(),
                 "scores_batch_std": sub["score"].std(ddof=1), "logged_zero_std": e["frac_reward_zero_std"],
                 "scores_zero_std": gs["zero_var"].mean(), "logged_mean_len": e["completions/mean_length"],
                 "scores_mean_len": sub["n_tokens"].mean()})
C = pd.DataFrame(cons)
C.to_csv(os.path.join(OUT, "log_consistency.csv"), index=False)
T["log_consistency"] = {
    "max_abs_reward_diff": float((C.logged_reward - C.scores_reward).abs().max()),
    "max_abs_len_diff": float((C.logged_mean_len - C.scores_mean_len).abs().max()),
    "max_abs_zero_std_diff": float((C.logged_zero_std - C.scores_zero_std).abs().max()),
    "max_abs_logged_std_vs_batch_std": float((C.logged_reward_std - C.scores_batch_std).abs().max()),
    "max_abs_logged_std_vs_group_std_mean": float((C.logged_reward_std - C.scores_group_std_mean).abs().max()),
}
json.dump(T, open(os.path.join(OUT, "tables_01.json"), "w"), indent=1, default=str)
print(json.dumps({k: T[k] for k in ("checks", "decision_by_cell", "sample_pass_by_cell", "log_consistency")}, indent=1, default=str))
print(json.dumps(T["prompt_pass_by_twin_level"], indent=0))
print(json.dumps(T["prompt_pass_by_twin_ps"], indent=0))
print(json.dumps(T["prompt_pass_by_twin_env"], indent=0))
print(json.dumps(T["groups"], indent=0, default=str)[:3000])
print(json.dumps(T["pass_dispersion"], indent=0))
print(json.dumps(T["ask_rate_gap_A_minus_B"], indent=0))
print(json.dumps(T["pairs_both_twins_same_level"], indent=0))
print("OK_01")
