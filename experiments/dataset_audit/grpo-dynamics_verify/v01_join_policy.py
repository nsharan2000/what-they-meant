#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 1 (independent re-derivation; does not read the analyst's outputs).

Checks GD-1's numbers from the raw files:
  * are the 496 GRPO prompts byte-identical to sft_train decision-row prompts?
  * M1/M2 decision-class counts per twin / cell from scores.jsonl (joined to grpo_train by scenario_id+level)
  * gold-class (reward base == +1) rates, ask rates, d', per-prompt pass distribution, majority-gold share
  * A approval cards that name the discrepancy
  * NEW: what does the GRPO reward (wtm.reward) say about the SFT TARGETS themselves? (reward/SFT consistency)
  * NEW: first character of SFT decision targets per cell (is the class fixed by the first token?)
Inputs (read-only): dataset/curated_v121/{grpo_train,sft_train,sft_val}.jsonl,
  experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl
Outputs: results/v01_policy.json, results/v01_samples.csv, results/v01_groups.csv
"""
import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.reward import reward as wtm_reward  # noqa: E402
from wtm.episode import decision_grade  # noqa: E402
from wtm.labels import adjudicate  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
D = os.path.join(ROOT, "dataset", "curated_v121")
L = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo")


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


grpo = jl(os.path.join(D, "grpo_train.jsonl"))
sft = jl(os.path.join(D, "sft_train.jsonl"))
val = jl(os.path.join(D, "sft_val.jsonl"))
scores = jl(os.path.join(L, "scores.jsonl"))
R = {}

# ---------------------------------------------------------------- 1. identity of GRPO prompts with SFT decision rows
canon = lambda msgs: json.dumps(msgs, ensure_ascii=False, sort_keys=True)  # noqa: E731
sft_by_prompt = collections.defaultdict(list)
for i, r in enumerate(sft):
    sft_by_prompt[canon(r["prompt"])].append((i, r.get("turn"), r["scenario_id"], r["level"], r["prompt_setting"]))
val_prompts = {canon(r["prompt"]) for r in val}
ident = collections.Counter()
grpo_to_sft = {}
for gi, r in enumerate(grpo):
    hits = sft_by_prompt.get(canon(r["prompt"]), [])
    dec = [h for h in hits if h[1] == "decision" and h[2] == r["scenario_id"] and h[3] == r["level"]]
    ident["identical_to_sft_decision_same_key"] += bool(dec)
    ident["identical_to_any_sft_row"] += bool(hits)
    ident["identical_to_sft_nondecision_row"] += any(h[1] != "decision" for h in hits)
    ident["identical_to_sft_val_row"] += canon(r["prompt"]) in val_prompts
    ident["multiple_sft_hits"] += len(hits) > 1
    if dec:
        grpo_to_sft[gi] = dec[0][0]
R["grpo_rows"] = len(grpo)
R["prompt_identity"] = dict(ident)

# ---------------------------------------------------------------- 2. what the GRPO reward says about the SFT targets
tgt_rows = []
for gi, si in grpo_to_sft.items():
    g = grpo[gi]
    scn = json.loads(g["scenario"])
    kinds = json.loads(g["tool_kinds"])
    tgt = sft[si]["completion"][-1]["content"]
    r, parts = wtm_reward(tgt, scn, kinds)
    exp = scn["expected"]["decision"]
    cell = "A" if g["twin"].startswith("A") else ("B_act" if exp == "act" else "B_approve")
    tgt_rows.append({"grpo_i": gi, "sft_i": si, "scenario_id": g["scenario_id"], "level": g["level"], "cell": cell,
                     "tgt_decision": parts["decision"], "tgt_reward": r, "tgt_base": parts["base"],
                     "tgt_first_char": tgt.lstrip()[:1], "tgt_is_tool": tgt.lstrip().startswith("<tool_call>")})
T = pd.DataFrame(tgt_rows)
R["sft_target_graded_by_reward"] = {c: {k: int(v) for k, v in s["tgt_decision"].value_counts().items()}
                                    for c, s in T.groupby("cell")}
R["sft_target_base_not_top"] = int((T["tgt_base"] < 1.0).sum())
R["sft_target_base_not_top_by_cell"] = {c: int((s["tgt_base"] < 1.0).sum()) for c, s in T.groupby("cell")}
R["sft_target_starts_with_tool_call_by_cell"] = {c: {"tool": int(s["tgt_is_tool"].sum()), "n": int(len(s))}
                                                 for c, s in T.groupby("cell")}
R["sft_target_not_top_examples"] = T[T["tgt_base"] < 1.0].head(12)[
    ["grpo_i", "sft_i", "scenario_id", "level", "cell", "tgt_decision", "tgt_base"]].to_dict("records")
T.to_csv(os.path.join(OUT, "v01_sft_targets_graded.csv"), index=False)

# ---------------------------------------------------------------- 3. join scores -> grpo rows
key = {}
for gi, r in enumerate(grpo):
    k = (r["scenario_id"], r["level"])
    assert k not in key, k
    key[k] = gi
recs = []
for n, s in enumerate(scores):
    m = s["meta"]
    gi = key[(m["scenario_id"], m["level"])]
    g = grpo[gi]
    scn = json.loads(g["scenario"])
    assert scn["twin"] == m["twin"]
    exp = scn["expected"]["decision"]
    cell = "A" if g["twin"].startswith("A") else ("B_act" if exp == "act" else "B_approve")
    d = s["components"]["decision"]
    # re-score is impossible without the completion text; check the logged score decomposes into base+format+verbosity
    comp = s["components"]
    dg = decision_grade(adjudicate(scn), d) if d not in ("lookup", "invalid", "truncated") else None
    recs.append({"n": n, "step": m["step"], "grpo_i": gi, "scenario_id": m["scenario_id"], "pair_id": g["pair_id"],
                 "level": m["level"], "ps": g["prompt_setting"], "env": g["env_id"], "cell": cell, "twin": g["twin"][0],
                 "decision": d, "score": s["score"], "base": comp["base"], "fmt": comp.get("format", 0.0),
                 "verb": comp.get("verbosity", 0.0), "n_tokens": m["n_tokens"], "end_id": m["end_id"],
                 "strict": None if dg is None else bool(dg["success"]),
                 "lenient": None if dg is None else bool(dg["success_lenient"]),
                 "zero_prefix": len(g["prompt"]) == 2})
S = pd.DataFrame(recs)
S["gold"] = S["base"] >= 1.0 - 1e-9
S["is_ask"] = S["decision"].str.startswith("ask")
S["is_ask_or_tell"] = S["decision"].str.startswith(("ask", "tell"))
S["is_approve"] = S["decision"].str.startswith("approve")
S["score_decomp_ok"] = (S["score"] - (S["base"] + S["fmt"] + S["verb"])).abs() < 1e-6
S.to_csv(os.path.join(OUT, "v01_samples.csv"), index=False)
R["n_samples"] = int(len(S))
R["score_decomposition_ok"] = int(S["score_decomp_ok"].sum())
R["steps"] = sorted(S["step"].unique().tolist())[:3] + ["..."] + sorted(S["step"].unique().tolist())[-2:]
per_step = S.groupby("step").agg(n=("n", "size"), prompts=("grpo_i", "nunique"))
R["all_steps_32_samples_4_prompts"] = bool(((per_step["n"] == 32) & (per_step["prompts"] == 4)).all())
R["distinct_prompts"] = int(S["grpo_i"].nunique())
R["prompts_in_more_than_one_step"] = int((S.groupby("grpo_i")["step"].nunique() > 1).sum())
R["class_counts_by_twin"] = {t: {k: int(v) for k, v in s["decision"].value_counts().items()} for t, s in S.groupby("twin")}
R["class_counts_by_cell"] = {t: {k: int(v) for k, v in s["decision"].value_counts().items()} for t, s in S.groupby("cell")}
R["n_by_cell"] = {c: int(len(s)) for c, s in S.groupby("cell")}
R["gold_rate_by_cell"] = {c: round(float(s["gold"].mean()), 4) for c, s in S.groupby("cell")}
R["strict_rate_by_cell"] = {c: round(float(s["strict"].fillna(False).astype(bool).mean()), 4) for c, s in S.groupby("cell")}
R["lenient_rate_by_cell"] = {c: round(float(s["lenient"].fillna(False).astype(bool).mean()), 4) for c, s in S.groupby("cell")}


def ndtri(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


for col in ("is_ask", "is_ask_or_tell"):
    pa = float(S.loc[S.twin == "A", col].mean())
    pb = float(S.loc[S.twin == "B", col].mean())
    R[f"P({col}|A)"] = round(pa, 4)
    R[f"P({col}|B)"] = round(pb, 4)
    R[f"dprime_{col}"] = round(ndtri(pa) - ndtri(pb), 4)
pa = float(S.loc[S.twin == "A", "is_approve"].mean())
pb = float(S.loc[S.twin == "B", "is_approve"].mean())
R["P(approve|A)"], R["P(approve|B)"] = round(pa, 4), round(pb, 4)
R["P(approve_or_commit|B)"] = round(float((S.loc[S.twin == "B", "is_approve"] | S.loc[S.twin == "B", "decision"].str.startswith("commit")).mean()), 4)

# A approval cards naming the discrepancy
A = S[S.twin == "A"]
R["A_approval_cards"] = int(A["is_approve"].sum())
R["A_approval_cards_informed_or_gold"] = int(A["decision"].isin(["approve_informed", "approve_gold"]).sum())

# ---------------------------------------------------------------- 4. per prompt (group) table
G = S.groupby("grpo_i").agg(step=("step", "first"), scenario_id=("scenario_id", "first"), pair_id=("pair_id", "first"),
                             level=("level", "first"), ps=("ps", "first"), env=("env", "first"), cell=("cell", "first"),
                             twin=("twin", "first"), pass_gold=("gold", "mean"), p_ask=("is_ask_or_tell", "mean"),
                             p_approve=("is_approve", "mean"), r_mean=("score", "mean"),
                             r_std1=("score", lambda x: float(np.std(x, ddof=1))), zero_prefix=("zero_prefix", "first"),
                             classes=("decision", lambda x: json.dumps(collections.Counter(x).most_common())))
G = G.reset_index()
G.to_csv(os.path.join(OUT, "v01_groups.csv"), index=False)
R["groups"] = int(len(G))
R["groups_0_lt_pass_lt_1"] = int(((G.pass_gold > 0) & (G.pass_gold < 1)).sum())
R["groups_pass_0"] = int((G.pass_gold == 0).sum())
R["groups_pass_1"] = int((G.pass_gold == 1).sum())
R["groups_pass_gt_half"] = round(float((G.pass_gold > 0.5).mean()), 4)
R["groups_pass_ge_half"] = round(float((G.pass_gold >= 0.5).mean()), 4)
# plurality class == gold class (a different greedy proxy)
def plural_gold(gi):
    s = S[S.grpo_i == gi]
    c = s["decision"].value_counts()
    top = c[c == c.max()].index.tolist()
    gold = set(s.loc[s.gold, "decision"])
    return any(t in gold for t in top) and len(top) == 1
R["groups_plurality_class_is_gold_unique"] = round(float(np.mean([plural_gold(g) for g in G.grpo_i])), 4)
R["groups_zero_reward_variance"] = int((G.r_std1 < 1e-9).sum())
R["mean_within_group_std_ddof1"] = round(float(G.r_std1.mean()), 4)
R["groups_by_cell"] = {c: int(len(s)) for c, s in G.groupby("cell")}
R["pass_gold_by_cell_prompt_mean"] = {c: round(float(s.pass_gold.mean()), 4) for c, s in G.groupby("cell")}
# implied NLL of the gold class (same smoothing as the analyst, and two alternatives)
for lab, lo in (("clip_1_16", 1 / 16), ("clip_1_32", 1 / 32)):
    p = G.pass_gold.clip(lo, 1 - lo)
    nll = -np.log(p)
    R[f"implied_gold_nll_{lab}"] = {"mean": round(float(nll.mean()), 4), "median": round(float(nll.median()), 4)}
# ask/approve decision only (class family), ignoring informed/generic: NLL of the correct FAMILY
fam_ok = np.where(G.twin == "A", G.p_ask, np.where(G.cell == "B_act", 1 - G.p_ask - 0.0, G.p_approve))
G["p_family"] = np.where(G.twin == "A", G.p_ask, S.groupby("grpo_i").apply(
    lambda s: float((s.is_approve | s.decision.str.startswith("commit")).mean()), include_groups=False).reindex(G.grpo_i).values)
nllf = -np.log(G.p_family.clip(1 / 16, 1 - 1 / 16))
R["implied_family_nll_clip_1_16"] = {"mean": round(float(nllf.mean()), 4), "median": round(float(nllf.median()), 4),
                                     "note": "A: P(ask or tell); B: P(approve or commit) -- the first-token family"}
R["examples"] = {}
for gi in (204, 137, 232, 262, 372, 435, 487, 468, 1, 494, 279, 294):
    if gi in set(G.grpo_i):
        g = G[G.grpo_i == gi].iloc[0]
        R["examples"][str(gi)] = {"sid": g.scenario_id, "level": g.level, "ps": g.ps, "step": int(g.step),
                                  "classes": json.loads(g.classes), "zero_prefix": bool(g.zero_prefix)}
    else:
        R["examples"][str(gi)] = "not sampled"
json.dump(R, open(os.path.join(OUT, "v01_policy.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
print("OK_v01")
