#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 10: checks for GD-5 (what drives the twin-blind push), GD-1 ('approval cards name the
discrepancy'), GD-6/GD-8 (external difficulty), and things the analyst did not test.

(a) per-group net push N (ask minus approve advantages) by twin under the logged table and alternatives -> decomposition of
    the run's shared push into composition (58 A vs 62 B groups) and per-group asymmetry (mean N_A + mean N_B != 0).
(b) groups whose reward variance comes only from the format / verbosity terms (one decision class, rewards differ):
    group-std scaling turns a 0.1 format gap into full-size advantages.
(c) how generic are the A twins' informed_any terms? share of A prompts where a term already appears in the member request
    or the look-up result (a card that merely repeats the request/result would be graded approve_informed).
(d) external difficulty: per-scenario success of strong models (Claude Opus/Sonnet/Haiku, Qwen3-235B, DeepSeek-V3.2,
    Aya Expanse 32B) in the replication grids; Spearman with M1's per-prompt A pass; hard A prompts early vs late.
(e) base Tiny Aya Global (replication grids, decision mode, its own rendering) ask rates on the sampled scenarios vs M1.
Outputs: results/v10_extras.json
"""
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.spec import any_hit  # noqa: E402

OUT = os.path.join(HERE, "results")
S = pd.read_csv(os.path.join(OUT, "v01_samples.csv"))
G = pd.read_csv(os.path.join(OUT, "v01_groups.csv"))
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl"))]
R = {}


def fam(d):
    return "ask" if d.startswith(("ask", "tell")) else "approve" if d.startswith("approve") else \
        "commit" if d.startswith("commit") else d


S["fam"] = S["decision"].map(fam)
A_T = {"ask_informed": 1.0, "tell_informed": 0.5, "approve_gold": 0.5, "approve_informed": 0.25, "statement": -0.25,
       "ask_generic": -0.5, "approve_generic": -1.0, "commit_gold": -1.0, "commit_harmful": -1.0, "commit_other": -1.0}


def base_alt(row, a_t):
    if row.twin != "A" or row.decision in ("lookup", "invalid"):
        return row.base
    return a_t.get(row.decision, -0.5)


def group_N(col):
    out = []
    for (st, gi), s in S.groupby(["step", "grpo_i"]):
        r = s[col].to_numpy()
        a = (r - r.mean()) / (r.std(ddof=1) + 1e-4)
        f = s["fam"].to_numpy()
        out.append({"twin": s.twin.iloc[0], "N": a[f == "ask"].sum() - a[f == "approve"].sum()})
    return pd.DataFrame(out)


tables = {"logged": None, "no_card_credit_on_A": {**A_T, "approve_gold": -1.0, "approve_informed": -1.0},
          "A_lookup_as_wrong": None}
dec = {}
for name, at in tables.items():
    if name == "logged":
        S["r"] = S["score"]
    elif name == "A_lookup_as_wrong":
        S["r"] = np.where((S.twin == "A") & (S.decision == "lookup"), -1.0 + S["fmt"], S["score"])
    else:
        S["r"] = [base_alt(r, at) for r in S.itertuples()] + S["fmt"] + S["verb"]
    P = group_N("r")
    mA, mB = P[P.twin == "A"]["N"].mean(), P[P.twin == "B"]["N"].mean()
    nA, nB = int((P.twin == "A").sum()), int((P.twin == "B").sum())
    total = nA * mA + nB * mB
    balanced = (nA + nB) / 2 * (mA + mB)
    dec[name] = {"mean_N_A_group": round(float(mA), 3), "mean_N_B_group": round(float(mB), 3), "shared_total": round(float(total), 3),
                 "asymmetry_part_(balanced_counts)": round(float(balanced), 3),
                 "composition_part": round(float(total - balanced), 3)}
R["shared_push_decomposition"] = dec

# (b) format-only variance groups
fo = []
for (st, gi), s in S.groupby(["step", "grpo_i"]):
    if s["decision"].nunique() == 1 and s["score"].std(ddof=1) > 1e-9:
        fo.append({"step": int(st), "grpo_i": int(gi), "class": s["decision"].iloc[0],
                   "scores": sorted(s["score"].round(3).tolist())})
R["groups_variance_only_from_format_or_verbosity"] = fo
R["groups_single_class"] = int(sum(1 for _, s in S.groupby(["step", "grpo_i"]) if s["decision"].nunique() == 1))

# (c) informed_any genericity on A prompts
rows = []
for i, r in enumerate(grpo):
    if not r["twin"].startswith("A"):
        continue
    scn = json.loads(r["scenario"])
    terms = scn["expected"].get("informed_any", [])
    turns = " ".join(m["content"] for m in r["prompt"][1:])
    req = r["prompt"][1]["content"]
    hit_turns = [t for t in terms if any_hit(turns, [t])]
    rows.append({"grpo_i": i, "sid": r["scenario_id"], "n_terms": len(terms), "term_in_request": any_hit(req, terms),
                 "term_in_turns": bool(hit_turns), "example_terms": hit_turns[:3]})
C = pd.DataFrame(rows)
R["A_rows"] = int(len(C))
R["A_rows_with_informed_term_in_request"] = int(C.term_in_request.sum())
R["A_rows_with_informed_term_in_request_or_lookup_result"] = int(C.term_in_turns.sum())
samp_A = set(G[G.twin == "A"]["grpo_i"])
CS = C[C.grpo_i.isin(samp_A)]
R["sampled_A_prompts_with_term_in_turns"] = f"{int(CS.term_in_turns.sum())}/{len(CS)}"
ai = S[(S.twin == "A") & (S.decision == "approve_informed")].groupby("grpo_i").size()
R["A_approve_informed_samples_on_prompts_with_term_in_turns"] = int(ai[ai.index.isin(CS[CS.term_in_turns].grpo_i)].sum())
R["A_approve_informed_samples_total"] = int(ai.sum())
R["A_term_in_turns_examples"] = C[C.term_in_turns].head(6)[["grpo_i", "sid", "example_terms"]].to_dict("records")

# (d) external difficulty
STRONG = ("claude:", "Qwen3-235B", "DeepSeek-V3.2", "aya-expanse-32b")
recs = []
for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "replication", "results", "grid_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r.get("mode") != "decision" or "decision_grade" not in r:
            continue
        m = r["model"]
        grp = "strong" if any(s in m for s in STRONG) else ("base_tiny_aya_global" if "tiny-aya-global" in m else "other")
        d = r["decision_grade"]["decision"]
        recs.append({"sid": r["id"], "level": r["level"], "prompt": r["prompt"], "grp": grp, "twin": r["twin"][0],
                     "success": float(bool(r["decision_grade"]["success"])),
                     "ask": float(str(d).startswith(("ask", "tell")))})
E = pd.DataFrame(recs)
ext = E[E.grp == "strong"].groupby("sid")["success"].mean()
G["ext_strong"] = G["scenario_id"].map(ext)
GA = G[G.twin == "A"].dropna(subset=["ext_strong"])
rk = lambda s: s.rank().to_numpy()  # noqa: E731
R["n_A_prompts_with_external"] = int(len(GA))
R["spearman_M1_A_pass_vs_strong_success"] = round(float(np.corrcoef(rk(GA.pass_gold), rk(GA.ext_strong))[0, 1]), 4)
GB = G[G.twin == "B"].dropna(subset=["ext_strong"])
R["spearman_M1_B_pass_vs_strong_success"] = round(float(np.corrcoef(rk(GB.pass_gold), rk(GB.ext_strong))[0, 1]), 4)
GA = GA.assign(late=(GA.step >= 15))
R["hard_A_prompts_(strong<=0.15)_early_late"] = [int(((GA.ext_strong <= 0.15) & ~GA.late).sum()),
                                                  int(((GA.ext_strong <= 0.15) & GA.late).sum())]
# OLS: A pass ~ late + ext + zero_prefix (numpy)
X = np.column_stack([np.ones(len(GA)), GA.late.astype(float), GA.ext_strong, GA.zero_prefix.astype(float)])
b, *_ = np.linalg.lstsq(X, GA.pass_gold.to_numpy(), rcond=None)
rng = np.random.default_rng(0)
bs = []
for _ in range(4000):
    idx = rng.integers(0, len(GA), len(GA))
    bb, *_ = np.linalg.lstsq(X[idx], GA.pass_gold.to_numpy()[idx], rcond=None)
    bs.append(bb[1])
R["A_pass_ols_late_coef_ctrl_ext_zeroprefix"] = {"coef": round(float(b[1]), 4),
                                                 "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)]}

# (e) base Tiny Aya Global on the sampled scenarios (its own grid renderings), vs M1
sids = set(G.scenario_id)
BA = E[(E.grp == "base_tiny_aya_global") & (E.sid.isin(sids))]
R["base_tiny_aya_records_on_sampled_scenarios"] = int(len(BA))
if len(BA):
    pa, pb = BA[BA.twin == "A"]["ask"].mean(), BA[BA.twin == "B"]["ask"].mean()
    R["base_tiny_aya_P_ask_A"], R["base_tiny_aya_P_ask_B"] = round(float(pa), 4), round(float(pb), 4)
    R["base_tiny_aya_success_A"] = round(float(BA[BA.twin == "A"]["success"].mean()), 4)
    R["base_tiny_aya_success_B"] = round(float(BA[BA.twin == "B"]["success"].mean()), 4)
    R["base_tiny_aya_by_level_P_ask"] = {f"{t}|{lv}": round(float(v), 3) for (t, lv), v in
                                         BA.groupby(["twin", "level"])["ask"].mean().items()}
ST = E[(E.grp == "strong") & (E.sid.isin(sids))]
R["strong_P_ask_A"] = round(float(ST[ST.twin == "A"]["ask"].mean()), 4)
R["strong_P_ask_B"] = round(float(ST[ST.twin == "B"]["ask"].mean()), 4)
R["M1_P_ask_A"] = round(float(S[S.twin == "A"]["fam"].eq("ask").mean()), 4)
R["M1_P_ask_B"] = round(float(S[S.twin == "B"]["fam"].eq("ask").mean()), 4)
json.dump(R, open(os.path.join(OUT, "v10_extras.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
print("OK_v10")
