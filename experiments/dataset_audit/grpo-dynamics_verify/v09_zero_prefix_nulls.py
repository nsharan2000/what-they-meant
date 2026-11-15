#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 9: GD-4 (zero-prefix decision prompts) and GD-9 (empty/null raw tool fields, needless asks).

GD-4: count GRPO rows whose prompt is [system, user] only; in sft_train, count [system, user]-shaped prompts by turn; the
sampled zero-prefix prompts' classes; lookups per prompt over all 960 samples; what the EVALUATION does with a look-up at a
decision point (wtm/episode.py run_episode: look-ups before the first decision are free) vs the reward (-0.1).
GD-9: per GRPO row, parse every <tool_result> JSON in the prompt and count null / empty-string / empty-list / empty-dict
leaves; share of rows with >= 1 such leaf by twin; M1's B ask rate (ask_* or tell_*) by empty-leaf count; Spearman (ranks
via pandas). Also the twin-difference size with difflib (the analyst's metric) for the record.
Outputs: results/v09_zero_prefix_nulls.json
"""
import collections
import difflib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.spec import any_hit  # noqa: E402

OUT = os.path.join(HERE, "results")
D = os.path.join(ROOT, "dataset", "curated_v121")
grpo = [json.loads(l) for l in open(os.path.join(D, "grpo_train.jsonl"))]
sft = [json.loads(l) for l in open(os.path.join(D, "sft_train.jsonl"))]
S = pd.read_csv(os.path.join(OUT, "v01_samples.csv"))
G = pd.read_csv(os.path.join(OUT, "v01_groups.csv"))
R = {}

# ---------------- GD-4
zp = [i for i, r in enumerate(grpo) if len(r["prompt"]) == 2]
R["grpo_zero_prefix_rows"] = len(zp)
R["grpo_zero_prefix_scenarios"] = dict(collections.Counter(grpo[i]["scenario_id"] for i in zp))
R["sft_two_message_prompts_by_turn"] = dict(collections.Counter(r["turn"] for r in sft if len(r["prompt"]) == 2))
R["sft_prompt_len_by_turn"] = {f"{k[0]}|{k[1]}": v for k, v in
                               collections.Counter((r["turn"], len(r["prompt"])) for r in sft).items()}
lk = S[S.decision == "lookup"].groupby("grpo_i").size().sort_values(ascending=False)
R["lookups_total"] = int(lk.sum())
R["lookups_by_twin"] = S[S.decision == "lookup"].groupby("twin").size().to_dict()
R["lookup_prompts"] = {str(int(k)): int(v) for k, v in lk.items()}
R["lookups_in_top5_prompts"] = int(lk.head(5).sum())
zs = S[S.grpo_i.isin(zp)]
R["sampled_zero_prefix_prompts"] = sorted(zs.grpo_i.unique().tolist())
R["sampled_zero_prefix_classes"] = zs["decision"].value_counts().to_dict()
R["sampled_zero_prefix_by_twin"] = {f"{k[0]}|{k[1]}": int(v) for k, v in zs.groupby("twin")["decision"].value_counts().items()}
R["lookup_rows_top"] = [{"grpo_i": int(k), "sid": grpo[int(k)]["scenario_id"], "level": grpo[int(k)]["level"],
                         "zero_prefix": int(k) in zp, "n_lookup": int(v)} for k, v in lk.head(8).items()]
# what is in the zero-prefix SFT decision targets
R["zero_prefix_sft_targets_first60"] = sorted({r["completion"][0]["content"][:60] for r in sft
                                               if len(r["prompt"]) == 2 and r["turn"] == "decision"})[:6]

# ---------------- GD-9
TR = re.compile(r"<tool_result name=\"[^\"]*\">\n?(.*?)\n?</tool_result>", re.S)


def leaves(x):
    if isinstance(x, dict):
        if not x:
            yield "empty"
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, list):
        if not x:
            yield "empty"
        for v in x:
            yield from leaves(v)
    elif x is None:
        yield "null"
    elif isinstance(x, str) and not x.strip():
        yield "empty"
    else:
        yield "value"


rows = []
for i, r in enumerate(grpo):
    n_empty = n_leaf = n_unparsed = 0
    for m in r["prompt"][2:]:
        if m["role"] != "user":
            continue
        for body in TR.findall(m["content"]):
            try:
                obj = json.loads(body)
            except ValueError:
                n_unparsed += 1
                continue
            for lf in leaves(obj):
                n_leaf += 1
                n_empty += lf != "value"
    rows.append({"grpo_i": i, "twin": r["twin"][0], "n_empty": n_empty, "n_leaf": n_leaf, "n_unparsed": n_unparsed,
                 "zero_prefix": len(r["prompt"]) == 2})
N = pd.DataFrame(rows)
R["rows_with_unparsed_tool_results"] = int((N.n_unparsed > 0).sum())
nz = N[~N.zero_prefix]
R["share_rows_with_empty_or_null_leaf"] = {t: round(float((s.n_empty > 0).mean()), 4) for t, s in nz.groupby("twin")}
GB = G[G.twin == "B"].merge(N[["grpo_i", "n_empty", "n_leaf"]], on="grpo_i")
R["B_pask_by_empty_ge2"] = {"ge2": round(float(GB.loc[GB.n_empty >= 2, "p_ask"].mean()), 4),
                            "lt2": round(float(GB.loc[GB.n_empty < 2, "p_ask"].mean()), 4),
                            "n_ge2": int((GB.n_empty >= 2).sum()), "n_lt2": int((GB.n_empty < 2).sum())}
rk = lambda s: s.rank().to_numpy()  # noqa: E731
R["B_spearman_pask_vs_n_empty"] = round(float(np.corrcoef(rk(GB["p_ask"]), rk(GB["n_empty"]))[0, 1]), 4)
R["B_spearman_pask_vs_empty_share"] = round(float(np.corrcoef(rk(GB["p_ask"]), rk(GB["n_empty"] / GB["n_leaf"].clip(lower=1)))[0, 1]), 4)
rng = np.random.default_rng(0)
x, y = rk(GB["p_ask"]), rk(GB["n_empty"])
null = [np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(20000)]
R["B_spearman_perm_p"] = round(float(np.mean(np.abs(null) >= abs(np.corrcoef(x, y)[0, 1]))), 4)
GA = G[G.twin == "A"].merge(N[["grpo_i", "n_empty"]], on="grpo_i")
R["A_spearman_pask_vs_n_empty"] = round(float(np.corrcoef(rk(GA["p_ask"]), rk(GA["n_empty"]))[0, 1]), 4)
# B prompts with low pass
low = G[(G.twin == "B")].sort_values("pass_gold").head(6)
R["B_lowest_pass"] = [{"grpo_i": int(r.grpo_i), "sid": r.scenario_id, "level": r.level, "pass": r.pass_gold,
                       "p_ask": r.p_ask, "classes": json.loads(r.classes),
                       "n_empty": int(N.loc[N.grpo_i == r.grpo_i, "n_empty"].iloc[0])} for r in low.itertuples()]
R["B_approve_prompts_pass0"] = int(((G.cell == "B_approve") & (G.pass_gold == 0)).sum())
R["B_approve_prompts_pass_1of8"] = int(((G.cell == "B_approve") & (np.isclose(G.pass_gold, 0.125))).sum())

# B ask_informed when no informed term is in the B prompt (any_hit = the grader's own matcher)
bi = []
for gi in G[G.twin == "B"]["grpo_i"]:
    r = grpo[gi]
    scn = json.loads(r["scenario"])
    text = " ".join(m["content"] for m in r["prompt"][1:])          # member request + look-ups (not the system prompt)
    full = " ".join(m["content"] for m in r["prompt"])
    terms = scn["expected"].get("informed_any", [])
    bi.append({"grpo_i": gi, "sid": r["scenario_id"], "level": r["level"],
               "term_in_turns": any_hit(text, terms), "term_in_full": any_hit(full, terms),
               "ask_informed": int(((S.grpo_i == gi) & (S.decision.isin(["ask_informed", "tell_informed"]))).sum())})
BI = pd.DataFrame(bi)
R["B_informed_asks_total"] = int(BI.ask_informed.sum())
R["B_informed_asks_on_prompts_without_any_term_anywhere"] = int(BI.loc[~BI.term_in_full, "ask_informed"].sum())
R["B_prompts_without_any_term_anywhere"] = f"{int((~BI.term_in_full).sum())}/{len(BI)}"
R["B_informed_asks_without_term_examples"] = BI.loc[(~BI.term_in_full) & (BI.ask_informed > 0)].head(6).to_dict("records")

# twin difference by difflib (changed characters), same-setting cells
cells = collections.defaultdict(dict)
for i, r in enumerate(grpo):
    cells[(r["pair_id"], r["level"], r["prompt_setting"])][r["twin"][0]] = i
dc = []
for k, v in cells.items():
    if "A" in v and "B" in v:
        a, b = grpo[v["A"]]["prompt"], grpo[v["B"]]["prompt"]
        n = 0
        for ma, mb in zip(a, b):
            if ma["content"] != mb["content"]:
                sm = difflib.SequenceMatcher(None, ma["content"], mb["content"], autojunk=False)
                n += sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        dc.append(n)
R["twin_diff_difflib_chars"] = {"n": len(dc), "median": float(np.median(dc)), "min": int(min(dc)), "max": int(max(dc))}
json.dump(R, open(os.path.join(OUT, "v09_zero_prefix_nulls.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
print("OK_v09")
