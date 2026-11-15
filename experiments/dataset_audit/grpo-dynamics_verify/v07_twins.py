#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 7: twin structure of the GRPO pool (GD-3 feasibility, GD-8 twin-difference claims) and a
NEW twin-confusion test.

(a) (pair, level) cells: how many hold both twins; how many with the same prompt setting; token-length difference of twins
    (token_stats); whether sampled twins ever shared a step.
(b) where same-setting twin prompts differ: per message (system / user request / assistant prefix / tool result), size of
    the differing span (common prefix/suffix trimmed), and the position of the first differing character measured in
    characters from the END of the prompt (a proxy for 'inside the last 4,096 tokens').
(c) NEW twin confusion: is M1's ask rate driven by the PAIR (shared content) rather than the twin? Pair-level correlation
    of P(ask) on the A prompt vs the B prompt of the same pair (sampled prompts, any level), with a permutation null;
    plus how often B 'ask_informed' could only come from the A twin's discrepancy terms (informed_any terms absent from
    the B prompt text).
Outputs: results/v07_twins.json, results/v07_twin_diffs.csv
"""
import collections
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl"))]
tok = {t["i"]: t for t in (json.loads(l) for l in open(os.path.join(ROOT, "experiments", "dataset_audit", "results",
                                                                        "token_stats.jsonl"))) if t["file"] == "grpo_train.jsonl"}
G = pd.read_csv(os.path.join(OUT, "v01_groups.csv"))
S = pd.read_csv(os.path.join(OUT, "v01_samples.csv"))
R = {}
cells = collections.defaultdict(dict)
for i, r in enumerate(grpo):
    cells[(r["pair_id"], r["level"])][r["twin"][0]] = i
R["pair_level_cells"] = len(cells)
both = {k: v for k, v in cells.items() if "A" in v and "B" in v}
R["cells_with_both_twins"] = len(both)
R["cells_missing_a_twin"] = sorted([f"{k[0]}|{k[1]}|{sorted(v)}" for k, v in cells.items() if not ("A" in v and "B" in v)])
same_ps = {k: v for k, v in both.items() if grpo[v["A"]]["prompt_setting"] == grpo[v["B"]]["prompt_setting"]}
R["cells_both_same_prompt_setting"] = len(same_ps)
R["mean_abs_token_diff_twins"] = round(float(np.mean([abs(tok[v["A"]]["prompt_tokens"] - tok[v["B"]]["prompt_tokens"])
                                                     for v in both.values()])), 2)
R["mean_abs_token_diff_twins_same_ps"] = round(float(np.mean([abs(tok[v["A"]]["prompt_tokens"] - tok[v["B"]]["prompt_tokens"])
                                                             for v in same_ps.values()])), 2)
# sampled twins sharing a step
samp = G.set_index("grpo_i")
pairs_sampled = [(k, v) for k, v in both.items() if v["A"] in samp.index and v["B"] in samp.index]
R["cells_both_twins_sampled"] = len(pairs_sampled)
R["cells_both_twins_sampled_same_step"] = int(sum(samp.loc[v["A"], "step"] == samp.loc[v["B"], "step"] for _, v in pairs_sampled))


def span(a, b):
    """length of the differing middle span of a vs b after trimming the common prefix and suffix; and the prefix length"""
    p = 0
    n = min(len(a), len(b))
    while p < n and a[p] == b[p]:
        p += 1
    s = 0
    while s < n - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return max(len(a) - p - s, len(b) - p - s), p


rows = []
for (pid, lv), v in same_ps.items():
    a, b = grpo[v["A"]], grpo[v["B"]]
    pa, pb = a["prompt"], b["prompt"]
    rec = {"pair_id": pid, "level": lv, "ps": a["prompt_setting"], "n_msgs_A": len(pa), "n_msgs_B": len(pb)}
    if len(pa) != len(pb):
        rec["where"] = "different_message_count"
        rows.append(rec)
        continue
    names = ["system", "user_request"] + [f"m{j}" for j in range(2, len(pa))]
    names = [("assistant_prefix" if m["role"] == "assistant" else ("tool_result" if j >= 2 else names[j]))
             for j, m in enumerate(pa)]
    diffs = {}
    for j, (ma, mb) in enumerate(zip(pa, pb)):
        if ma["content"] != mb["content"]:
            d, _ = span(ma["content"], mb["content"])
            diffs[f"{j}:{names[j]}"] = d
    full_a = "\x00".join(m["content"] for m in pa)
    full_b = "\x00".join(m["content"] for m in pb)
    d_all, pre = span(full_a, full_b)
    rec.update({"where": "+".join(k.split(":")[1] for k in diffs), "diff_chars_total": sum(diffs.values()),
                "diff_span_full": d_all, "first_diff_chars_from_end_A": len(full_a) - pre, "len_chars_A": len(full_a),
                "system_differs": any(k.startswith("0:") for k in diffs), "tokens_A": tok[v["A"]]["prompt_tokens"]})
    rows.append(rec)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "v07_twin_diffs.csv"), index=False)
R["same_ps_cells_identical_prompts"] = int((D.get("diff_chars_total", pd.Series(dtype=float)) == 0).sum())
R["median_diff_chars_total"] = float(D["diff_chars_total"].median())
R["where_counts"] = {k: int(v) for k, v in D["where"].value_counts().items()}
R["system_differs_share"] = round(float(D["system_differs"].mean()), 4)
# chars-from-end -> approx tokens from end using each prompt's chars/token ratio
D["first_diff_tokens_from_end_est"] = D["first_diff_chars_from_end_A"] / (D["len_chars_A"] / D["tokens_A"])
R["first_diff_tokens_from_end_est_quantiles"] = {q: round(float(D["first_diff_tokens_from_end_est"].quantile(q)), 1)
                                                  for q in (0.1, 0.5, 0.9, 1.0)}
R["share_first_diff_beyond_4096_tokens_from_end"] = round(float((D["first_diff_tokens_from_end_est"] > 4096).mean()), 4)
R["smallest_diffs"] = D.nsmallest(3, "diff_chars_total")[["pair_id", "level", "ps", "where", "diff_chars_total"]].to_dict("records")

# (c) twin confusion: pair-level P(ask) on A vs on B (sampled prompts, averaged over levels within a pair)
pa_pair = G[G.twin == "A"].groupby("pair_id")["p_ask"].mean()
pb_pair = G[G.twin == "B"].groupby("pair_id")["p_ask"].mean()
common = sorted(set(pa_pair.index) & set(pb_pair.index))
x, y = pa_pair.loc[common].to_numpy(), pb_pair.loc[common].to_numpy()
r_obs = float(np.corrcoef(x, y)[0, 1])
rng = np.random.default_rng(0)
null = [np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(20000)]
R["pairs_with_both_twins_sampled_any_level"] = len(common)
R["pair_corr_pask_A_vs_B"] = round(r_obs, 4)
R["pair_corr_perm_p_two_sided"] = round(float(np.mean(np.abs(null) >= abs(r_obs))), 4)
R["pairs_A_minus_B_pask_mean"] = round(float(np.mean(x - y)), 4)
R["pairs_A_pask_le_B_pask"] = int((x <= y).sum())
R["pair_examples_A_le_B"] = [{"pair": p, "pask_A": round(float(pa_pair[p]), 3), "pask_B": round(float(pb_pair[p]), 3)}
                             for p in common if pa_pair[p] <= pb_pair[p]][:8]

# B ask_informed: are the informed terms present in the B prompt itself?
def norm(s):
    return re.sub(r"\s+", " ", s.lower())


bi = []
for gi in G[G.twin == "B"]["grpo_i"]:
    r = grpo[gi]
    scn = json.loads(r["scenario"])
    text = norm(" ".join(m["content"] for m in r["prompt"]))
    terms = scn["expected"].get("informed_any", [])
    present = [t for t in terms if norm(t) in text]
    n_inf = int(((S.grpo_i == gi) & (S.decision == "ask_informed")).sum())
    bi.append({"grpo_i": gi, "sid": r["scenario_id"], "n_terms": len(terms), "n_present": len(present), "ask_informed": n_inf})
BI = pd.DataFrame(bi)
R["B_prompts_with_any_informed_term_present"] = f"{int((BI.n_present > 0).sum())}/{len(BI)}"
R["B_ask_informed_on_prompts_with_terms_present"] = int(BI.loc[BI.n_present > 0, "ask_informed"].sum())
R["B_ask_informed_on_prompts_without_terms"] = int(BI.loc[BI.n_present == 0, "ask_informed"].sum())
# are A and B informed_any lists identical within a pair?
inf = collections.defaultdict(dict)
for r in grpo:
    inf[r["pair_id"]][r["twin"][0]] = tuple(json.loads(r["scenario"])["expected"].get("informed_any", []))
R["pairs_informed_any_identical_A_B"] = f"{sum(1 for v in inf.values() if 'A' in v and 'B' in v and v['A'] == v['B'])}/" \
                                        f"{sum(1 for v in inf.values() if 'A' in v and 'B' in v)}"
json.dump(R, open(os.path.join(OUT, "v07_twins.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print("OK_v07")
