"""DPO-pairs audit, part 4: where the A/B difference sits in the prompt, and in what context the rejected sides were generated.

(i)  For every A/B twin at the same level: which prompt message holds the lines that differ (system / member request / look-up
     result; P2 hint removed) and roughly how many tokens before the end of the prompt the first differing line starts (char offset
     x the row's exact tokens-per-char from token_stats.jsonl; CAVEATED approximation). Tiny Aya: 3 of 4 layers see only the last
     4,096 tokens (sliding_window 4096, pattern 4).
(ii) Rejected sides generated without the decisive look-up: the grid episode's earlier calls, the grid prompt setting, and whether
     the A and B twins received the very same rejected text.
Outputs: results/context_checks.json, results/twin_diff_location.csv

    python3 experiments/dataset_audit/dpo-pairs/context_checks.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.render import golden_text  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = os.path.join(ROOT, "dataset", "curated_v121")
HINT = golden_text("P2")
WINDOW = 4096


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


pairs = load(os.path.join(OUT, "pairs.jsonl"))
dpo = load(os.path.join(D, "dpo_train.jsonl"))
by_pl = collections.defaultdict(dict)
for p in pairs:
    by_pl[(p["pair_id"], p["level"])][p["twin"]] = p

ROLE_NAME = {0: "system", 1: "member request"}


def segs(prompt):
    """(message index, role label, line, char offset of the line start in the concatenated prompt text)"""
    out, off = [], 0
    for i, m in enumerate(prompt):
        c = m["content"].replace("\n\n" + HINT, "") if m["role"] == "system" else m["content"]
        label = ROLE_NAME.get(i, "look-up call" if m["role"] == "assistant" else "look-up result")
        pos = 0
        for ln in c.split("\n"):
            if ln.strip():
                out.append((i, label, ln.strip(), off + pos))
            pos += len(ln) + 1
        off += len(c) + 1
    return out, off


rows = []
for (pid, lv), v in sorted(by_pl.items()):
    if "A" not in v or "B" not in v:
        continue
    out = {}
    for tw in ("A", "B"):
        p = v[tw]
        s, total = segs(dpo[p["idx"]]["prompt"])
        other = segs(dpo[v["B" if tw == "A" else "A"]["idx"]]["prompt"])[0]
        oc = collections.Counter(x[2] for x in other)
        diff = [x for x in s if oc[x[2]] == 0]
        tpc = p["prompt_tokens"] / max(1, sum(len(m["content"]) for m in dpo[p["idx"]]["prompt"]))
        first = min((x[3] for x in diff), default=None)
        out[tw] = {"idx": p["idx"], "diff_messages": sorted({x[1] for x in diff}),
                   "first_diff_tokens_from_end": None if first is None else round((total - first) * tpc),
                   "prompt_tokens": p["prompt_tokens"], "n_diff_lines": len(diff)}
    rows.append({"pair_id": pid, "level": lv, "A_idx": out["A"]["idx"], "B_idx": out["B"]["idx"],
                 "A_diff_in": "+".join(out["A"]["diff_messages"]), "B_diff_in": "+".join(out["B"]["diff_messages"]),
                 "A_first_diff_tokens_from_end": out["A"]["first_diff_tokens_from_end"],
                 "A_prompt_tokens": out["A"]["prompt_tokens"], "A_n_diff_lines": out["A"]["n_diff_lines"],
                 "beyond_window": (out["A"]["first_diff_tokens_from_end"] or 0) > WINDOW,
                 "only_in_result": out["A"]["diff_messages"] == ["look-up result"]})
tw = pd.DataFrame(rows)
tw.to_csv(os.path.join(OUT, "twin_diff_location.csv"), index=False)


def vc(s):
    return {str(k): int(v) for k, v in s.value_counts().items()}


# (ii) rejected-generation context
obs = [p for p in pairs if p["rejected_source"] == "observed_failure"]
same_rej = []
for (pid, lv), v in sorted(by_pl.items()):
    if "A" in v and "B" in v and v["A"]["rejected_text"] == v["B"]["rejected_text"]:
        same_rej.append((v["A"], v["B"]))
ctx = {
    "observed_n": len(obs),
    "generated_as_first_action_no_lookup": sum(1 for p in obs if p["grid_n_prior_calls"] == 0),
    "generated_after_exactly_the_canonical_lookups": sum(1 for p in obs if p["grid_prior_calls_match_canonical"]),
    "generated_under_P1_golden_block": sum(1 for p in obs if p["grid_prompt"] == "P1"),
    "generated_by_other_model_than_tiny_aya_global": sum(1 for p in obs if "tiny-aya-global" not in p["model"]),
    "by_level": {lv: {"n": len(g), "first_action": sum(1 for p in g if p["grid_n_prior_calls"] == 0),
                      "P1": sum(1 for p in g if p["grid_prompt"] == "P1")}
                 for lv, g in sorted(collections.defaultdict(list, {lv: [p for p in obs if p["level"] == lv]
                                                                    for lv in ("L0r", "L1", "L2", "L3")}).items())},
    "identical_A_B_rejected_groups": len(same_rej),
    "identical_A_B_rejected_both_first_action": sum(1 for a, b in same_rej if a.get("grid_n_prior_calls") == 0
                                                    and b.get("grid_n_prior_calls") == 0),
    "identical_A_B_rejected_examples": [{"pair_id": a["pair_id"], "level": a["level"], "A_idx": a["idx"], "B_idx": b["idx"],
                                         "A_grid": [a["grid_file"], a["grid_line"]], "B_grid": [b["grid_file"], b["grid_line"]],
                                         "text": a["rejected_text"][:100]} for a, b in same_rej[:4]],
}
res = {"twin_difference_location": {
    "n_twin_levels": len(tw), "diff_in": vc(tw.A_diff_in), "only_in_lookup_result": int(tw.only_in_result.sum()),
    "first_diff_tokens_from_end_quantiles": {str(q): float(tw.A_first_diff_tokens_from_end.quantile(q))
                                             for q in (0, 0.25, 0.5, 0.75, 0.9, 1)},
    "beyond_4096_from_end": int(tw.beyond_window.sum()),
    "beyond_4096_by_level": vc(tw[tw.beyond_window].level),
    "examples_beyond": tw[tw.beyond_window].head(4)[["pair_id", "level", "A_idx", "B_idx", "A_diff_in",
                                                      "A_first_diff_tokens_from_end", "A_prompt_tokens"]].to_dict("records"),
    "examples_result_only": tw[tw.only_in_result].head(3)[["pair_id", "level", "A_idx", "B_idx"]].to_dict("records")},
    "rejected_generation_context": ctx}
json.dump(res, open(os.path.join(OUT, "context_checks.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
print("CONTEXT_CHECKS_OK")
