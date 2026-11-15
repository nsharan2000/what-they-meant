"""checked_ok 'M1's wrong-class outputs are mostly fresh, not verbatim twin targets': re-derived from sample lengths.

GRPO meta.n_tokens = generated ids incl. the end token(s). The offset to the token_stats target length is estimated from samples
whose class equals the row's own gold class (mode of n_tokens - own target tokens). A flipped sample 'length-matches' the twin
target when n_tokens - offset == twin target tokens (same pair, same level when available, else any level).
Chance rate: the same test against a random other scenario's target of the same kind (message vs tool call).
Output: results/v9_regurg.json
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

from vcommon import ROOT, cur, dump, load_jsonl, is_call, target

rng = np.random.default_rng(7)
S = load_jsonl(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))
ts = [t for t in load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")) if t["file"] == "sft_train.jsonl"]
tr = cur("sft_train")
tok = {}            # (scenario_id, level) -> target tokens of the decision row
kind = {}
for r, t in zip(tr, ts):
    if r["turn"] == "decision":
        tok[(r["scenario_id"], r["level"])] = t["completion_tokens"]
        kind[r["scenario_id"]] = "call" if is_call(target(r)) else "msg"
GOLD_OK = {"A": {"ask_informed"}, "B": {"approve_generic", "approve_informed", "commit_gold"}}
TOOLC = {"approve_generic", "approve_informed", "approve_gold", "commit_gold", "commit_other", "commit_harmful", "lookup"}
off = Counter()
for s in S:
    m = s["meta"]
    k = (m["scenario_id"], m["level"])
    if k in tok and s["components"]["decision"] in GOLD_OK[m["twin"][0]]:
        off[m["n_tokens"] - tok[k]] += 1
offset = off.most_common(1)[0][0]
own_match = sum(v for d, v in off.items() if d == offset) / sum(off.values())
flip, match, chance = 0, 0, []
by_kind = defaultdict(list)
for s in S:
    m = s["meta"]
    sid, lv, tw = m["scenario_id"], m["level"], m["twin"][0]
    d = s["components"]["decision"]
    flipped = (tw == "A" and d in TOOLC) or (tw == "B" and d not in TOOLC and d != "invalid")
    if not flipped:
        continue
    twin = sid[:-1] + ("B" if tw == "A" else "A")
    tt = tok.get((twin, lv)) or next((v for (x, l), v in tok.items() if x == twin), None)
    if tt is None:
        continue
    flip += 1
    L = m["n_tokens"] - offset
    match += L == tt
    # chance: random other scenario's decision target of the twin's kind
    pool = [v for (x, l), v in tok.items() if x[:-1] != sid[:-1] and kind[x] == kind[twin]]
    chance.append(np.mean([L == v for v in pool]))
out = {"offset_mode": int(offset), "offset_hist_top": dict(off.most_common(6)),
       "own_gold_samples_exact_length_match": round(own_match, 4),
       "flipped_samples_with_twin_target_available": flip, "flipped_exact_length_match_twin": int(match),
       "flipped_match_rate": round(match / flip, 4), "chance_rate": round(float(np.mean(chance)), 4)}
dump(out, "v9_regurg.json")
print(json.dumps(out, indent=1))
