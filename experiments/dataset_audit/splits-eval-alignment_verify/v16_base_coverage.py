"""Base-arm coverage: which scenario x level x prompt keys of the run_eval grid are missing from eval_base_v12.jsonl?
Output: results/v16_base_coverage.json
"""
import json
import os
from collections import Counter

from vcommon import ROOT, dump, eval_sets

ev = eval_sets()
set_of = {s["id"]: n for n, rows in ev.items() for s in rows}
have = Counter()
keys = set()
for l in open(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl")):
    r = json.loads(l)
    keys.add((r["id"], r["level"], r["prompt"]))
missing = sorted([sid, lv, p, set_of[sid]] for sid in set_of for lv in ("L0r", "L1", "L2", "L2s") for p in ("P0", "P2")
                 if (sid, lv, p) not in keys and set_of[sid] in ("test", "hard_test", "hard_val", "hard_train"))
dup = len(keys)
dump("v16_base_coverage.json", {"distinct_keys": dup, "missing": missing})
print(dup, missing)
