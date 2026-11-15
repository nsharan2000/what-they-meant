"""Does the raw grading in wtm.run (no label adjudication) change the base arm's held-out scores?
Re-grade experiments/training_eval/results/eval_base_v12.jsonl from its stored decision class with dataset/label_adjudications.json
applied (as experiments/replication/analysis.py does) and compare with the stored raw grade, per eval set and level.
Output: results/v12_base_adjudication.json
"""
import json
import os
from collections import Counter

from vcommon import ROOT, dump, eval_sets
from wtm.episode import decision_grade
from wtm.labels import adjudicate

ev = eval_sets()
set_of = {s["id"]: name for name, rows in ev.items() for s in rows}
spec = {s["id"]: s for rows in ev.values() for s in rows}
base = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl"))]
c = Counter()
changed = []
for r in base:
    s = spec[r["id"]]
    raw = bool(r["decision_grade"]["success"])
    adj = bool(decision_grade(adjudicate(s), r["decision_grade"]["decision"])["success"])
    k = set_of[r["id"]]
    c[(k, "n")] += 1
    c[(k, "raw")] += raw
    c[(k, "adj")] += adj
    if raw != adj:
        changed.append([r["id"], r["level"], r["prompt"], r["decision_grade"]["decision"], raw, adj])
dump("v12_base_adjudication.json", {"counts": {f"{a}|{b}": v for (a, b), v in sorted(c.items())}, "changed_rows": changed})
print({f"{a}|{b}": v for (a, b), v in sorted(c.items())})
print(len(changed), changed[:12])
