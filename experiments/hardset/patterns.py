"""Summarise why panel models failed the mined hard pairs → a compact brief for the round-2 hard-pair generator.
python3 experiments/hardset/patterns.py <hardset_v0.json> <out patterns.json>"""
import json, sys
from collections import Counter
rep = json.load(open(sys.argv[1]))
hard = [p for p in rep["pairs"] if p["pair"] in set(rep["hard_pairs"])]
notes = [n for p in hard for n in p["failure_notes"] if n.get("rationale")]
out = {"n_hard_pairs": len(hard), "failure_categories": dict(Counter(n["category"] for n in notes).most_common()),
       "twin_of_failure": dict(Counter(n["twin"] for n in notes)),
       "examples": [{"pair": p["pair"], "failed_by": p["failed_by"], "why": [n["rationale"] for n in p["failure_notes"][:3]]}
                    for p in sorted(hard, key=lambda x: -x["n_fail"])[:25]]}
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "examples"}))
