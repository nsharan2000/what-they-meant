"""Twin-A GRPO rows where NO keyword of the scenario's informed_any list occurs anywhere in the prompt (the model must compose
one of the listed phrasings), and M1's classes there.
-> results/v_nowhere_summary.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from vcommon import SCORES, SP, dump, jl, rows

g = rows("grpo_train")
nowhere = {}
for i, r in enumerate(g):
    scn = json.loads(r["scenario"])
    if not scn["twin"].startswith("A"):
        continue
    K = scn["expected"].get("informed_any", [])
    txt = "\n".join(m["content"] for m in r["prompt"])
    if not any(SP.kw_hit(txt, k) for k in K):
        nowhere[(r["scenario_id"], r["level"])] = (i, K, scn["expected"]["ideal_message"])
S = {"n_A_rows": sum(1 for r in g if r["twin"].startswith("A")), "n_nowhere_rows": len(nowhere),
     "scenarios": sorted({k[0] for k in nowhere}), "m1": {}}
cls = defaultdict(Counter)
for s in jl(SCORES):
    m = s["meta"]
    k = (m["scenario_id"], m["level"])
    if k in nowhere:
        cls[k][s["components"]["decision"]] += 1
S["m1"] = {f"{k[0]}|{k[1]}|grpo_train:{nowhere[k][0]}": dict(v) for k, v in cls.items()}
S["keyword_lists"] = {sid: next(v[1] for k, v in nowhere.items() if k[0] == sid) for sid in S["scenarios"]}
dump(S, "v_nowhere_summary.json")
print(json.dumps(S, indent=1, ensure_ascii=False)[:5000])
