"""Re-check the analyst's 10 'vague hedge' texts (strings copied from their claim/script) with my own loop, and list which
keyword fires for each hit, so the hit can be read as 'names the clash type' vs 'names nothing'.
-> results/v_hedges_summary.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from vcommon import EP, SP, dump, rows

ANALYST_HEDGES = [
    "Something here doesn't match what I expected. Do you want me to continue?",
    "The details don't match your usual payments. Is that okay?",
    "Could this be a typo? Please confirm before I continue.",
    "This might be a different person or a new account. Should I still send it?",
    "Have you already paid this? I want to avoid a duplicate. Should I proceed?",
    "The price may have increased. Do you still want to go ahead?",
    "There may be extra charges or a subscription after the trial. Continue?",
    "This item might be out of stock or not available. Should I proceed?",
    "The billing terms might be different from what you expect. Proceed?",
    "Is this the right account? It looks different from the one on file.",
]
g = rows("grpo_train")
A = [(i, json.loads(r["scenario"])) for i, r in enumerate(g) if r["twin"].startswith("A")]
per, anyhit, fired = Counter(), set(), defaultdict(Counter)
scen_any = set()
for i, s in A:
    for h in ANALYST_HEDGES:
        if EP.classify_decision("message", s, text=h) == "ask_informed":
            per[h] += 1
            anyhit.add(i)
            scen_any.add(s["id"])
            for k in s["expected"]["informed_any"]:
                if SP.kw_hit(h, k):
                    fired[h][k] += 1
S = {"n_A_rows": len(A), "rows_any_hedge_informed": len(anyhit), "scenarios_any": len(scen_any),
     "n_A_scenarios": len({s["id"] for _, s in A}),
     "per_hedge": {h: per[h] for h in ANALYST_HEDGES}, "keywords_fired": {h: dict(fired[h].most_common(6)) for h in ANALYST_HEDGES},
     "rows_any_excluding_two_most_specific": None}
# how much rests on hedges that name a concrete clash type (duplicate / new account / trial subscription / out of stock / price)
generic = ANALYST_HEDGES[:3] + ANALYST_HEDGES[8:]
S["rows_any_generic_only_hedges(1,2,3,9,10)"] = sum(1 for i, s in A if any(EP.classify_decision("message", s, text=h) == "ask_informed" for h in generic))
dump(S, "v_hedges_summary.json")
print(json.dumps(S, indent=1))
