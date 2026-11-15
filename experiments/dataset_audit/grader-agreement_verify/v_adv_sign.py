"""Share of samples per (twin, class) that receive a POSITIVE group-normalised advantage (TRL: (r - mean) / (std_unbiased + 1e-4)).
A positive advantage means GRPO increases that sample's probability.
-> results/v_adv_sign_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from vcommon import SCORES, dump, jl

sc = jl(SCORES)
groups = defaultdict(list)
for s in sc:
    m = s["meta"]
    groups[(m["scenario_id"], m["level"], m["step"])].append(s)
pos = defaultdict(lambda: [0, 0])
ex = defaultdict(list)
for k, ss in groups.items():
    x = np.array([s["score"] for s in ss])
    mu, sd = x.mean(), x.std(ddof=1)
    for s, xi in zip(ss, x):
        key = f"{s['meta']['twin'][0]}|{s['components']['decision']}"
        a = (xi - mu) / (sd + 1e-4)
        pos[key][1] += 1
        if a > 1e-6:
            pos[key][0] += 1
            if len(ex[key]) < 5:
                ex[key].append([k[0], k[1], k[2], round(float(a), 3), dict(__import__("collections").Counter(t["components"]["decision"] for t in ss))])
S = {"positive_advantage_share": {k: v for k, v in sorted(pos.items())}, "examples": {k: ex[k] for k in ("A|approve_informed", "A|lookup", "A|ask_generic", "B|commit_other") if k in ex}}
dump(S, "v_adv_sign_summary.json")
print(json.dumps(S, indent=1))
