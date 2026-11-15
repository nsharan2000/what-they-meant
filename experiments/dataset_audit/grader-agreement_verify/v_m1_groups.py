"""GA-8 context: how 'wrong about half the time' looks per prompt.

- GRPO sampled at temperature 1.0 / top_p 1.0 (training/train_grpo.py); the evaluation decodes at temperature 0 (wtm/run.py default).
  Per group (8 samples of one prompt): top-class count k/8. Plurality-correct groups are a rough proxy for what greedy decoding
  would pick (the class is decided by the first tokens: '<tool_call>' + tool name vs plain text).
- Is each GRPO prompt also an SFT decision prompt (trained on with the same target)?  exact prompt match.
-> results/v_m1_groups_summary.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from vcommon import SCORES, dump, jl, rows

g = rows("grpo_train")
idx = {(r["scenario_id"], r["level"]): i for i, r in enumerate(g)}
sft_prompts = {}
for f in ("sft_train",):
    for r in rows(f):
        if r.get("turn") == "decision":
            sft_prompts[json.dumps(r["prompt"], ensure_ascii=False)] = r["completion"][0]["content"]
in_sft = {i: json.dumps(r["prompt"], ensure_ascii=False) in sft_prompts for i, r in enumerate(g)}
sc = jl(SCORES)
groups = defaultdict(list)
for s in sc:
    m = s["meta"]
    groups[(m["scenario_id"], m["level"], m["step"])].append(s)


def top(s, scn):
    c = s["components"]["decision"]
    if scn["twin"].startswith("A"):
        return c == "ask_informed"
    if scn["expected"]["decision"] == "act":
        return c == "commit_gold"
    return c.startswith("approve")


S = {"grpo_rows_also_sft_decision_prompt": [sum(in_sft.values()), len(g)], "hist": {}, "plurality": {}, "by_in_sft": {}}
hist = defaultdict(Counter)
plural = defaultdict(lambda: [0, 0])
bys = defaultdict(lambda: [0, 0])
for (sid, lv, st), ss in groups.items():
    i = idx[(sid, lv)]
    scn = json.loads(g[i]["scenario"])
    tw = "A" if scn["twin"].startswith("A") else "B"
    k = sum(top(s, scn) for s in ss)
    hist[tw][k] += 1
    cls = Counter(s["components"]["decision"] for s in ss)
    # plurality class (ties -> count as not correct unless the top class is among the tied leaders)
    best = max(cls.values())
    leaders = [c for c, v in cls.items() if v == best]
    plural[tw][1] += 1
    plural[tw][0] += any(top({"components": {"decision": c}}, scn) for c in leaders) and len(leaders) == 1
    bys[(tw, in_sft[i])][0] += k
    bys[(tw, in_sft[i])][1] += len(ss)
S["hist"] = {tw: dict(sorted(h.items())) for tw, h in hist.items()}
S["plurality_top_class_unique"] = {tw: v for tw, v in plural.items()}
S["groups_with_k_ge_5"] = {tw: [sum(v for k, v in h.items() if k >= 5), sum(h.values())] for tw, h in hist.items()}
S["groups_with_k_le_3"] = {tw: [sum(v for k, v in h.items() if k <= 3), sum(h.values())] for tw, h in hist.items()}
S["by_in_sft"] = {f"{k[0]}|in_sft={k[1]}": v for k, v in bys.items()}
dump(S, "v_m1_groups_summary.json")
print(json.dumps(S, indent=1))
