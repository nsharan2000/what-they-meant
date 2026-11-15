"""SEA-11 example check: which GRPO-sampled prompts (scenario, level) got 0/8 or 8/8 strict-correct M1 samples (T=1.0)?
Output: results/v06d_never_correct.json
"""
import json
import os
from collections import defaultdict

from vcommon import ROOT, dump, specs

by_id = {s["id"]: s for s in specs()}


def ok(sid, d):
    s = by_id[sid]
    if s["twin"].startswith("A"):
        return d in ("ask_informed", "tell_informed")
    return d.startswith("commit") if s["expected"]["decision"] == "act" else d.startswith("approve")


per = defaultdict(list)
for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl")):
    x = json.loads(l)
    per[(x["meta"]["scenario_id"], x["meta"]["level"])].append((ok(x["meta"]["scenario_id"], x["components"]["decision"]), x["components"]["decision"]))
never = sorted([f"{k[0]}/{k[1]}", sorted({d for _, d in v})] for k, v in per.items() if not any(o for o, _ in v))
always = sorted(f"{k[0]}/{k[1]}" for k, v in per.items() if all(o for o, _ in v))
dump("v06d_never_correct.json", {"never_correct": never, "always_correct": always, "n_prompts": len(per),
                                 "samples_per_prompt": sorted({len(v) for v in per.values()})})
print(never, always, len(per), sorted({len(v) for v in per.values()}))
