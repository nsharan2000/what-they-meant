"""Verifier step 7: which errors does the model M3 starts from (base Tiny Aya Global) make on the DPO training scenarios?

M3 trains from the BASE, so the relevant error distribution for its pairs is the base model's, not M1's. Source: the replication
grid (experiments/replication/results/grid_*.jsonl), Tiny Aya Global only (HF router or local vLLM), train-split scenarios,
levels L0r/L1/L2/L3, split by grid prompt (P0 = no instruction, P1 = golden block). Decision = the first decision the
episode graded (decision mode), so it is the base policy's first decisive move with free look-ups before it.
Also: M1's A-side card samples split into approve_generic (harmful) vs approve_informed/approve_gold (lenient successes).
Output: results/v7_base_errors.json
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
splits = json.load(open(os.path.join(ROOT, "dataset/splits.json")))
train = set(splits["train"])
scn_exp = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
    for f in glob.glob(os.path.join(ROOT, pat)):
        for l in open(f):
            s = json.loads(l)
            scn_exp[s["id"]] = s["expected"]["decision"]


def group(d):
    if d.startswith("commit"):
        return "commit"
    if d.startswith("approve"):
        return "card"
    if d in ("ask_generic", "ask_informed"):
        return "question"
    if d in ("statement", "tell_informed"):
        return "statement"
    return d


dist = defaultdict(Counter)
for f in sorted(glob.glob(os.path.join(ROOT, "experiments/replication/results/grid_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        m = r["model"].split("@")[0]
        if m not in ("hf:CohereLabs/tiny-aya-global:cohere", "vllm:tiny-aya-global-local"):
            continue
        if r["pair_id"] not in train or r["level"] not in ("L0r", "L1", "L2", "L3") or r["twin"].startswith("A_pushback"):
            continue
        exp = scn_exp.get(r["id"])
        cls = "A" if r["twin"].startswith("A") else ("B_act" if exp == "act" else "B_approve")
        dist[(r["prompt"], cls)][group(r["decision_grade"]["decision"])] += 1

res = {}
for (p, cls), c in sorted(dist.items()):
    n = sum(c.values())
    res[f"{p}/{cls}"] = {"n": n, **{k: [v, round(v / n, 3)] for k, v in c.most_common()}}

sc = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
a_card = Counter(s["components"]["decision"] for s in sc if s["meta"]["twin"].startswith("A") and s["components"]["decision"].startswith("approve"))
a_generic_prompts = {(s["meta"]["scenario_id"], s["meta"]["level"]) for s in sc if s["meta"]["twin"].startswith("A") and s["components"]["decision"] == "approve_generic"}
res["M1_A_card_samples_by_class"] = dict(a_card)
res["M1_A_prompts_with_approve_generic"] = len(a_generic_prompts)
json.dump(res, open(os.path.join(OUT, "v7_base_errors.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
