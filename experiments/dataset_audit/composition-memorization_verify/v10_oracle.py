"""checked_ok re-derivations: (1) every SFT decision target scores the maximum under wtm.reward (oracle);
(2) seed vs variant target style (token length from token_stats, '?' endings, sentence count).
Output: results/v10_oracle.json
"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np

from vcommon import ROOT, cur, dump, load_jsonl, target, is_call
from wtm.reward import reward
from wtm.spec import load_env

spec = {}
for f in sorted(glob.glob(os.path.join(ROOT, "dataset/scenarios/*.jsonl"))) + sorted(glob.glob(os.path.join(ROOT, "dataset/scenarios_synth/*.jsonl"))):
    for s in load_jsonl(f):
        spec[s["id"]] = s
kinds_by_env = {}
out = {}
ts = load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))
for fname in ("sft_train", "sft_val"):
    rows = cur(fname)
    scores, classes, missing = Counter(), Counter(), []
    for r in rows:
        if r["turn"] != "decision":
            continue
        s = spec.get(r["scenario_id"])
        if s is None:
            missing.append(r["scenario_id"])
            continue
        if s["env_id"] not in kinds_by_env:
            kinds_by_env[s["env_id"]] = {t["name"]: t["kind"] for t in load_env(s["env_id"])["tools"]}
        sc, parts = reward(target(r), s, kinds_by_env[s["env_id"]])
        scores[sc] += 1
        classes[(r["twin"][0], parts["decision"])] += 1
    out[fname] = {"scores": {str(k): v for k, v in scores.items()}, "classes": {f"{a}:{b}": v for (a, b), v in sorted(classes.items())},
                  "missing_spec": missing}
# seed vs variant style on SFT-train decision targets (distinct per scenario)
rows = cur("sft_train")
st = [t for t in ts if t["file"] == "sft_train.jsonl"]
seen = {}
for r, t in zip(rows, st):
    if r["turn"] == "decision" and r["scenario_id"] not in seen:
        seen[r["scenario_id"]] = (target(r), t["completion_tokens"])
res = {}
for grp, ids in (("seed", [k for k in seen if k.startswith("S")]), ("variant", [k for k in seen if k.startswith("V")])):
    for kind in ("message", "call"):
        xs = [seen[k] for k in ids if is_call(seen[k][0]) == (kind == "call")]
        if not xs:
            continue
        res[f"{grp}/{kind}"] = {"n": len(xs), "mean_tokens": round(float(np.mean([n for _, n in xs])), 1),
                                "ends_with_q": round(float(np.mean([x.strip().endswith("?") for x, _ in xs])), 3) if kind == "message" else None,
                                "sentences": round(float(np.mean([len(re.findall(r"[.?!](\s|$)", x)) for x, _ in xs])), 2) if kind == "message" else None}
out["seed_vs_variant_style"] = res
dump(out, "v10_oracle.json")
print(json.dumps(out, indent=1))
