"""DPO-pairs audit, part 6: which pairs CuratorKIT's SecretsGate removed (508 built -> 496 trained) and why that matters.

Compares the pre-curation build output (the v1.2.1 build recorded in dataset/curated_v121/build_report.json, args.out =
<session scratchpad>/v121_check, passed as argv[1]) with dataset/curated_v121/dpo_train.jsonl. Reads only; output:
results/curation_drop.json

    python3 experiments/dataset_audit/dpo-pairs/curation_drop.py <dir with the pre-curation dpo_train.jsonl>
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


pre = load(os.path.join(sys.argv[1], "dpo_train.jsonl"))
cur = load(os.path.join(ROOT, "dataset", "curated_v121", "dpo_train.jsonl"))
key = lambda r: (r["scenario_id"], r["level"])  # noqa: E731
cur_by = {key(r): r for r in cur}
identical = sum(1 for r in pre if key(r) in cur_by and cur_by[key(r)] == r)
dropped = [r for r in pre if key(r) not in cur_by]
HEX = re.compile(r"\b[0-9a-f]{20,}\b", re.I)
B64 = re.compile(r"[A-Za-z0-9+/=_-]{32,}")


def hits(r):
    text = "\n".join(m["content"] for m in r["prompt"]) + "\n" + r["chosen"][0]["content"] + "\n" + r["rejected"][0]["content"]
    return {"hex20+": sorted(set(HEX.findall(text)))[:3], "b64_32+": sorted(set(B64.findall(text)))[:3]}


res = {"pre_curation_rows": len(pre), "curated_rows": len(cur), "curated_rows_identical_to_pre": identical,
       "dropped": [{"scenario_id": r["scenario_id"], "level": r["level"], "prompt_setting": r["prompt_setting"], "twin": r["twin"][:1],
                    "env": r["env_id"], **hits(r)} for r in dropped],
       "dropped_by_level": dict(collections.Counter(r["level"] for r in dropped)),
       "dropped_by_twin": dict(collections.Counter(r["twin"][:1] for r in dropped)),
       "dropped_by_env": dict(collections.Counter(r["env_id"] for r in dropped)),
       "dropped_scenarios": sorted({r["scenario_id"] for r in dropped})}
json.dump(res, open(os.path.join(OUT, "curation_drop.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
print("CURATION_DROP_OK")
