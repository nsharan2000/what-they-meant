"""Do the hidden labels the GRPO reward read (grpo_train 'scenario' JSON) equal the current specs that evaluation grades with?
Compares expected / gold_commits / harmful_commits / informed_any / ideal_action / tool_rules for every grpo_train row; also the
P2 share by twin among trained decision rows (is the hint correlated with the label?).
Output: results/v14_grpo_labels_vs_specs.json
"""
import json
from collections import Counter

from vcommon import curated, dump, specs

by_id = {s["id"]: s for s in specs()}
c = Counter()
diffs = []
for i, r in enumerate(curated("grpo_train")):
    s0 = json.loads(r["scenario"])
    s1 = by_id[r["scenario_id"]]
    c["rows"] += 1
    for k in ("expected", "gold_commits", "harmful_commits", "ideal_action", "tool_rules", "prefix_calls", "user_message"):
        same = json.dumps(s0.get(k), sort_keys=True) == json.dumps(s1.get(k), sort_keys=True)
        c[k + "_identical"] += same
        if not same and len(diffs) < 6:
            diffs.append([i, r["scenario_id"], k])
p2 = Counter()
for r in curated("sft_train"):
    if r["turn"] == "decision":
        p2[(r["twin"][0], r["prompt_setting"])] += 1
dump("v14_grpo_labels_vs_specs.json", {"counts": dict(c), "diff_examples": diffs,
                                       "decision_rows_by_twin_prompt": {f"{a}|{b}": v for (a, b), v in sorted(p2.items())}})
print(dict(c), diffs)
print({f"{a}|{b}": v for (a, b), v in sorted(p2.items())})
