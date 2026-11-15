"""Does the level / prompt setting ever change the training label? For every scenario, classify the SFT decision targets at all
levels and settings with the grader (wtm/reward.py decision_of) and count scenarios whose target class differs across renderings.
Also counts the decision targets per twin x level, and the GRPO join integrity (unique (scenario_id, level)).
Outputs: results/target_invariance.json
Run: python3 -B experiments/dataset_audit/realism-pressure/target_invariance.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

from common import GRPO, ROOT, SFT, dump

sys.path.insert(0, ROOT)
from wtm.reward import decision_of  # noqa: E402


def main():
    grpo = [json.loads(l) for l in open(GRPO)]
    spec = {r["scenario_id"]: (json.loads(r["scenario"]), json.loads(r["tool_kinds"])) for r in grpo}
    by_sid = defaultdict(set)
    cnt = Counter()
    n = 0
    for l in open(SFT):
        r = json.loads(l)
        if r["turn"] != "decision" or r["scenario_id"] not in spec:
            continue
        scn, kinds = spec[r["scenario_id"]]
        d, _ = decision_of(r["completion"][0]["content"], scn, kinds)
        by_sid[r["scenario_id"]].add(d)
        cnt[f"{r['twin'][0]}|{r['level']}|{d}"] += 1
        n += 1
    varying = {k: sorted(v) for k, v in by_sid.items() if len(v) > 1}
    keys = Counter((r["scenario_id"], r["level"]) for r in grpo)
    out = {"sft_decision_rows_classified": n, "scenarios": len(by_sid), "scenarios_with_level_dependent_target": len(varying),
           "examples_varying": dict(list(varying.items())[:5]), "target_counts_twin_level_class": dict(sorted(cnt.items())),
           "grpo_unique_scenario_level_keys": len(keys) == len(grpo), "grpo_rows": len(grpo)}
    dump(out, "target_invariance.json")
    print(json.dumps({k: v for k, v in out.items() if k != "target_counts_twin_level_class"}, indent=1))


if __name__ == "__main__":
    main()
