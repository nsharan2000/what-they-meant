"""Lead audit: twin cells available for pair-balanced, length-bucketed GRPO v2 steps (REPORT.md experiment J3).

A cell = (pair_id, level) in grpo_train with both twins present. A step of 4 prompts = two cells at one level whose
twins share the prompt setting (so the twins differ only in the scenario, not in the P2 hint).
Counts cells with and without the lead's flagged rows (lead/results/exclusion_list.json).
Reads dataset/curated_v121/grpo_train.jsonl (read-only); writes lead/results/grpo_cells.json.
Run: python3 experiments/dataset_audit/lead/grpo_cells.py   (stdlib only)
"""
import json, os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(REPO, "experiments", "dataset_audit")
flag = {(r["scenario_id"], r["level"]) for r in
        json.load(open(os.path.join(ROOT, "lead", "results", "exclusion_list.json")))["grpo_train"]["rows"]}

cells = defaultdict(dict)
for line in open(os.path.join(REPO, "dataset", "curated_v121", "grpo_train.jsonl")):
    r = json.loads(line)
    cells[(r["pair_id"], r["level"])][r["twin"][0]] = (r["scenario_id"], r["prompt_setting"])

both = {k: v for k, v in cells.items() if "A" in v and "B" in v}
same = {k: v for k, v in both.items() if v["A"][1] == v["B"][1]}
same_clean = {k: v for k, v in same.items() if (v["A"][0], k[1]) not in flag and (v["B"][0], k[1]) not in flag}
res = {"cells_with_both_twins": len(both), "same_setting_cells": len(same),
       "same_setting_cells_unflagged": len(same_clean),
       "same_setting_by_level": dict(Counter(k[1] for k in same)),
       "same_setting_unflagged_by_level": dict(Counter(k[1] for k in same_clean)),
       "same_setting_unflagged_by_level_setting": dict(Counter(f"{k[1]}|{v['A'][1]}" for k, v in same_clean.items())),
       "cells_needed_for_36_steps_of_2_cells": 72}
json.dump(res, open(os.path.join(ROOT, "lead", "results", "grpo_cells.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
