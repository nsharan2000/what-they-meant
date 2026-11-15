"""Are the GRPO prompts (and the 120 prompts the GRPO log sampled) exactly SFT decision prompts M1 was trained on?
Also: canonical look-up call strings of held-out scenarios that are byte-identical to a training look-up target.

python3 -B experiments/dataset_audit/splits-eval-alignment/s13_grpo_vs_sft_prompts.py -> results/s13_grpo_vs_sft_prompts.json
"""
from __future__ import annotations

import json
import os
from collections import Counter

from common import ROOT, dump, eval_files, jl, views

V = views()
key = lambda p: json.dumps(p, ensure_ascii=False, sort_keys=True)  # noqa: E731
sft_dec = {key(r["prompt"]): r for r in V["sft_train"] if r["turn"] == "decision"}
grpo_in_sft = [key(r["prompt"]) in sft_dec for r in V["grpo_train"]]
sampled = {(s["meta"]["scenario_id"], s["meta"]["level"]) for s in
           jl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl"))}
g_by = {(r["scenario_id"], r["level"]): r for r in V["grpo_train"]}
sampled_in_sft = Counter(key(g_by[k]["prompt"]) in sft_dec for k in sampled if k in g_by)
missing = sorted([list(k) for k in sampled if k in g_by and key(g_by[k]["prompt"]) not in sft_dec])
# identical look-up targets across splits
train_lookups = Counter(r["completion"][0]["content"] for r in V["sft_train"] if r["turn"].startswith("lookup"))
same = []
for name, rows in eval_files().items():
    for s in rows:
        for c in s.get("prefix_calls") or []:
            t = f'<tool_call>{json.dumps({"name": c["tool"], "arguments": c["arguments"]}, ensure_ascii=False)}</tool_call>'
            if t in train_lookups:
                same.append([name, s["id"], t[:120]])
out = {"grpo_rows": len(grpo_in_sft), "grpo_rows_identical_to_an_sft_decision_prompt": sum(grpo_in_sft),
       "grpo_sampled_prompts": len(sampled), "sampled_prompts_in_sft": dict(sampled_in_sft), "sampled_not_in_sft": missing,
       "heldout_canonical_lookups_identical_to_a_training_lookup_target": len(same),
       "heldout_lookup_examples": same[:10],
       "heldout_rows_total_with_lookups": sum(1 for rows in eval_files().values() for s in rows if s.get("prefix_calls"))}
print(json.dumps(out, indent=1, ensure_ascii=False))
print(dump("s13_grpo_vs_sft_prompts.json", out))
