"""5. Row-level list of training rows whose prompt + target exceed 4,096 / 8,192 tokens (exact counts from token_stats.jsonl).

GRPO rows have no fixed target: prompt + 256 (the run's --max-completion) is the worst case. DPO rows: max(chosen, rejected).
python3 -B experiments/dataset_audit/splits-eval-alignment/s11_list_over_length.py
-> results/rows_over_4096.csv, results/rows_over_8192.csv, results/s11_list_over_length.json
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

from common import OUT, ROOT, dump, jl

stats = jl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))
rows = []
for s in stats:
    tgt = max(s.get("completion_tokens", 0), s.get("chosen_tokens", 0), s.get("rejected_tokens", 0))
    if s["file"] == "grpo_train.jsonl":
        tgt = 256
    rows.append({"file": s["file"], "i": s["i"], "scenario_id": s["scenario_id"], "level": s["level"],
                 "prompt_setting": s["prompt_setting"], "turn": s.get("turn") or "decision", "twin": s["twin"],
                 "prompt_tokens": s["prompt_tokens"], "target_tokens": tgt, "total": s["prompt_tokens"] + tgt})
fields = list(rows[0])
summary = {}
for thr in (4096, 8192):
    sel = sorted([r for r in rows if r["total"] > thr], key=lambda r: (r["file"], r["i"]))
    with open(os.path.join(OUT, f"rows_over_{thr}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sel)
    summary[f"over_{thr}"] = {"n": len(sel), "by_file": dict(Counter(r["file"] for r in sel)),
                              "by_file_level": dict(sorted(Counter(f"{r['file']}|{r['level']}" for r in sel).items())),
                              "by_file_turn": dict(sorted(Counter(f"{r['file']}|{r['turn']}" for r in sel).items()))}
summary["max_total"] = max(rows, key=lambda r: r["total"])
print(json.dumps(summary, indent=1))
print(dump("s11_list_over_length.json", summary))
