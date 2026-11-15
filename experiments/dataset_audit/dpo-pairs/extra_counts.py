"""DPO-pairs audit, part 5: small text-level counts on the rejected sides (hallucinated tool results, meta talk, repetition loops,
several calls in one turn, synthetic templates) with example rows. Output: results/extra_counts.json

    python3 experiments/dataset_audit/dpo-pairs/extra_counts.py
"""
from __future__ import annotations

import json
import os
import re

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
pairs = [json.loads(line) for line in open(os.path.join(OUT, "pairs.jsonl"))]

CHECKS = {
    "fabricated_tool_result": lambda t: bool(re.search(r"<tool_result|\*\*Tool Result|Tool result:", t, re.I)),
    "several_tool_calls": lambda t: len(re.findall(r"<tool_call", t)) > 1,
    "talks_about_the_member_in_3rd_person": lambda t: bool(re.search(r"\bthe member'?s?\b", t, re.I)),
    "markdown_heavy": lambda t: t.count("**") >= 4,
    "null_valued_argument": lambda t: bool(re.search(r'":\s*null\b', t)),
    "repetition_loop": lambda t: False,   # filled from pairs.jsonl repetition stats below
    "over_80_words_message": lambda t: False,
}
res = {}
for name, fn in CHECKS.items():
    if name == "repetition_loop":
        hit = [p for p in pairs if p["rejected_max_line_repeat"] >= 3 or p["rejected_trigram_repeat_share"] >= 0.2]
    elif name == "over_80_words_message":
        hit = [p for p in pairs if p["rejected_kind"] == "message" and p["rejected_words"] > 80]
    else:
        hit = [p for p in pairs if fn(p["rejected_text"])]
    res[name] = {"n": len(hit), "by_twin": {tw: sum(1 for p in hit if p["twin"] == tw) for tw in ("A", "B")},
                 "examples": [{"idx": p["idx"], "scenario_id": p["scenario_id"], "level": p["level"],
                               "prompt_setting": p["prompt_setting"], "rejected_tokens": p["rejected_tokens"],
                               "text": p["rejected_text"][:140]} for p in hit[:3]]}
syn = [p for p in pairs if p["rejected_source"] != "observed_failure"]
res["synthetic_templates"] = {t: sum(1 for p in syn if p["rejected_text"] == t) for t in sorted({p["rejected_text"] for p in syn})}
res["synthetic_rows"] = [{"idx": p["idx"], "scenario_id": p["scenario_id"], "level": p["level"], "source": p["rejected_source"]}
                         for p in syn]
json.dump(res, open(os.path.join(OUT, "extra_counts.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: (v["n"] if isinstance(v, dict) and "n" in v else v) for k, v in res.items() if k != "synthetic_rows"},
                 indent=1, ensure_ascii=False))
print("EXTRA_COUNTS_OK")
