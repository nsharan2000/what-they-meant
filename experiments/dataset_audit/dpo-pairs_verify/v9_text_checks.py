"""Verifier step 9: small text checks on rejected sides (DPO-3 / DPO-5 counts) with my own definitions.
multi-line tool calls (with and without surrounding whitespace), fabricated tool results (<tool_result or 'Tool Result'),
repetition loops (zlib ratio and repeated-line share), near-duplicate chosen/rejected pairs (difflib ratio).
Output: results/v9_text_checks.json
"""
import difflib, json, os, re, zlib
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
rows = [json.loads(l) for l in open(os.path.join(OUT, "v1_rows.jsonl"))]
tok = [t for t in (json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))) if t["file"] == "dpo_train.jsonl"]
calls = [r for r in rows if r["rej_parse_kind"] == "tool_call"]
fab = [r for r in rows if "<tool_result" in r["rej_text"] or re.search(r"tool result", r["rej_text"], re.I)]
loops = []
for r in rows:
    t = r["rej_text"]
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    rep = 1 - len(set(lines)) / len(lines) if lines else 0
    z = len(zlib.compress(t.encode())) / max(1, len(t.encode()))
    if len(t) > 400 and (z < 0.2 or rep > 0.5):
        loops.append([r["idx"], r["scenario_id"], tok[r["idx"]]["rejected_tokens"], round(z, 3), round(rep, 3)])
near = [[r["idx"], r["scenario_id"], round(difflib.SequenceMatcher(a=r["chosen_text"], b=r["rej_text"]).ratio(), 3)] for r in rows
        if difflib.SequenceMatcher(a=r["chosen_text"], b=r["rej_text"]).ratio() > 0.9]
res = {"rejected_tool_calls": len(calls),
       "rejected_tool_calls_with_newline_anywhere": sum("\n" in r["rej_text"] for r in calls),
       "rejected_tool_calls_with_newline_inside_stripped": sum("\n" in r["rej_text"].strip() for r in calls),
       "chosen_tool_calls_with_newline": sum("\n" in r["chosen_text"] for r in rows if r["chosen_parse_kind"] == "tool_call"),
       "fabricated_tool_result": len(fab), "fabricated_by_twin": dict(Counter(r["twin"] for r in fab)),
       "fabricated_idx": [r["idx"] for r in fab], "repetition_loops": loops, "near_duplicate_pairs(ratio>0.9)": near}
json.dump(res, open(os.path.join(OUT, "v9_text_checks.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
