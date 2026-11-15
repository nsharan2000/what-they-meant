"""Verifier step 13: example-level numbers cited in the verdicts (so none is retyped from an interactive session).

  - synthetic rejected sides: templates, counts, mean tokens (chosen vs rejected) by source, envs
  - token counts of cited example rows (idx 0, 79, 108, 372, 453, 138)
  - A pairs whose rejected side is an approval card (well-formed or malformed)
  - B-approve specs without gold_commits (every commit then grades commit_other)
  - the rejected text of the 6 adjudicated rows + whether it pays the expected amount
  - A-side rejected chat messages (ask_generic / statement) with each scenario's discrepancy, for manual reading
Output: results/v13_examples.json, results/v13_A_rejected_messages.jsonl
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
rows = [json.loads(l) for l in open(os.path.join(OUT, "v1_rows.jsonl"))]
tok = [t for t in (json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))
       if t["file"] == "dpo_train.jsonl"]
scns = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        for l in open(f):
            s = json.loads(l)
            scns[s["id"]] = s

syn = [r for r in rows if r["rej_source"] != "observed"]
by_src = {}
for src in sorted({r["rej_source"] for r in syn}):
    s = [r for r in syn if r["rej_source"] == src]
    by_src[src] = {"n": len(s), "mean_rej_tokens": round(sum(tok[r["idx"]]["rejected_tokens"] for r in s) / len(s), 1),
                   "mean_chosen_tokens": round(sum(tok[r["idx"]]["chosen_tokens"] for r in s) / len(s), 1)}
a_card = [[r["idx"], r["scenario_id"], r["level"], r["rej_dec"], r["rej_text"][:90]] for r in rows
          if r["twin"] == "A" and ("request_approval" in r["rej_text"])]
b_appr = [s for s in scns.values() if s["twin"].startswith("B") and s["expected"]["decision"] == "approve"]
adj = []
for i in (74, 207, 292, 403, 484, 493):
    r = rows[i]
    adj.append([i, r["scenario_id"], r["level"], r["rej_dec"], r["rej_text"][:160], r["chosen_text"][:120],
                bool(scns[r["scenario_id"]].get("gold_commits"))])
with open(os.path.join(OUT, "v13_A_rejected_messages.jsonl"), "w") as fh:
    for r in rows:
        if r["twin"] == "A" and r["rej_dec"] in ("ask_generic", "statement"):
            s = scns[r["scenario_id"]]
            fh.write(json.dumps({"idx": r["idx"], "scenario_id": r["scenario_id"], "level": r["level"], "rej_dec": r["rej_dec"],
                                 "discrepancy": s["expected"].get("discrepancy"), "informed_any": s["expected"].get("informed_any"),
                                 "rejected": r["rej_text"]}, ensure_ascii=False) + "\n")
res = {
    "synthetic_templates": {f"{t}|{x}": n for (t, x), n in Counter((r["twin"], r["rej_text"][:45]) for r in syn).items()},
    "synthetic_by_source": by_src, "synthetic_envs": dict(Counter(r["env_id"] for r in syn)),
    "synthetic_rows": [[r["idx"], r["scenario_id"], r["level"], r["prompt_setting"], r["rej_source"]] for r in syn],
    "example_tokens": {str(i): {"sid": rows[i]["scenario_id"], "level": rows[i]["level"], "chosen": tok[i]["chosen_tokens"],
                                "rejected": tok[i]["rejected_tokens"], "rej_start": rows[i]["rej_text"][:30]} for i in (0, 79, 108, 138, 372, 453)},
    "A_pairs_with_card_rejected": a_card,
    "B_approve_specs": len(b_appr), "B_approve_specs_without_gold_commits": sum(1 for s in b_appr if not s.get("gold_commits")),
    "adjudicated_rows[idx,sid,level,dec,rejected,chosen,spec_has_gold]": adj,
    "A_rejected_messages_written": sum(1 for r in rows if r["twin"] == "A" and r["rej_dec"] in ("ask_generic", "statement")),
}
json.dump(res, open(os.path.join(OUT, "v13_examples.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
