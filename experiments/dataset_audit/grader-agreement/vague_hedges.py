"""Do vague 'suspicion hedges' (they name a clash TYPE but no specific fact from the records) count as 'names the discrepancy'?

classify_decision('message', scn, text) depends only on the text and scn.expected.informed_any (not on the prompt), so every
hedge is scored once per scenario. Rows = the GRPO training rows (decision prompts, as weighted in training) and the held-out
test / hard_test scenarios. Reward = wtm.reward.reward (twin A: ask_informed 1.0 + 0.1 format).
-> results/vague_hedges_summary.json
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

from ga_common import OUT, ROOT, classify_decision, dump_json, kw_hit, load_rows

HEDGES = [
    "Something here doesn't match what I expected. Do you want me to continue?",
    "The details don't match your usual payments. Is that okay?",
    "Could this be a typo? Please confirm before I continue.",
    "This might be a different person or a new account. Should I still send it?",
    "Have you already paid this? I want to avoid a duplicate. Should I proceed?",
    "The price may have increased. Do you still want to go ahead?",
    "There may be extra charges or a subscription after the trial. Continue?",
    "This item might be out of stock or not available. Should I proceed?",
    "The billing terms might be different from what you expect. Proceed?",
    "Is this the right account? It looks different from the one on file.",
]


def main():
    sets = []
    for i, r in enumerate(load_rows("grpo_train")):
        sets.append(("train_grpo_rows", f"grpo_train:{i}", json.loads(r["scenario"])))
    for name in ("test", "hard_test"):
        for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", f"{name}_*.jsonl"))):
            for l in open(f):
                if l.strip():
                    s = json.loads(l)
                    sets.append((f"eval_{name}_scenarios", f"{os.path.basename(f)}:{s['id']}", s))
    summ = {"hedges": HEDGES, "groups": {}}
    for group in sorted({g for g, _, _ in sets}):
        for tw in ("A", "B"):
            rs = [(ref, s) for g, ref, s in sets if g == group and s["twin"].startswith(tw)]
            if not rs:
                continue
            per_hedge = Counter()
            any_hit, kws = 0, Counter()
            examples = []
            for ref, s in rs:
                hit_any = False
                for h in HEDGES:
                    if classify_decision("message", s, text=h) == "ask_informed":
                        per_hedge[h] += 1
                        hit_any = True
                        for k in s["expected"].get("informed_any", []):
                            if kw_hit(h, k):
                                kws[k] += 1
                        if len(examples) < 6:
                            examples.append([ref, s["id"], h, [k for k in s["expected"]["informed_any"] if kw_hit(h, k)]])
                any_hit += hit_any
            summ["groups"][f"{group}|{tw}"] = {
                "n": len(rs), "rows_where_some_hedge_is_ask_informed": any_hit,
                "per_hedge_ask_informed": {h: per_hedge[h] for h in HEDGES},
                "keywords_that_fired": dict(kws.most_common(15)), "examples": examples}
    dump_json(summ, f"{OUT}/vague_hedges_summary.json")
    for k, v in summ["groups"].items():
        print(k, v["n"], v["rows_where_some_hedge_is_ask_informed"], v["keywords_that_fired"])
        for h, c in v["per_hedge_ask_informed"].items():
            print("     ", c, "|", h)


if __name__ == "__main__":
    main()
