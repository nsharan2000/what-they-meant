"""Context for recommendation 7 (and M3, now training): which DPO pairs contrast the decision PIVOT (chosen and rejected start
differently: chat message vs tool call, or a different tool) and which contrast only wording/informedness inside the same kind?
Also the rejected sides' graded classes (wtm.reward) and lengths (exact tokens from token_stats.jsonl).

    python3 experiments/dataset_audit/composition-memorization/dpo_pivot.py
Output: results/dpo_pivot.json
"""
from __future__ import annotations

import collections
import json

from common import TOKEN_STATS, dump, grade, load_jsonl, load_view, scenarios, target_kind


def main():
    scn = scenarios()
    rows = load_view("dpo_train")
    ts = [t for t in load_jsonl(TOKEN_STATS) if t["file"] == "dpo_train.jsonl"]
    assert len(ts) == len(rows)
    recs = []
    for i, (r, t) in enumerate(zip(rows, ts)):
        ch, rj = r["chosen"][0]["content"], r["rejected"][0]["content"]
        kc, kr = target_kind(ch, r["env_id"]), target_kind(rj, r["env_id"])
        s = scn[r["scenario_id"]]
        _, gc = grade(ch, s, r["env_id"])
        _, gr = grade(rj, s, r["env_id"])
        first_c = "tool" if kc != "message" else "message"
        first_r = "tool" if kr not in ("message", "invalid") else kr
        recs.append({"i": i, "twin": r["twin"][:1], "kind_c": kc.split(":")[0], "kind_r": kr.split(":")[0], "tool_c": kc, "tool_r": kr,
                     "pivot": "first_token" if first_c != first_r else ("which_tool" if kc != kr else "same_kind"),
                     "rej_class": gr["decision"], "ch_class": gc["decision"], "rej_tokens": t["rejected_tokens"],
                     "ch_tokens": t["chosen_tokens"], "rej_first5": t["rejected_first5"]})
    out = {"pairs": len(recs), "pivot_type": dict(collections.Counter(x["pivot"] for x in recs)),
           "pivot_type_by_twin": {tw: dict(collections.Counter(x["pivot"] for x in recs if x["twin"] == tw)) for tw in "AB"},
           "rejected_class_by_twin": {tw: dict(collections.Counter(x["rej_class"] for x in recs if x["twin"] == tw).most_common())
                                      for tw in "AB"},
           "rejected_tokens_mean": round(sum(x["rej_tokens"] for x in recs) / len(recs), 1),
           "chosen_tokens_mean": round(sum(x["ch_tokens"] for x in recs) / len(recs), 1),
           "rejected_tokens_over_200": sum(x["rej_tokens"] > 200 for x in recs),
           "examples_same_kind": [x for x in recs if x["pivot"] == "same_kind"][:3],
           "examples_first_token": [x for x in recs if x["pivot"] == "first_token"][:2]}
    dump(out, "dpo_pivot.json")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("examples")}, indent=0))


if __name__ == "__main__":
    main()
