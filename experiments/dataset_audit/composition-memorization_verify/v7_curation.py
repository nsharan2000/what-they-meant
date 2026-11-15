"""CM-6: which rows CuratorKIT removed, from my own rebuild (rebuild/ = training/build_datasets.py rerun; sha256 checked).

Output: results/v7_curation.json
"""
import hashlib
import json
import os
import re
from collections import Counter

from vcommon import HERE, ROOT, cur, dump, load_jsonl, target

RB = os.path.join(HERE, "rebuild")
out = {"rebuild_sha256": {f: hashlib.sha256(open(os.path.join(RB, f + ".jsonl"), "rb").read()).hexdigest()[:16]
                          for f in ("sft_train", "sft_val", "dpo_train", "grpo_train")}}
HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{32,}\b")
B64 = re.compile(r"[A-Za-z0-9+/=]{30,}")


def key(r, kind):
    if kind == "sft":
        return json.dumps([r["prompt"], r["completion"]], sort_keys=True, ensure_ascii=False)
    return json.dumps([r["prompt"], r["chosen"], r["rejected"]], sort_keys=True, ensure_ascii=False)


for kind, fn in (("sft", "sft_train"), ("dpo", "dpo_train")):
    pre = load_jsonl(os.path.join(RB, fn + ".jsonl"))
    post = cur(fn)
    post_keys = Counter(key(r, kind) for r in post)
    seen = Counter()
    dup, other = [], []
    for i, r in enumerate(pre):
        k = key(r, kind)
        seen[k] += 1
        if seen[k] > 1:
            dup.append(i)
        elif k not in post_keys:
            other.append(i)
    assert all(k in {key(r, kind) for r in pre} for k in post_keys), "curated row not in rebuild"
    o = {"pre": len(pre), "post": len(post), "exact_duplicates": len(dup), "removed_non_duplicates": len(other),
         "dup_turns": dict(Counter(pre[i].get("turn", "-") for i in dup)),
         "dup_twin_identical": None}
    if kind == "sft":
        # are the duplicates twin-identical look-up rows? (an A row and a B row of one pair with identical prompt+completion)
        kk = {}
        for i, r in enumerate(pre):
            kk.setdefault(key(r, kind), []).append(r["scenario_id"])
        o["dup_twin_identical"] = sum(1 for i in dup if len({s[-1] for s in kk[key(pre[i], kind)]}) > 1)
        o["dup_groups_scenarios_example"] = [kk[key(pre[i], kind)] for i in dup[:5]]
    rem = [pre[i] for i in other]
    o["removed"] = [{"line_in_rebuild": i, "scenario_id": pre[i]["scenario_id"], "level": pre[i]["level"],
                     "ps": pre[i]["prompt_setting"], "turn": pre[i].get("turn", "-"),
                     "hex32plus": sorted({m for msg in pre[i]["prompt"] for m in HEX.findall(msg["content"])})[:3],
                     "has_hex32plus": any(HEX.search(m["content"]) for m in pre[i]["prompt"] + (pre[i].get("completion") or pre[i]["chosen"]))}
                    for i in other]
    o["removed_by_scenario"] = dict(Counter(r["scenario_id"][:-1] for r in rem))
    o["removed_by_turn"] = dict(Counter(r.get("turn", "-") for r in rem))
    o["removed_by_level"] = dict(Counter(r["level"] for r in rem))
    o["removed_with_hex32plus"] = sum(x["has_hex32plus"] for x in o["removed"])
    o["removed_decision_renderings"] = sorted(f'{r["scenario_id"]}/{r["level"]}' for r in rem if r.get("turn", "decision") == "decision")
    # control: rows that KEPT a 32+ hex string
    o["kept_rows_with_hex32plus"] = sum(any(HEX.search(m["content"]) for m in r["prompt"]) for r in post)
    out[fn] = o
# GRPO
pre = load_jsonl(os.path.join(RB, "grpo_train.jsonl"))
post = cur("grpo_train")
pk = {json.dumps(r["prompt"], ensure_ascii=False) for r in post}
gone = [r for r in pre if json.dumps(r["prompt"], ensure_ascii=False) not in pk]
out["grpo_train"] = {"pre": len(pre), "post": len(post), "removed": sorted(f'{r["scenario_id"]}/{r["level"]}' for r in gone)}
# twin balance after curation (decision rows)
post = cur("sft_train")
out["sft_decision_twin_balance_post"] = dict(Counter(r["twin"][0] for r in post if r["turn"] == "decision"))
dump(out, "v7_curation.json")
print(json.dumps({k: (v if k != "sft_train" and k != "dpo_train" else {kk: vv for kk, vv in v.items() if kk != "removed"}) for k, v in out.items()}, indent=1))
