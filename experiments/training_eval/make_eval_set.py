"""Write the evaluation sets from dataset/splits.json: dataset/eval/{test,val}_<env>.jsonl and hard_{test,val,train}_<env>.jsonl.

test = the 13 held-out seed clusters (seed pairs + pushback rows) and their synthetic variants (never in any training view);
val  = the 5 validation clusters (checkpoint selection only).
hard = the generated hard set (dataset/scenarios_hard), split by the cluster of the seed each pair was varied from: every row gets
       held_split = the split of its source cluster (source.post_id; variant_of must agree). Why: 27 of the 45 hard pairs are
       variants of TRAIN clusters, so only hard_test is held out; hard_train measures in-distribution variants of trained seed
       patterns, and hard_val must never drive checkpoint selection (training-pipeline review 2026-11-14, item 8).
Also writes SHA256SUMS over every *.jsonl in the output directory so every reported number names the exact spec version.
python3 experiments/training_eval/make_eval_set.py [--out dataset/eval]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from wtm.spec import ROOT, load_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "eval"))
ns = ap.parse_args()
splits = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
split_of = {pid: k for k in ("train", "val", "test") for pid in splits[k]}
rows = [r for p in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl")
        for f in sorted(glob.glob(os.path.join(ROOT, p))) for r in load_jsonl(f)]
out_dir = ns.out
os.makedirs(out_dir, exist_ok=True)
stats = Counter()


def write(name: str, rs: list[dict]) -> None:
    with open(os.path.join(out_dir, name), "w") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


for split in ("test", "val"):
    by_env = {}
    for r in rows:
        if split_of.get(r["pair_id"]) == split:
            by_env.setdefault(r["env_id"], []).append(r)
            stats[(split, "synthetic" if r.get("synthetic") else "seed")] += 1
    for env, rs in by_env.items():
        write(f"{split}_{env}.jsonl", rs)

hard = [r for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios_hard", "*.jsonl"))) for r in load_jsonl(f)]
by_split_env, hard_pairs, unresolved = {}, {}, []
for r in hard:
    by_post = splits["clusters"].get(r.get("source", {}).get("post_id"), {}).get("split")
    by_seed = split_of.get(r.get("variant_of"))
    if by_post is None or by_post != by_seed:
        unresolved.append((r["id"], by_post, by_seed))
        continue
    by_split_env.setdefault((by_post, r["env_id"]), []).append(r | {"held_split": by_post})
    hard_pairs.setdefault(by_post, set()).add(r["pair_id"])
if unresolved:
    raise SystemExit(f"EVAL_SET_FAIL: hard rows whose source cluster split is missing or ambiguous: {unresolved[:10]}")
for (split, env), rs in sorted(by_split_env.items()):
    write(f"hard_{split}_{env}.jsonl", rs)
    stats[(f"hard_{split}", "rows")] += len(rs)
for split, pids in hard_pairs.items():
    stats[(f"hard_{split}", "pairs")] = len(pids)

with open(os.path.join(out_dir, "SHA256SUMS"), "w") as fh:
    for f in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        fh.write(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {os.path.basename(f)}\n")
print(json.dumps({f"{a}/{b}": v for (a, b), v in sorted(stats.items())}))
