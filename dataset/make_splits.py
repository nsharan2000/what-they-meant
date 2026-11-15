"""Group-aware splits: a seed pair, its pushback variant and all its synthetic variants share one source-post cluster, and a
cluster lives in exactly one split (no incident pattern leaks from train to test). Stratified by environment; deterministic.

python3 dataset/make_splits.py [--test 0.3] [--val 0.1] → dataset/splits.json {"train": [pair_ids], "val": [...], "test": [...],
                                                                           "clusters": {...}, "policy": "..."}
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.spec import ROOT, load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=float, default=0.3)
    ap.add_argument("--val", type=float, default=0.1)
    ns = ap.parse_args()
    rows = [r for p in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl")
            for f in sorted(glob.glob(os.path.join(ROOT, p))) for r in load_jsonl(f)]
    cluster_of_pair, env_of_cluster = {}, {}
    for r in rows:
        c = r["source"].get("post_id") or r["pair_id"]
        cluster_of_pair[r["pair_id"]] = c
        env_of_cluster[c] = r["env_id"]
    by_env = defaultdict(list)
    for c, e in env_of_cluster.items():
        by_env[e].append(c)
    split_of = {}
    for e, cs in by_env.items():
        cs = sorted(cs, key=lambda c: hashlib.sha256(f"wtm-split-{c}".encode()).hexdigest())
        n_test = max(1, round(ns.test * len(cs)))
        n_val = max(1, round(ns.val * len(cs))) if len(cs) >= 6 else 0
        for i, c in enumerate(cs):
            split_of[c] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    out = {"train": [], "val": [], "test": []}
    for pid, c in sorted(cluster_of_pair.items()):
        out[split_of[c]].append(pid)
    out["clusters"] = {c: {"env": env_of_cluster[c], "split": split_of[c],
                           "pairs": sorted(p for p, cc in cluster_of_pair.items() if cc == c)} for c in sorted(split_of)}
    out["policy"] = ("clusters = source posts; a seed pair, its pushback variant and its synthetic variants share a cluster; "
                     f"per-environment deterministic hash split, test≈{ns.test}, val≈{ns.val}")
    json.dump(out, open(os.path.join(ROOT, "dataset", "splits.json"), "w"), indent=1)
    stat = {k: len(v) for k, v in out.items() if isinstance(v, list)}
    stat["clusters"] = {s: sum(1 for c in split_of.values() if c == s) for s in ("train", "val", "test")}
    stat["seed_pairs"] = {s: sum(1 for p in out[s] if p.startswith("S")) for s in ("train", "val", "test")}
    print(json.dumps(stat))


if __name__ == "__main__":
    main()
