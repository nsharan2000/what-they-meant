"""VERIFY RP-8 spec-level counts with the analyst's groups: specs whose scenario_records already exist in the member file
(same id for dict records with an id, else exact equality). Groups: train pool = dataset/scenarios + dataset/scenarios_synth;
dataset/eval/test_*, hard_test_*, val_*, hard_val_*, hard_train_*.
Outputs: results/v17_spec_dups.json
"""
from __future__ import annotations

import glob
import json
import os

from vcommon import ROOT, dump, load_jsonl

GROUPS = {"train_pool": ["dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"],
          "test": ["dataset/eval/test_*.jsonl"], "hard_test": ["dataset/eval/hard_test_*.jsonl"],
          "val": ["dataset/eval/val_*.jsonl"], "hard_val": ["dataset/eval/hard_val_*.jsonl"],
          "hard_train": ["dataset/eval/hard_train_*.jsonl"]}


def main():
    members = {}
    out = {}
    for g, pats in GROUPS.items():
        n = d = 0
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(ROOT, pat))):
                for s in load_jsonl(f):
                    if "member_id" not in s:
                        continue
                    n += 1
                    k = (s["env_id"], s["member_id"])
                    if k not in members:
                        members[k] = json.load(open(os.path.join(ROOT, "envs", s["env_id"], "members", f"{s['member_id']}.json")))
                    m = members[k]
                    hit = False
                    for x in s.get("scenario_records") or []:
                        ex, rec = m.get(x["field"], []), x["record"]
                        if (isinstance(rec, dict) and rec.get("id") and
                                any(isinstance(e, dict) and e.get("id") == rec["id"] for e in ex)) or rec in ex:
                            hit = True
                    d += int(hit)
        out[g] = {"specs": n, "with_dup_scenario_records": d}
    dump(out, "v17_spec_dups.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
