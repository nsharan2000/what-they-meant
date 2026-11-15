"""Do models notice the duplicated records of the realistic prompts? Scan first-decision chat messages on twin B (where a question
is needless) for duplicate talk ("twice", "duplicate", "same transaction ID", "listed two") by level, split by whether the
scenario has scenario_records already present in its member file (results/dup_records.json, current specs).
Outputs: results/dup_mentions.json
Run: python3 -B experiments/dataset_audit/realism-pressure/dup_mentions.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from common import RES, dump, iter_episode_files, short_model, twin_letter

DUP_RE = re.compile(r"\btwice\b|duplicat|listed two|two identical|same (transaction )?id|appears two times|shows up two", re.I)


def main():
    dup_ids = set()
    for g in json.load(open(os.path.join(RES, "dup_records.json")))["spec_level"].values():
        dup_ids |= set(g["ids"])
    agg = defaultdict(lambda: [0, 0])     # (level, dup_flag) -> [B messages, B messages with duplicate talk]
    ex = []
    for src, path in iter_episode_files():
        if src == "grid_open_seedtools_P0":
            continue
        with open(path) as fh:
            for i, l in enumerate(fh):
                r = json.loads(l)
                if twin_letter(r["twin"]) != "B":
                    continue
                msgs = [e for e in r.get("events", []) if e.get("kind") == "message" and not e.get("narration")]
                if not msgs:
                    continue
                txt = msgs[0].get("text", "")
                k = (r["level"], int(r["id"] in dup_ids))
                agg[k][0] += 1
                if DUP_RE.search(txt):
                    agg[k][1] += 1
                    if len(ex) < 5 and r["level"] in ("L1", "L2", "L2s") and r["id"] in dup_ids:
                        ex.append({"ref": f"{path}:{i}", "model": short_model(r["model"]), "id": r["id"], "level": r["level"],
                                   "prompt": r["prompt"], "decision": r["decision_grade"]["decision"], "text": txt[:240]})
    rows = [{"level": lv, "scenario_has_dup_records": bool(d), "B_first_messages": n, "with_duplicate_talk": k,
             "share": round(k / n, 4) if n else None} for (lv, d), (n, k) in sorted(agg.items())]
    dump({"rows": rows, "examples": ex}, "dup_mentions.json")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
