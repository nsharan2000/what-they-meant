"""Step 0: rebuild the PRE-curation v1.2.1 views (training/build_datasets.py, same args as dataset/curated_v121/build_report.json)
into cache/rebuild_v121 (git-ignored by the repo's **/cache/ rule), check it is the build that was curated, and attribute every
row CuratorKIT removed (exact dedup vs SecretsGate) so missing twin counterparts can be explained.

python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s0_rebuild.py
-> results/s0_rebuild.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, CUR, FILES, RES, ROOT, jdump, load_rows, sha256_file, twin_letter  # noqa: E402

OUT = os.path.join(CACHE, "rebuild_v121")


def h(*parts):
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def key(r):
    tgt = r.get("completion") or r.get("chosen")
    rej = r.get("rejected")
    return h(json.dumps(r["prompt"], ensure_ascii=False), json.dumps(tgt, ensure_ascii=False) if tgt else "",
             json.dumps(rej, ensure_ascii=False) if rej else "")


def main():
    if not os.path.exists(os.path.join(OUT, "build_report.json")):
        cmd = [sys.executable, "-B", "training/build_datasets.py", "--scenarios", "dataset/scenarios/*.jsonl",
               "dataset/scenarios_synth/*.jsonl", "--splits", "dataset/splits.json", "--failures",
               "experiments/replication/results/grid_*.jsonl", "--out", OUT]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        print(p.stdout[-2000:], p.stderr[-2000:])
        if p.returncode:
            raise SystemExit("rebuild failed")
    rep_new = json.load(open(os.path.join(OUT, "build_report.json")))
    rep_old = json.load(open(os.path.join(CUR, "build_report.json")))
    out = {"rebuild_dir": OUT, "counts_rebuild": rep_new["counts"], "counts_reported": rep_old["counts"],
           "counts_match": rep_new["counts"] == rep_old["counts"],
           "dropped_match": rep_new["dropped"] == rep_old["dropped"], "sha256": {}}
    for f in FILES:
        out["sha256"][f] = sha256_file(os.path.join(OUT, f + ".jsonl"))
    out["sha256_curated_sft_val_equals_rebuild"] = sha256_file(os.path.join(CUR, "sft_val.jsonl")) == out["sha256"]["sft_val"]

    # map curated rows back to rebuild rows; classify every rebuild row that is not in the curated view
    attrib = {}
    for f in FILES:
        pre = load_rows(f, OUT)
        cur = load_rows(f, CUR)
        cur_keys = Counter(key(r) for r in cur)
        pre_keys = Counter(key(r) for r in pre)
        missing_from_pre = sum(1 for k in cur_keys if k not in pre_keys)
        by_content = defaultdict(list)   # SFT dedup is on (prompt, target) content
        for r in pre:
            by_content[key(r)].append(r)
        dropped = []
        used = Counter()
        seen_pre = Counter()
        for r in pre:
            k = key(r)
            seen_pre[k] += 1
            if used[k] < cur_keys.get(k, 0):           # this copy survives (curated keeps the first occurrence)
                used[k] += 1
                continue
            if k in cur_keys:                          # an exact duplicate of a kept row
                why = "curator_exact_duplicate"
                twin_of_kept = twin_letter(by_content[k][0]["twin"])
            elif seen_pre[k] > 1:                      # duplicate of a row that the SecretsGate removed
                why = "curator_exact_duplicate_of_secrets_gated_row"
                twin_of_kept = None
            else:
                why = "grpo_prompt_not_in_curated_dpo" if f == "grpo_train" else "curator_secrets_gate"
                twin_of_kept = None
            dropped.append({"scenario_id": r["scenario_id"], "pair_id": r["pair_id"], "twin": twin_letter(r["twin"]),
                            "level": r["level"], "prompt_setting": r["prompt_setting"], "turn": r["turn"], "reason": why,
                            "kept_duplicate_twin": twin_of_kept})
        attrib[f] = {"n_pre": len(pre), "n_curated": len(cur), "curated_rows_not_in_rebuild": missing_from_pre,
                     "dropped": dropped,
                     "dropped_by_reason_twin": dict(Counter(f"{d['reason']}|{d['twin']}|{d['turn']}" for d in dropped))}
    out["curation_attribution"] = attrib
    jdump(out, os.path.join(RES, "s0_rebuild.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "curation_attribution"}, indent=1))
    for f in FILES:
        a = attrib[f]
        print(f, a["n_pre"], "->", a["n_curated"], "not-in-rebuild:", a["curated_rows_not_in_rebuild"], a["dropped_by_reason_twin"])


if __name__ == "__main__":
    main()
