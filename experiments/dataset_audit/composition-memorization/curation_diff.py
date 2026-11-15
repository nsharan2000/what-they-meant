"""Which SFT rows did CuratorKIT remove (1,000 built -> 904 trained)? The pre-curation build is regenerated deterministically into
rebuild_v121/ (training/build_datasets.py, same args as dataset/curated_v121/build_report.json; byte-identical to the build the
curation ran on, checked by sha256 in the log of this audit). Rows are matched on (prompt, completion) JSON.

    python3 -I training/build_datasets.py --scenarios 'dataset/scenarios/*.jsonl' 'dataset/scenarios_synth/*.jsonl' \
        --splits dataset/splits.json --failures 'experiments/replication/results/grid_*.jsonl' \
        --out experiments/dataset_audit/composition-memorization/rebuild_v121
    python3 experiments/dataset_audit/composition-memorization/curation_diff.py
Output: results/curation_diff.json
"""
from __future__ import annotations

import collections
import json
import os
import re

from common import DATA, dump, load_jsonl, source_of, target_of

HERE = os.path.dirname(os.path.abspath(__file__))
HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{16,}\b")
B64_RE = re.compile(r"[A-Za-z0-9+/=_-]{20,}")


def key(r):
    return json.dumps([r["prompt"], r["completion"]], sort_keys=True, ensure_ascii=False)


def main():
    out = {}
    for view in ("sft_train", "dpo_train"):
        pre = load_jsonl(os.path.join(HERE, "rebuild_v121", f"{view}.jsonl"))
        post = load_jsonl(os.path.join(DATA, f"{view}.jsonl"))
        kf = (lambda r: json.dumps([r["prompt"], r["chosen"], r["rejected"]], sort_keys=True, ensure_ascii=False)) \
            if view == "dpo_train" else key
        post_keys = collections.Counter(kf(r) for r in post)
        pre_keys = collections.Counter(kf(r) for r in pre)
        seen, dups, removed = collections.Counter(), [], []
        for i, r in enumerate(pre):
            k = kf(r)
            seen[k] += 1
            if seen[k] > 1:
                dups.append((i, r))
            elif post_keys[k] == 0:
                removed.append((i, r))

        def desc(i, r):
            blob = "\n".join(m["content"] for m in r["prompt"]) + "\n" + target_of(r)
            return {"pre_row": i, "scenario_id": r["scenario_id"], "level": r["level"], "ps": r["prompt_setting"],
                    "turn": r.get("turn", "-"), "twin": r["twin"], "source": source_of(r["scenario_id"]),
                    "hex16_in_row": sorted(set(HEX_RE.findall(blob)))[:3]}
        out[view] = {"pre": len(pre), "post": len(post), "distinct_pre": len(pre_keys),
                     "exact_duplicates": len(dups), "removed_not_duplicate": len(removed),
                     "dups_by_turn": dict(collections.Counter(r.get("turn", "-") for _, r in dups)),
                     "dups_scenarios": sorted({r["scenario_id"] for _, r in dups}),
                     "removed_by_turn": dict(collections.Counter(r.get("turn", "-") for _, r in removed)),
                     "removed_by_scenario": dict(collections.Counter(r["scenario_id"] for _, r in removed).most_common()),
                     "removed_by_pair": dict(collections.Counter(r["pair_id"] for _, r in removed).most_common()),
                     "removed_rows": [desc(i, r) for i, r in removed],
                     "dup_examples": [desc(i, r) for i, r in dups[:5]]}
        # do whole decision renderings vanish? (scenario, level) present pre but with no decision row post
        if view == "sft_train":
            pre_dec = {(r["scenario_id"], r["level"]) for r in pre if r["turn"] == "decision"}
            post_dec = {(r["scenario_id"], r["level"]) for r in post if r["turn"] == "decision"}
            lost = sorted(pre_dec - post_dec)
            out[view]["decision_renderings_lost"] = len(lost)
            out[view]["decision_renderings_lost_list"] = lost
            pre_pairs = collections.Counter(r["pair_id"] for r in pre)
            post_pairs = collections.Counter(r["pair_id"] for r in post)
            out[view]["pairs_losing_rows"] = {p: [pre_pairs[p], post_pairs[p]] for p in pre_pairs if post_pairs[p] < pre_pairs[p]}
            # twin balance of decision renderings after curation: pair-levels with only one twin's decision row
            post_tw = collections.defaultdict(set)
            for r in post:
                if r["turn"] == "decision" and r["twin"] in ("A_should_ask", "B_should_act"):
                    post_tw[(r["pair_id"], r["level"])].add(r["twin"][:1])
            out[view]["pair_levels_with_one_twin_only"] = sorted([list(k) + ["".join(sorted(v))] for k, v in post_tw.items()
                                                                  if len(v) == 1])
    # provenance: hashes of the regenerated pre-curation files vs the build CuratorKIT consumed (build_report.json args.out, if it
    # still exists on this machine) and the curated val (copied unchanged); CuratorKIT's own stage counts for comparison
    import hashlib

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    rep = json.load(open(os.path.join(DATA, "build_report.json")))
    orig_out = rep["args"]["out"]
    out["provenance"] = {
        "rebuild_sft_train_sha256": sha(os.path.join(HERE, "rebuild_v121", "sft_train.jsonl")),
        "original_build_sft_train_sha256": sha(os.path.join(orig_out, "sft_train.jsonl")),
        "original_build_path": orig_out,
        "rebuild_sft_val_sha256": sha(os.path.join(HERE, "rebuild_v121", "sft_val.jsonl")),
        "curated_sft_val_sha256": sha(os.path.join(DATA, "sft_val.jsonl")),
        "rebuild_build_report_counts": json.load(open(os.path.join(HERE, "rebuild_v121", "build_report.json")))["counts"],
        "curated_build_report_counts": rep["counts"]}
    out["provenance"]["sft_train_identical"] = (out["provenance"]["rebuild_sft_train_sha256"]
                                                == out["provenance"]["original_build_sft_train_sha256"])
    cur = json.load(open(os.path.join(DATA, "curation_report.json")))
    out["curatorkit_stage_counts_sft"] = {"exact_duplicates_removed": cur["sft"]["manifest"]["dedup_stats"]["exact_duplicates_removed"],
                                          "secrets_rejected": cur["sft"]["manifest"]["stage_counts"]["SecretsGate"]["rejected_count"],
                                          "rejected_breakdown": cur["sft"]["manifest"]["rejected_breakdown"]}
    out["curatorkit_stage_counts_dpo"] = {"secrets_rejected": cur["dpo"]["manifest"]["stage_counts"]["SecretsGate"]["rejected_count"]}
    dump(out, "curation_diff.json")
    print(json.dumps(out["provenance"]), json.dumps(out["curatorkit_stage_counts_sft"]))
    for v in ("sft_train", "dpo_train"):
        print(v, {k: out[v][k] for k in ("pre", "post", "exact_duplicates", "removed_not_duplicate", "dups_by_turn",
                                         "removed_by_turn", "removed_by_pair")})
    print("lost decision renderings:", out["sft_train"]["decision_renderings_lost_list"])
    print("one-twin pair-levels:", out["sft_train"]["pair_levels_with_one_twin_only"])


if __name__ == "__main__":
    main()
