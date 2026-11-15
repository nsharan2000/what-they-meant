"""Verifier step 5: rebuild v1.2.1 before curation (training/build_datasets.py, output only under this folder) and compare.

Checks: (a) the rebuilt dpo_train has the build_report counts (508 pairs); (b) which rebuilt pairs CuratorKIT dropped (the
curated file keeps 496), by env / level / twin / scenario; whether the dropped prompts contain hex/base58 wallet-like strings;
(c) the 496 kept rows equal their rebuilt rows exactly; (d) DPO is a subset of SFT (same prompt, chosen == SFT completion);
(e) DPO overlap with sft_val and the evaluation scenario files.
Output: results/v5_rebuild_compare.json (rebuild in rebuild_v121/)
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
REB = os.path.join(HERE, "rebuild_v121")

if not os.path.exists(os.path.join(REB, "dpo_train.jsonl")):
    cmd = [sys.executable, "training/build_datasets.py", "--scenarios", "dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl",
           "--splits", "dataset/splits.json", "--failures", "experiments/replication/results/grid_*.jsonl", "--out", REB]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(p.stdout[-2000:], p.stderr[-2000:])
    assert p.returncode == 0


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return json.dumps([r["prompt"], (r.get("chosen") or r.get("completion")), r.get("rejected")], ensure_ascii=False, sort_keys=True)


cur = jl(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))
reb = jl(os.path.join(REB, "dpo_train.jsonl"))
rep_orig = json.load(open(os.path.join(ROOT, "dataset/curated_v121/build_report.json")))
rep_new = json.load(open(os.path.join(REB, "build_report.json")))
cur_keys = Counter(key(r) for r in cur)
reb_keys = {key(r): r for r in reb}
kept = [r for r in reb if key(r) in cur_keys]
dropped = [r for r in reb if key(r) not in cur_keys]
cur_not_in_reb = [r for r in cur if key(r) not in reb_keys]

HEX = re.compile(r"\b0x[0-9a-fA-F]{20,}\b")
B58 = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{26,}\b")


def secrets_like(r):
    txt = "\n".join(m["content"] for m in r["prompt"]) + r["chosen"][0]["content"] + r["rejected"][0]["content"]
    return {"hex": len(HEX.findall(txt)), "b58_like": len([x for x in B58.findall(txt) if re.search(r"\d", x) and re.search(r"[A-Za-z]", x)])}


sft_tr = jl(os.path.join(ROOT, "dataset/curated_v121/sft_train.jsonl"))
sft_va = jl(os.path.join(ROOT, "dataset/curated_v121/sft_val.jsonl"))
sft_map = {}
for i, r in enumerate(sft_tr):
    if r.get("turn") == "decision":
        sft_map[json.dumps(r["prompt"], ensure_ascii=False)] = (i, r["completion"][0]["content"])
val_prompts = {json.dumps(r["prompt"], ensure_ascii=False) for r in sft_va}
sub = [sft_map.get(json.dumps(r["prompt"], ensure_ascii=False)) for r in cur]
eval_ids = set()
for f in glob.glob(os.path.join(ROOT, "dataset/eval/*.jsonl")):
    for s in jl(f):
        eval_ids.add(s["id"])
        eval_ids.add(s.get("pair_id"))
splits = json.load(open(os.path.join(ROOT, "dataset/splits.json")))
pair_split = {p: k for k in ("train", "val", "test") for p in splits[k]}
chosen_texts = Counter(r["chosen"][0]["content"] for r in cur)
rej_texts = Counter(r["rejected"][0]["content"] for r in cur)
all_sft_targets = {r["completion"][0]["content"] for r in sft_tr + sft_va}

res = {
    "rebuilt_counts": rep_new["counts"], "original_report_counts": rep_orig["counts"],
    "rebuilt_dpo_rejected_source": rep_new["dpo_rejected_source"], "original_dpo_rejected_source": rep_orig["dpo_rejected_source"],
    "curated_rows": len(cur), "rebuilt_rows": len(reb), "kept_exact": len(kept), "dropped": len(dropped),
    "curated_rows_not_in_rebuild": len(cur_not_in_reb),
    "dropped_rows": sorted([[r["scenario_id"], r["level"], r["prompt_setting"], r["env_id"], secrets_like(r)] for r in dropped]),
    "dropped_by_env": dict(Counter(r["env_id"] for r in dropped)), "dropped_by_level": dict(Counter(r["level"] for r in dropped)),
    "dropped_by_twin": dict(Counter(r["twin"][:1] for r in dropped)),
    "kept_wallet_rows_with_hex_or_b58": sum(1 for r in kept if r["env_id"] == "wallet" and (secrets_like(r)["hex"] or secrets_like(r)["b58_like"])),
    "kept_rows_by_level": dict(Counter(r["level"] for r in kept)),
    "subset_of_sft_decision_rows": sum(1 for x in sub if x is not None),
    "chosen_equals_sft_completion": sum(1 for x, r in zip(sub, cur) if x and x[1] == r["chosen"][0]["content"]),
    "examples_sft_line": [[i, r["scenario_id"], r["level"], sub[i][0] if sub[i] else None] for i, r in enumerate(cur[:3])],
    "dpo_prompt_in_sft_val": sum(1 for r in cur if json.dumps(r["prompt"], ensure_ascii=False) in val_prompts),
    "dpo_pair_split": dict(Counter(pair_split.get(r["pair_id"]) for r in cur)),
    "dpo_pairs_in_eval_files": sorted({r["pair_id"] for r in cur if r["pair_id"] in eval_ids or r["scenario_id"] in eval_ids}),
    "text_both_chosen_and_rejected": sum(1 for t in rej_texts if t in chosen_texts),
    "rejected_equals_some_sft_target": sum(1 for t in rej_texts if t in all_sft_targets),
}
json.dump(res, open(os.path.join(OUT, "v5_rebuild_compare.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
