"""Integrity re-checks: (a) curated_v121 sha256 vs the trainers' recorded hashes; (b) dataset/eval files vs current specs
(test/val rows == spec rows; hard rows == scenarios_hard rows + held_split) and dataset/eval/SHA256SUMS; (c) max prompt tokens.
Output: results/v15_files_integrity.json
"""
import hashlib
import json
import os

from vcommon import CUR, ROOT, dump, eval_sets, hard_specs, specs


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


m1 = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m1_sft/args.json")))
m2 = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/args.json")))
local = {f: sha(os.path.join(CUR, f)) for f in ("sft_train.jsonl", "sft_val.jsonl", "dpo_train.jsonl", "grpo_train.jsonl")}


def jl_(f):
    return [json.loads(l) for l in open(os.path.join(CUR, f)) if l.strip()]


def rows_sha_sft(rows):     # = training/train_sft.py rows_sha
    return hashlib.sha256(json.dumps([[r["prompt"], r["completion"]] for r in rows], sort_keys=True).encode()).hexdigest()


def rows_sha_grpo(rows):    # = training/train_grpo.py data_sha256 (no --levels filter in m2_grpo/args.json)
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


A = {"local_file_sha256(not what trainers record)": local,
     "m1_train_rows_sha_match": rows_sha_sft(jl_("sft_train.jsonl")) == m1["train_sha256"],
     "m1_val_rows_sha_match": rows_sha_sft(jl_("sft_val.jsonl")) == m1["val_sha256"],
     "m2_grpo_rows_sha_match": rows_sha_grpo(jl_("grpo_train.jsonl")) == m2["data_sha256"],
     "m2_levels_arg": m2.get("levels")}

spec = {s["id"]: s for s in specs()}
hard = {s["id"]: s for s in hard_specs()}
diff = []
n = 0
for name, rows in eval_sets().items():
    for r in rows:
        n += 1
        if name.startswith("hard_"):
            ref = dict(hard[r["id"]])
            ref["held_split"] = r.get("held_split")
        else:
            ref = spec[r["id"]]
        if json.dumps(r, sort_keys=True) != json.dumps(ref, sort_keys=True):
            diff.append([name, r["id"]])
sums = {}
for line in open(os.path.join(ROOT, "dataset/eval/SHA256SUMS")):
    h, f = line.split()
    sums[f] = h == sha(os.path.join(ROOT, "dataset/eval", f))
B = {"eval_rows": n, "rows_differing_from_specs": diff, "SHA256SUMS_ok": sum(sums.values()), "SHA256SUMS_files": len(sums)}
mx = max(json.loads(l)["prompt_tokens"] for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))
C = {"max_prompt_tokens_any_training_row": mx, "plus_grpo_completion_budget_256": mx + 256}
dump("v15_files_integrity.json", {"a_training_files": A, "b_eval_files": B, "c_lengths": C})
print(A, B, C)
