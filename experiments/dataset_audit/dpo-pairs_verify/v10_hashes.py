"""Verifier step 10: are the local curated_v121 files the ones the M1/M2 runs trained on, and what pairs_sha256 will M3 record?
Recomputes the hashes exactly as training/train_sft.py (rows_sha), training/train_grpo.py (data_sha256) and training/train_dpo.py
(pairs_sha256) do, and compares with the args.json the runs saved. Output: results/v10_hashes.json
"""
import hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
D = os.path.join(ROOT, "dataset/curated_v121"); L = os.path.join(ROOT, "experiments/training_eval/results/train_logs")
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
sft_tr, sft_va, grpo, dpo = (jl(os.path.join(D, f)) for f in ("sft_train.jsonl", "sft_val.jsonl", "grpo_train.jsonl", "dpo_train.jsonl"))
rows_sha = lambda rows: hashlib.sha256(json.dumps([[r["prompt"], r["completion"]] for r in rows], sort_keys=True).encode()).hexdigest()
m1 = json.load(open(os.path.join(L, "m1_sft/args.json"))); m2 = json.load(open(os.path.join(L, "m2_grpo/args.json")))
res = {"sft_train_sha_local": rows_sha(sft_tr), "sft_train_sha_m1_args": m1["train_sha256"],
       "sft_val_sha_local": rows_sha(sft_va), "sft_val_sha_m1_args": m1["val_sha256"],
       "grpo_sha_local": hashlib.sha256(json.dumps(grpo, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
       "grpo_sha_m2_args": m2["data_sha256"],
       "dpo_pairs_sha256_local(train_dpo formula)": hashlib.sha256(json.dumps([[r["prompt"], r["chosen"], r["rejected"]] for r in dpo], sort_keys=True).encode()).hexdigest()}
res["sft_train_match"] = res["sft_train_sha_local"] == res["sft_train_sha_m1_args"]
res["sft_val_match"] = res["sft_val_sha_local"] == res["sft_val_sha_m1_args"]
res["grpo_match"] = res["grpo_sha_local"] == res["grpo_sha_m2_args"]
json.dump(res, open(os.path.join(HERE, "results", "v10_hashes.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
