"""Resolution of the null results (P2 effect, prompt-length effect, request-shared effect): 95 % permutation interval of the
difference in per-prompt M1 accuracy under the null, from results/v2_grpo_prompts.csv + results/v3_dupes.json inputs.
Output: results/v15_power.json"""
import json, os
from collections import defaultdict
import numpy as np, pandas as pd
from vcommon import RES, ROOT, cur, dump, request, load_jsonl
rng = np.random.default_rng(3)
pp = pd.read_csv(os.path.join(RES, "v2_grpo_prompts.csv"))
ts = {t["i"]: t["prompt_tokens"] for t in load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")) if t["file"] == "grpo_train.jsonl"}
grpo = cur("grpo_train"); sft = cur("sft_train")
req_twins = defaultdict(set)
for r in sft:
    if r["turn"] == "decision":
        req_twins[(r["pair_id"], request(r))].add(r["twin"][0])
pp["long"] = pp.grpo_line.map(ts) >= 4096
pp["shared"] = pp.grpo_line.map(lambda i: len(req_twins[(grpo[i]["pair_id"], request(grpo[i]))]) == 2)
pp["p2"] = pp.ps == "P2"
acc = pp.acc.to_numpy()
out = {"per_prompt_acc_sd": round(float(acc.std(ddof=1)), 4)}
for col in ("p2", "long", "shared"):
    g = pp[col].to_numpy().copy()
    obs = acc[g].mean() - acc[~g].mean()
    null = []
    for _ in range(10000):
        rng.shuffle(g)
        null.append(acc[g].mean() - acc[~g].mean())
    null = np.array(null)
    out[col] = {"n_true": int(pp[col].sum()), "n_false": int((~pp[col]).sum()), "diff": round(float(obs), 4),
                "null_95": [round(float(np.quantile(null, 0.025)), 4), round(float(np.quantile(null, 0.975)), 4)],
                "p_two_sided": round(float((np.abs(null) >= abs(obs) - 1e-12).mean()), 4)}
dump(out, "v15_power.json"); print(json.dumps(out, indent=1))
