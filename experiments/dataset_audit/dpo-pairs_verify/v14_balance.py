"""Verifier step 14: nominal balance of dpo_train.jsonl (twin x level, twin x env, prompt setting). Output: results/v14_balance.json"""
import json, os
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
d = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))]
res = {"twin": dict(Counter(r["twin"][:1] for r in d)),
       "twin_x_level": {f"{t}:{l}": n for (t, l), n in sorted(Counter((r["twin"][:1], r["level"]) for r in d).items())},
       "twin_x_env": {f"{t}:{e}": n for (t, e), n in sorted(Counter((r["twin"][:1], r["env_id"]) for r in d).items())},
       "prompt_setting": dict(Counter(r["prompt_setting"] for r in d))}
json.dump(res, open(os.path.join(HERE, "results", "v14_balance.json"), "w"), indent=1)
print(json.dumps(res))
