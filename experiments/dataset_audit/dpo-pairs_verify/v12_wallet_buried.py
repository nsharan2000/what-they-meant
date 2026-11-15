"""Verifier step 12: which wallet-env DPO rows remain at the buried levels (L2/L3) after SecretsGate, and which are crypto
transfers (request names a crypto asset or a wallet/deposit address)? Output: results/v12_wallet_buried.json"""
import json, os, re
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
dpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))]
reb = [json.loads(l) for l in open(os.path.join(HERE, "rebuild_v121/dpo_train.jsonl"))]
CRYPTO = re.compile(r"\b(ETH|BTC|USDC|USDT|PYUSD|SOL|crypto|deposit address|hardware wallet|wallet address)\b", re.I)
def table(rows):
    t = defaultdict(list)
    for r in rows:
        if r["env_id"] == "wallet" and r["level"] in ("L2", "L3"):
            req = r["prompt"][1]["content"]
            t[r["level"]].append([r["scenario_id"], bool(CRYPTO.search(req)), req[:70]])
    return {k: sorted(v) for k, v in sorted(t.items())}
kept, before = table(dpo), table(reb)
res = {"kept": kept, "before_curation": before,
       "crypto_rows_kept": {k: sum(1 for x in v if x[1]) for k, v in kept.items()},
       "crypto_rows_before": {k: sum(1 for x in v if x[1]) for k, v in before.items()}}
json.dump(res, open(os.path.join(HERE, "results", "v12_wallet_buried.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("crypto_rows_kept", "crypto_rows_before")}))
for lv, v in kept.items():
    print(lv, [x for x in v if x[1]])
for lv, v in before.items():
    print('before', lv, [x[0] for x in v if x[1]])
