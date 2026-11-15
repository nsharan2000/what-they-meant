"""CM-2: expected share of rows in optimizer step s whose exact target already occurred in an EARLIER optimizer step of the
same epoch, under a uniformly random row order (HF Trainer RandomSampler), 16 rows per step, 2,000 permutations.
Also the same for decision rows only (target + first-token class). Output: results/v12_order.json"""
import json
import numpy as np
from vcommon import cur, dump, target
rng = np.random.default_rng(0)
tr = cur("sft_train")
ids = {}
t = np.array([ids.setdefault(target(r), len(ids)) for r in tr])
dec = np.array([r["turn"] == "decision" for r in tr])
B, n = 16, len(tr)
steps = int(np.ceil(n / B))
acc = np.zeros(steps); acc_dec = np.zeros(steps); cnt_dec = np.zeros(steps)
R = 2000
for _ in range(R):
    order = rng.permutation(n)
    seen = np.zeros(len(ids), bool)
    for s in range(steps):
        idx = order[s * B:(s + 1) * B]
        rep = seen[t[idx]]
        acc[s] += rep.mean()
        d = dec[idx]
        if d.any():
            acc_dec[s] += rep[d].sum(); cnt_dec[s] += d.sum()
        seen[t[idx]] = True
acc /= R
out = {"steps_per_epoch": steps, "repeat_share_at_step": {str(s): round(float(acc[s - 1]), 3) for s in (1, 2, 5, 10, 20, 29, 40, 48, 57)},
       "decision_rows_repeat_share_at_step": {str(s): round(float(acc_dec[s - 1] / cnt_dec[s - 1]), 3) for s in (5, 10, 20, 29, 48, 57)},
       "mean_repeat_share_epoch": round(float(acc.mean()), 3)}
dump(out, "v12_order.json"); print(json.dumps(out, indent=1))
