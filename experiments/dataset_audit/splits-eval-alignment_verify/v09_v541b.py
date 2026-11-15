"""SEA-9 verification: does training row V541B contradict the ruth_price member world used by held-out rows?

Lists: every ruth_price record mentioning lawn/mow or an Ethan; V541A/V541B request, look-up results (eval Simulator, L1),
scenario_records, expected decision, target; held-out ruth_price rows (val/test) with their ideal; M1 GRPO samples on V541A/B.
Output: results/v09_v541b.json
"""
import json
import os
import re
from collections import Counter

from vcommon import ROOT, curated, dump, episode_view, eval_sets, member, specs, splits, target_text

by_id = {s["id"]: s for s in specs()}
sp = splits()
pair_split = {p: k for k in ("train", "val", "test") for p in sp[k]}
m = member("wallet", "ruth_price")
rx = re.compile(r"lawn|mow|ethan", re.I)
recs = []
for key, val in m.items():
    if isinstance(val, list):
        for r in val:
            t = json.dumps(r, ensure_ascii=False)
            if rx.search(t):
                recs.append([key, t[:220]])
lawn_tx = [t for t in m.get("transactions", []) if re.search(r"lawn|mow", json.dumps(t), re.I)]
lawn_payees = Counter(t.get("counterparty") for t in lawn_tx)

out = {"ruth_price_records_mentioning_lawn_mow_ethan": recs, "lawn_transactions_n": len(lawn_tx),
       "lawn_transactions_by_counterparty": dict(lawn_payees)}
for sid in ("V541A", "V541B"):
    s = by_id[sid]
    system, user, tools, msgs = episode_view(s, "L1", "P0")
    out[sid] = {"split": pair_split.get(s["pair_id"]), "user_message": s["user_message"], "expected": s["expected"]["decision"],
                "scenario_records": s.get("scenario_records"), "lookup_results": [x["content"][:500] for x in msgs[2::2]],
                "ideal": s.get("ideal_action"), "label_audit": s.get("label_audit"),
                "sft_rows": [[i, r["level"], r["prompt_setting"], r["turn"]] for i, r in enumerate(curated("sft_train")) if r["scenario_id"] == sid]}
held = []
for name, rows in eval_sets().items():
    for s in rows:
        if s["member_id"] == "ruth_price":
            ia = s.get("ideal_action") or {}
            held.append([name, s["id"], s["user_message"], (ia.get("text") or json.dumps(ia.get("arguments")))[:220]])
out["heldout_ruth_price_rows"] = held
# training rows in the same member world stating who mows the lawn
tw = []
for i, r in enumerate(curated("sft_train")):
    t = target_text(r)
    if re.search(r"lawn|mow", t, re.I):
        tw.append([i, r["scenario_id"], r["level"], t[:160]])
out["sft_train_targets_mentioning_lawn"] = tw
sc = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
out["m1_samples_on_V541"] = {sid: dict(Counter(x["components"]["decision"] for x in sc if x["meta"]["scenario_id"] == sid)) for sid in ("V541A", "V541B")}
dump("v09_v541b.json", out)
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
