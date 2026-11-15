"""Lead audit: the flagged decision rows to exclude from any NEW training run (REPORT.md, approval item Q-A/Q-B).

Each rule comes from a verified lens finding (paths in REPORT.md section 2, problem 6):
  zero_lookup        S02A/B, S03A/B decision rows: no look-up before the decision (prompt-conflicts PC4, verified).
  p2_hint_vs_act     expected 'act' rendered with P2 ('...just show the approval card') (PC2 / SEA-3 / DPO-7, verified).
  l0r_act_no_auth    expected 'act' at L0r where no authorisation is visible: S14B, S15B, V152B (PC verifier, missed item).
  v152b_mislabel     V152B targets schedule_job (an invoice fetch) for a request to pay a subscription (PC2 verifier).
  authorised_card    S04B, V042B, V043B, V044B, V223B: card targets where the prompt authorises a direct commit that the
                     reward/eval grade as wrong because gold_commits is empty (PC3 / GA-5 / DPO-6, verified).
  v541b_world        V541B pays Ethan Price for lawn mowing although the member world says Ethan Cole mows (SEA-9, verified).
Reads dataset/curated_v121/{sft_train,dpo_train,grpo_train}.jsonl (read-only); writes lead/results/exclusion_list.json.
Run: python3 experiments/dataset_audit/lead/exclusion_list.py   (stdlib only)
"""
import json, os
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(REPO, "experiments", "dataset_audit")
DATA = os.path.join(REPO, "dataset", "curated_v121")


def scenario_of(row):
    if "scenario" in row:  # grpo rows embed the spec as a JSON string
        return json.loads(row["scenario"])
    return None


def rules(row, spec_by_id):
    sid, lvl, ps = row["scenario_id"], row["level"], row["prompt_setting"]
    exp = spec_by_id.get(sid, {}).get("expected", {}).get("decision")
    hits = []
    if sid in ("S02A", "S02B", "S03A", "S03B"):
        hits.append("zero_lookup")
    if exp == "act" and ps == "P2":
        hits.append("p2_hint_vs_act")
    if sid in ("S14B", "S15B", "V152B") and lvl == "L0r":
        hits.append("l0r_act_no_auth")
    if sid == "V152B":
        hits.append("v152b_mislabel")
    if sid in ("S04B", "V042B", "V043B", "V044B", "V223B"):
        hits.append("authorised_card")
    if sid == "V541B":
        hits.append("v541b_world")
    return hits


# expected decisions come from the specs embedded in grpo_train (identical to the current spec files per the
# grader-agreement and splits-eval verifiers: 0 drift)
spec_by_id = {}
for line in open(os.path.join(DATA, "grpo_train.jsonl")):
    r = json.loads(line)
    spec_by_id[r["scenario_id"]] = scenario_of(r)

out = {}
for name in ("sft_train", "dpo_train", "grpo_train"):
    rows = []
    for i, line in enumerate(open(os.path.join(DATA, f"{name}.jsonl"))):
        r = json.loads(line)
        if name == "sft_train" and r.get("turn") != "decision":
            continue
        hits = rules(r, spec_by_id)
        if hits:
            rows.append({"line": i, "scenario_id": r["scenario_id"], "level": r["level"],
                         "prompt_setting": r["prompt_setting"], "rules": hits})
    by_rule = Counter(h for x in rows for h in x["rules"])
    out[name] = {"n_rows": len(rows), "by_rule": dict(by_rule),
                 "scenarios": sorted({x["scenario_id"] for x in rows}), "rows": rows}
    print(f"{name}: {len(rows)} flagged decision rows; by rule {dict(by_rule)}")

act_ids = sorted(k for k, v in spec_by_id.items() if v and v.get("expected", {}).get("decision") == "act")
out["act_scenarios_in_grpo"] = act_ids
json.dump(out, open(os.path.join(ROOT, "lead", "results", "exclusion_list.json"), "w"), indent=1)
print("act scenarios:", act_ids)
