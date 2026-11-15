"""V08 - extra checks: (a) M1 informed-card rate on twin A by env x level (is it the card rule or the clash type?);
(b) policy end vs the 4,096 window at L1; (c) approximate re-derivation of PC9's 'prescribed vs unspecified' twin-A contrast with my
own grouping; (d) DPO rejected sides that are prompt-permitted direct commits.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/08_extra.py -> results/08_extra.json
"""
import collections
import json
import os

from vcommon import OUT, ROOT, bootstrap_diff, dump

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
G = {(r["sid"], r["level"]): r for r in T if r["file"] == "grpo_train"}
by = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl")):
    s = json.loads(l)
    by[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"])
res = {}
# (a)
c = collections.defaultdict(lambda: [0, 0, 0])
for k, v in by.items():
    g = G[k]
    if g["t"] != "A":
        continue
    key = f"{g['env']}|{'L0r' if g['level'] == 'L0r' else 'L1+'}"
    c[key][0] += sum(d == "approve_informed" for d in v)
    c[key][1] += len(v)
    c[key][2] += 1
res["A_approve_informed_env_level"] = {k: {"approve_informed": a, "samples": n, "prompts": p, "rate": round(a / n, 3)}
                                       for k, (a, n, p) in sorted(c.items())}
# bills only: L1+ (BL card rule names renewal/terms) vs L0r (bills request_approval doc has no content rule)
bills = lambda lv: (lambda g: g["t"] == "A" and g["env"] == "bills" and (g["level"] == "L0r") == (lv == "L0r"))
res["bills_A_approve_informed_L1plus_vs_L0r"] = bootstrap_diff(
    {k: [int(d == "approve_informed") for d in v] for k, v in by.items() if bills("L1+")(G[k])},
    {k: [int(d == "approve_informed") for d in v] for k, v in by.items() if bills("L0r")(G[k])})
# (b) L1 policy end vs window (same estimator as 06: preamble 550 + 0.2496 tokens/char)
l1 = [r for r in T if r["file"] == "sft_train" and r["turn"] == "decision" and r["level"] == "L1"]
dist = [r["prompt_tokens"] - 550 - 0.24962 * r["policy_chars"] for r in l1]
res["L1_policy_end_to_generation"] = {"n": len(l1), "max": round(max(dist)), "beyond_4096": sum(d > 4096 for d in dist)}
# (c) approximate PC9 contrast: prescribed (P2 hint, or L1+ P0 blocked request / bills card question) vs unspecified (L0r P0)
NONEXEC = {"S15A", "V152A", "S57A", "V571A", "S11A", "V113A", "S10A"}
presc = lambda g: g["t"] == "A" and (g["ps"] == "P2" or (g["level"] != "L0r" and (g["sid"] in NONEXEC or g["sid"] in ("S32A", "V321A", "V322A"))))
unsp = lambda g: g["t"] == "A" and g["level"] == "L0r" and g["ps"] == "P0"
ok = {"ask_informed", "tell_informed"}
res["A_prescribed_vs_L0rP0_correct(approx PC9)"] = bootstrap_diff(
    {k: [int(d in ok) for d in v] for k, v in by.items() if presc(G[k])},
    {k: [int(d in ok) for d in v] for k, v in by.items() if unsp(G[k])})
res["A_P2_vs_P0_same_level_L0r_correct"] = bootstrap_diff(
    {k: [int(d in ok) for d in v] for k, v in by.items() if G[k]["t"] == "A" and G[k]["level"] == "L0r" and G[k]["ps"] == "P2"},
    {k: [int(d in ok) for d in v] for k, v in by.items() if unsp(G[k])})
# (d) DPO rejected direct commits on approve-expected rows whose prompt shows a covering authorisation (PC3 scenarios)
d = []
for r in T:
    if r["file"] == "dpo_train" and r["sid"] in ("S04B", "V042B", "V043B", "V044B", "V223B") and r["rejected"]["decision"].startswith("commit"):
        d.append({"line": r["i"], "sid": r["sid"], "level": r["level"], "ps": r["ps"], "rejected_class": r["rejected"]["decision"],
                  "rejected_reward": r["rejected"]["reward"], "rejected": r["rejected"]["text"][:220]})
res["dpo_rejected_commits_on_authorised_approve_rows"] = d
p = dump("08_extra.json", res)
print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
print("wrote", p)
