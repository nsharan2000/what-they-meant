"""V03 - PC2 (direct-commit targets vs the P2 hint's 'just show the approval card'; V152B) and PC3 (authorisation-covered B rows
whose target is a card while GRPO scores the authorised commit -0.5). Also: which authorisation text each B row's prompt shows,
per level (my own phrase search over the member's standing instructions / notes / request), and M1 samples on these prompts.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/03_pc2_pc3.py -> results/03_pc2_pc3.json
"""
import collections
import json
import os
import re

from vcommon import OUT, ROOT, bootstrap_diff, dump, envs, reward, rows, specs

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
SP = specs()
E = envs()
RAW = {f: rows(f) for f in ("sft_train", "sft_val", "dpo_train", "grpo_train")}
ADJ = json.load(open(os.path.join(ROOT, "dataset/label_adjudications.json")))["rows"]
res = {}

# ---------------- PC2: expected-'act' rows, by file / level / setting ----------------
for f in ("sft_train", "sft_val", "dpo_train", "grpo_train"):
    act = [r for r in T if r["file"] == f and r["turn"] == "decision" and r["t"] == "B" and r["expected"] == "act"]
    res.setdefault("act_rows", {})[f] = {"n": len(act), "by_sid": dict(collections.Counter(r["sid"] for r in act)),
                                         "P2": sorted(f"{r['sid']}|{r['level']}|{r['ps']}|line {r['i']}" for r in act if r["ps"] == "P2")}
# reward of the hint-compliant card vs the commit on act rows (use the spec's ideal summary)
ex = {}
for r in [x for x in T if x["file"] == "grpo_train" and x["t"] == "B" and x["expected"] == "act"]:
    scn = SP[r["sid"]]
    kinds = {t["name"]: t["kind"] for t in E[r["env"]]["tools"]}
    card = '<tool_call>{"name": "request_approval", "arguments": {"summary": "%s"}}</tool_call>' % scn["expected"]["ideal_message"]
    ia = scn["ideal_action"]
    commit = '<tool_call>' + json.dumps({"name": ia["name"], "arguments": ia["arguments"]}) + '</tool_call>'
    ex[r["sid"]] = {"card_reward": reward(card, scn, kinds)[0], "commit_reward": reward(commit, scn, kinds)[0]}
res["act_reward_card_vs_commit"] = ex

# ---------------- authorisation text visible per B row (my own phrases, read from member files / requests) ----------------
AUTH = {   # scenario -> regex of the authorisation that covers the request (checked by reading the L1 prompts)
    "S04B": r"no approval needed below (?:INR|₹) ?1,500",
    "V042B": r"no approval needed below GBP 150|approval_needed_below_gbp\"?: ?150",
    "V043B": r"auto-approved under \$75",
    "V044B": r"no approval needed below USD 100",
    "V223B": r"pre-approved up to \$250",
    "S14B": r"charge or retry the card on file for a shipped order, up to USD 250 per order, without an approval card",
    "S15B": r"schedule overnight jobs without an approval card",
    "S10B": r"buy myproject\.dev if it is under USD 20",
    "V104B": r"renew my XMP Pro license if it's still under USD 90",
    "V152B": r"monthly software subscriptions under \$1,500 may be paid without approval",
}
vis = collections.defaultdict(dict)
for f in ("sft_train",):
    for i, x in enumerate(RAW[f]):
        if x.get("turn") != "decision" or x["scenario_id"] not in AUTH:
            continue
        full = "\n".join(m["content"] for m in x["prompt"])
        vis[x["scenario_id"]][f"{x['level']}|{x['prompt_setting']}|{i}"] = bool(re.search(AUTH[x["scenario_id"]], full))
res["authorisation_visible_sft_decision"] = vis

# PC3: approve-expected B rows whose prompt shows a covering authorisation
pc3 = []
for i, x in enumerate(RAW["sft_train"]):
    if x.get("turn") == "decision" and x["scenario_id"] in ("S04B", "V042B", "V043B", "V044B", "V223B"):
        full = "\n".join(m["content"] for m in x["prompt"])
        pc3.append({"line": i, "sid": x["scenario_id"], "level": x["level"], "ps": x["prompt_setting"],
                    "auth_visible": bool(re.search(AUTH[x["scenario_id"]], full)), "target": x["completion"][0]["content"][:160],
                    "adjudication_accept": ADJ.get(x["scenario_id"], {}).get("accept")})
res["pc3_rows"] = {"n": len(pc3), "auth_visible": sum(p["auth_visible"] for p in pc3), "rows": pc3}
# policy wording that permits the commit at L1+ (shopping/bills)
res["pc3_policy_quotes_present"] = {
    "shopping": sum("can be completed straight away without an approval card" in x["prompt"][0]["content"]
                    for x in RAW["sft_train"] if x["env_id"] == "shopping" and x["level"] != "L0r"),
    "bills": sum("A standing authorisation lets you act without a card" in x["prompt"][0]["content"]
                 for x in RAW["sft_train"] if x["env_id"] == "bills" and x["level"] != "L0r")}
# GRPO: commit_gold on these approve-expected rows
g = [r for r in T if r["file"] == "grpo_train" and r["sid"] in ("S04B", "V042B", "V043B", "V044B", "V223B")]
gg = {}
for r in g:
    scn = SP[r["sid"]]
    kinds = {t["name"]: t["kind"] for t in E[r["env"]]["tools"]}
    gc = scn.get("gold_commits") or []
    gg[r["sid"]] = {"expected": scn["expected"]["decision"], "gold_commits": gc}
res["pc3_grpo_rows"] = {"n": len(g), "specs": gg}

# ---------------- V152B: read the prompt ----------------
v = [x for x in RAW["sft_train"] if x["scenario_id"] == "V152B" and x.get("turn") == "decision"]
v152 = []
for x in v:
    s = x["prompt"][0]["content"]
    m = re.search(r"### Standing instructions\n(.*?)(?:\n\n|$)", s, re.S)
    n = re.search(r"### Notes on file\n(.*?)(?:\n\n|$)", s, re.S)
    v152.append({"level": x["level"], "ps": x["prompt_setting"], "user": x["prompt"][1]["content"],
                 "standing_instructions": m.group(1) if m else None, "notes": n.group(1) if n else None,
                 "lookup_result": x["prompt"][3]["content"] if len(x["prompt"]) > 3 else None,
                 "target": x["completion"][0]["content"]})
res["V152B"] = v152

# ---------------- M1 samples on act rows and on PC3 rows ----------------
G = {(r["sid"], r["level"]): r for r in T if r["file"] == "grpo_train"}
by = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl")):
    s = json.loads(l)
    by[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"])
res["m1_act_rows"] = {f"{k[0]}|{k[1]}|{G[k]['ps']}": dict(collections.Counter(v)) for k, v in by.items()
                      if G[k]["t"] == "B" and G[k]["expected"] == "act"}
res["m1_pc3_rows"] = {f"{k[0]}|{k[1]}|{G[k]['ps']}": dict(collections.Counter(v)) for k, v in by.items()
                      if k[0] in ("S04B", "V042B", "V043B", "V044B", "V223B")}
p = dump("03_pc2_pc3.json", res)
print(json.dumps({k: v for k, v in res.items() if k not in ("pc3_rows",)}, indent=1, ensure_ascii=False)[:9000])
print("wrote", p)
