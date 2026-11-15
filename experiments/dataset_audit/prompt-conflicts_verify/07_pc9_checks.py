"""V07 - PC9 (base Tiny Aya / M1 sensitivity to prompt content), PC8 mechanical counts, reward spot-checks used by PC1/PC4/PC10, and
re-checks of several 'checked_ok' claims.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/07_pc9_checks.py -> results/07_pc9_checks.json
"""
import collections
import json
import os
import re

from vcommon import OUT, ROOT, bootstrap_diff, dump, envs, jl, reward, rows, specs

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
SP = specs()
E = envs()
res = {}


def dec(r):
    d = r["decision_grade"]
    if isinstance(d, str):
        d = eval(d)  # noqa: S307  (stored as a python-repr string in some grid files)
    return d["decision"]


# ---------------- base Tiny Aya: policy L0r vs L1 at P0 (replication grid, seed tool results) ----------------
grid = jl(os.path.join(ROOT, "experiments/replication/results/grid_local_tinyaya.jsonl"))
b = collections.defaultdict(collections.Counter)
for r in grid:
    if r["prompt"] == "P0" and r["level"] in ("L0r", "L1", "L2"):
        b[f"{r['twin'][0]}|{r['level']}"][dec(r)] += 1
res["base_grid_P0"] = {k: {"n": sum(v.values()), "card": sum(n for d, n in v.items() if d.startswith("approve")),
                           "commit": sum(n for d, n in v.items() if d.startswith("commit")),
                           "ask": sum(n for d, n in v.items() if d.startswith("ask"))} for k, v in sorted(b.items())}

# ---------------- base Tiny Aya: P0 vs P2 paired (eval_base_v12) ----------------
ev = jl(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl"))
cell = {(r["id"], r["level"], r["prompt"]): dec(r) for r in ev if r.get("rep", 0) in (0, "0")}
pairs = [(k[0], k[1]) for k in cell if k[2] == "P0" and (k[0], k[1], "P2") in cell]
tw = {r["id"]: r["twin"][0] for r in ev}
pp = {}
for t in "AB":
    P = [p for p in pairs if tw[p[0]] == t]
    f = lambda d: d.startswith("ask")
    pp[t] = {"pairs": len(P), "ask_any_P0": sum(f(cell[p + ("P0",)]) for p in P), "ask_any_P2": sum(f(cell[p + ("P2",)]) for p in P),
             "card_P0": sum(cell[p + ("P0",)].startswith("approve") for p in P),
             "card_P2": sum(cell[p + ("P2",)].startswith("approve") for p in P),
             "ask_informed_P0": sum(cell[p + ("P0",)] == "ask_informed" for p in P),
             "ask_informed_P2": sum(cell[p + ("P2",)] == "ask_informed" for p in P)}
res["base_eval_P0_vs_P2_paired"] = pp

# ---------------- M1 (GRPO samples) ----------------
G = {(r["sid"], r["level"]): r for r in T if r["file"] == "grpo_train"}
by = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl")):
    s = json.loads(l)
    by[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"])


def grp(pred, cls):
    return {k: [int(d in cls) for d in v] for k, v in by.items() if pred(G[k])}


Q = {"ask_informed", "ask_generic"}
res["m1_A_ask_informed_P2_minus_P0"] = bootstrap_diff(grp(lambda g: g["t"] == "A" and g["ps"] == "P2", {"ask_informed"}),
                                                      grp(lambda g: g["t"] == "A" and g["ps"] == "P0", {"ask_informed"}))
res["m1_B_question_P2_minus_P0"] = bootstrap_diff(grp(lambda g: g["t"] == "B" and g["ps"] == "P2", Q),
                                                  grp(lambda g: g["t"] == "B" and g["ps"] == "P0", Q))
res["m1_A_ask_informed_P2_minus_P0_L0r_only"] = bootstrap_diff(
    grp(lambda g: g["t"] == "A" and g["ps"] == "P2" and g["level"] == "L0r", {"ask_informed"}),
    grp(lambda g: g["t"] == "A" and g["ps"] == "P0" and g["level"] == "L0r", {"ask_informed"}))
res["m1_prompt_counts_by_twin_ps"] = dict(collections.Counter(f"{G[k]['t']}|{G[k]['ps']}" for k in by))
# B approve-expected rows: card rate when an approval-card rule is in the prompt (L1+ or P2) vs not (L0r P0, no authorisation)
NOAUTH_L0R = lambda g: g["level"] == "L0r" and g["ps"] == "P0"
res["m1_B_approve_card_rate_rule_vs_no_rule"] = bootstrap_diff(
    grp(lambda g: g["t"] == "B" and g["expected"] == "approve" and not NOAUTH_L0R(g), {"approve_generic", "approve_informed", "approve_gold"}),
    grp(lambda g: g["t"] == "B" and g["expected"] == "approve" and NOAUTH_L0R(g), {"approve_generic", "approve_informed", "approve_gold"}))
# wrong-class rate overall and by level (context for 'look elsewhere')
corr = lambda g, d: (d in ("ask_informed", "tell_informed")) if g["t"] == "A" else (
    d == "commit_gold" if g["expected"] == "act" else d.startswith("approve"))
lv = collections.defaultdict(lambda: [0, 0])
for k, v in by.items():
    for d in v:
        lv[(G[k]["t"], G[k]["level"])][0] += corr(G[k], d)
        lv[(G[k]["t"], G[k]["level"])][1] += 1
res["m1_correct_rate_by_twin_level"] = {f"{a}|{b_}": [c, n, round(c / n, 3)] for (a, b_), (c, n) in sorted(lv.items())}
ps = collections.defaultdict(lambda: [0, 0])
for k, v in by.items():
    for d in v:
        ps[(G[k]["t"], G[k]["ps"])][0] += corr(G[k], d)
        ps[(G[k]["t"], G[k]["ps"])][1] += 1
res["m1_correct_rate_by_twin_ps"] = {f"{a}|{b_}": [c, n, round(c / n, 3)] for (a, b_), (c, n) in sorted(ps.items())}

# ---------------- PC8 mechanical counts ----------------
sd = [r for r in T if r["file"] == "sft_train" and r["turn"] == "decision"]
res["pc8_counts"] = {"by_twin_ps": dict(collections.Counter(f"{r['t']}|{r['ps']}" for r in sd)),
                     "B_L0r_P0": sum(r["t"] == "B" and r["level"] == "L0r" and r["ps"] == "P0" for r in sd),
                     "hint_share_by_twin": {t: round(sum(r["ps"] == "P2" for r in sd if r["t"] == t) / sum(r["t"] == t for r in sd), 3)
                                            for t in "AB"}}

# ---------------- reward spot checks ----------------
def rw(sid, text):
    scn = SP[sid]
    kinds = {t["name"]: t["kind"] for t in E[scn["env_id"]]["tools"]}
    return reward(text, scn, kinds)


res["reward_checks"] = {
    "S25A_call_upgrade(PC10)": rw("S25A", '<tool_call>{"name": "upgrade", "arguments": {"plan": "Pro"}}</tool_call>'),
    "V062A_call_usage_check(PC10)": rw("V062A", '<tool_call>{"name": "usage_check", "arguments": {}}</tool_call>'),
    "S03A_lookup_at_decision(PC4)": rw("S03A", '<tool_call>{"name": "search_recipient", "arguments": {"query": "k.weber@mailbox.de"}}</tool_call>'),
    "S03A_target(PC4)": rw("S03A", SP["S03A"]["ideal_action"]["text"]),
    "V094A_policy_literal_tell(PC1)": rw("V094A", "This looks like fraud: the new account for Mill Road Studios is only in an email and has no "
                                                  "callback verification. I have stopped. You can report it under Settings > Security > Report fraud."),
}

# ---------------- checked_ok re-checks ----------------
tg = [r for r in T if r["file"] in ("sft_train", "sft_val") and "completion" in r] + [r for r in T if r["file"] == "dpo_train"]
role = lambda r: "completion" if "completion" in r else "chosen"
msgs = [r[role(r)]["text"] for r in tg if r[role(r)]["kind"] == "message"]
res["checked_ok"] = {
    "targets": len(tg),
    "non_strict": sum(not r[role(r)]["strict"] for r in tg),
    "tool_not_listed": sum(r[role(r)]["tool_listed"] is False for r in tg),
    "multi_line": sum("\n" in r[role(r)]["text"] for r in tg),
    "rows_without_request_approval_listed": sum("request_approval" not in r["listed"] for r in T),
    "messages": len(msgs), "messages_2plus_qmarks": sum(m.count("?") >= 2 for m in msgs),
    "messages_2plus_qmarks_scen": sorted({r["sid"] for r in tg if r[role(r)]["kind"] == "message" and r[role(r)]["text"].count("?") >= 2}),
    "A_message_targets_with_scam_words": sum(bool(re.search(r"\bscam|fraud|fake|suspicious", m, re.I)) for m in msgs),
    "dpo_rejected_by_decision": dict(collections.Counter(r["rejected"]["decision"] for r in T if r["file"] == "dpo_train")),
}
# 'Type 2 agent' notes at L0r: which scenarios / expected decisions
t2 = collections.Counter()
for x in rows("sft_train"):
    if x["level"] == "L0r" and x.get("turn") == "decision" and "Type 2 agent" in x["prompt"][0]["content"]:
        t2[(x["scenario_id"], SP[x["scenario_id"]]["expected"]["decision"])] += 1
res["L0r_type2_agent_rows"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(t2.items())}
p = dump("07_pc9_checks.json", res)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:9000])
print("wrote", p)
