"""Re-derive every number the analyst took from the M1 policy samples (m2_grpo/scores.jsonl), with my own join and features.

- data identity: sha256 of grpo_train rows computed the way train_grpo.py does (json.dumps(rows, sort_keys=True)) vs args.json
- class counts by twin / expectation / level / prompt setting / step bucket
- error decomposition (type errors vs keyword-dependent)
- GRPO groups (8 consecutive samples per prompt) : spread, std, advantages (population std AND torch's unbiased std, as TRL uses)
- keyword visibility per GRPO prompt (my own computation with wtm.spec.kw_hit): request / look-up results / system
- look-ups per (scenario, level), adjudicated rows, B ask_informed vs visibility
-> results/v_scores_summary.json
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from vcommon import ROOT, SCORES, SP, adjudicate, dump, jl, rows, sha_rows

args = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/args.json")))
grpo = rows("grpo_train")
S = {}
S["data_sha_matches_args"] = sha_rows(grpo) == args["data_sha256"]

idx = {(r["scenario_id"], r["level"]): i for i, r in enumerate(grpo)}
S["grpo_rows"] = len(grpo)
S["grpo_key_unique"] = len(idx) == len(grpo)

# keyword visibility (own computation)
vis = {}
for i, r in enumerate(grpo):
    scn = json.loads(r["scenario"])
    K = scn["expected"].get("informed_any", [])
    p = r["prompt"]
    system, user = p[0]["content"], p[1]["content"]
    results = "\n".join(m["content"] for m in p[2:] if m["role"] == "user")
    n_prefix = sum(1 for m in p if m["role"] == "assistant")
    vis[i] = {"req": [k for k in K if SP.kw_hit(user, k)], "res": [k for k in K if SP.kw_hit(results, k)],
              "sys": [k for k in K if SP.kw_hit(system, k)], "n_prefix": n_prefix, "nK": len(K),
              "expect": "A" if scn["twin"].startswith("A") else ("B_act" if scn["expected"]["decision"] == "act" else "B_approve"),
              "adjudicated": bool(adjudicate(scn)["expected"].get("accept")), "ps": r["prompt_setting"]}

sc = jl(SCORES)
recs = []
for j, s in enumerate(sc):
    m = s["meta"]
    i = idx.get((m["scenario_id"], m["level"]))
    v = vis[i]
    recs.append({"j": j, "i": i, "sid": m["scenario_id"], "lv": m["level"], "tw": m["twin"][0], "step": m["step"],
                 "cls": s["components"]["decision"], "score": s["score"], "base": s["components"]["base"],
                 "fmt": s["components"]["format"], "trunc": m["truncated"], "ntok": m["n_tokens"], **v})
S["n_samples"] = len(recs)
S["unjoined"] = sum(r["i"] is None for r in recs)
S["truncated"] = sum(bool(r["trunc"]) for r in recs)
S["n_tokens_max"] = max(r["ntok"] for r in recs)

for tw in "AB":
    rs = [r for r in recs if r["tw"] == tw]
    S[f"{tw}_n"] = len(rs)
    S[f"{tw}_classes"] = dict(Counter(r["cls"] for r in rs).most_common())
S["by_expect"] = {e: dict(Counter(r["cls"] for r in recs if r["expect"] == e).most_common()) for e in ("A", "B_approve", "B_act")}
S["n_by_expect"] = dict(Counter(r["expect"] for r in recs))

# top class (task definition) and decomposition
def top(r):
    if r["expect"] == "A":
        return r["cls"] == "ask_informed"
    if r["expect"] == "B_act":
        return r["cls"] == "commit_gold"
    return r["cls"].startswith("approve")


def ok_reward(r):          # reward >= 1.0 (the top-class value incl. format)
    return r["score"] >= 1.0


A = [r for r in recs if r["tw"] == "A"]
B = [r for r in recs if r["tw"] == "B"]
S["decomp"] = {
    "A_top": sum(top(r) for r in A),
    "A_approve_or_commit": sum(r["cls"].startswith(("approve", "commit")) for r in A),
    "A_msg_not_top(ask_generic+statement+tell_informed)": sum(r["cls"] in ("ask_generic", "statement", "tell_informed") for r in A),
    "A_lookup": sum(r["cls"] == "lookup" for r in A),
    "A_invalid": sum(r["cls"] == "invalid" for r in A),
    "B_top_task_def(approve_* on approve rows, commit_gold on act rows)": sum(top(r) for r in B),
    "B_approve_any_or_commit_gold": sum(r["cls"].startswith("approve") or r["cls"] == "commit_gold" for r in B),
    "B_reward_ge_1": sum(ok_reward(r) for r in B),
    "B_chat_message": sum(r["cls"].startswith(("ask", "tell", "statement")) for r in B),
    "B_other": None,
}
S["decomp"]["B_other"] = len(B) - S["decomp"]["B_approve_any_or_commit_gold"] - S["decomp"]["B_chat_message"]
S["B_act_rows_classes"] = dict(Counter(r["cls"] for r in B if r["expect"] == "B_act"))
S["B_act_rows_reward_ge_1"] = sum(ok_reward(r) for r in B if r["expect"] == "B_act")

# by level / prompt setting: top-class rate
for key in ("lv", "ps"):
    S[f"top_by_{key}"] = {f"{tw}|{v}": [sum(top(r) for r in recs if r['tw'] == tw and r[key] == v),
                                         sum(1 for r in recs if r['tw'] == tw and r[key] == v)]
                          for tw in "AB" for v in sorted({r[key] for r in recs})}

# groups: TRL RepeatSampler puts the num_generations samples of one prompt next to each other; check that, then group
blocks = [recs[k:k + 8] for k in range(0, len(recs), 8)]
S["blocks_of_8_same_prompt"] = sum(len({(r["sid"], r["lv"], r["step"]) for r in b}) == 1 for b in blocks)
S["n_blocks"] = len(blocks)
keyed = defaultdict(list)
for r in recs:
    keyed[(r["sid"], r["lv"], r["step"])].append(r)
S["group_sizes"] = dict(Counter(len(g) for g in keyed.values()))
stds_pop, zero, adv_pop, adv_unb = [], 0, defaultdict(list), defaultdict(list)
for g in keyed.values():
    x = np.array([r["score"] for r in g], dtype=float)
    mu, sp, su = x.mean(), x.std(ddof=0), x.std(ddof=1)
    stds_pop.append(sp)
    if sp < 1e-9:
        zero += 1
        continue
    for r, xi in zip(g, x):
        adv_pop[(r["tw"], r["cls"])].append((xi - mu) / sp)
        adv_unb[(r["tw"], r["cls"])].append((xi - mu) / (su + 1e-4))
S["groups"] = {"n": len(keyed), "zero_spread": zero, "mean_within_std_pop": round(float(np.mean(stds_pop)), 4)}
S["adv_pop"] = {f"{k[0]}|{k[1]}": [len(v), round(float(np.mean(v)), 3)] for k, v in sorted(adv_pop.items())}
S["adv_trl_unbiased"] = {f"{k[0]}|{k[1]}": [len(v), round(float(np.mean(v)), 3)] for k, v in sorted(adv_unb.items())}

# keyword visibility splits (A)
def split(rs, pred, cls):
    t = [r for r in rs if pred(r)]
    f = [r for r in rs if not pred(r)]
    return {"true": [sum(r["cls"] == cls for r in t), len(t)], "false": [sum(r["cls"] == cls for r in f), len(f)]}


S["A_ask_informed_vs_kw_in_results"] = split(A, lambda r: bool(r["res"]), "ask_informed")
S["A_ask_generic_vs_kw_in_results"] = split(A, lambda r: bool(r["res"]), "ask_generic")
S["A_ask_generic_vs_kw_visible_anywhere"] = split(A, lambda r: bool(r["res"] or r["sys"] or r["req"]), "ask_generic")
S["B_ask_informed_vs_kw_visible_anywhere"] = split(B, lambda r: bool(r["res"] or r["sys"] or r["req"]), "ask_informed")
S["B_any_ask_vs_kw_visible_anywhere"] = {
    "true": [sum(r["cls"].startswith("ask") for r in B if (r["res"] or r["sys"] or r["req"])), sum(1 for r in B if (r["res"] or r["sys"] or r["req"]))],
    "false": [sum(r["cls"].startswith("ask") for r in B if not (r["res"] or r["sys"] or r["req"])), sum(1 for r in B if not (r["res"] or r["sys"] or r["req"]))]}
# GRPO prompt-level visibility counts
S["grpo_B_approve_rows_no_kw_visible"] = [sum(1 for i, v in vis.items() if v["expect"] == "B_approve" and not (v["res"] or v["sys"] or v["req"])),
                                          sum(1 for v in vis.values() if v["expect"] == "B_approve")]
S["grpo_A_rows_kw_in_request"] = sum(1 for v in vis.values() if v["expect"] == "A" and v["req"])
S["grpo_rows_no_prefix_call"] = sorted({(grpo[i]["scenario_id"]) for i, v in vis.items() if v["n_prefix"] == 0})
S["grpo_rows_no_prefix_call_n"] = sum(1 for v in vis.values() if v["n_prefix"] == 0)

# look-ups per (scenario, level)
lk = defaultdict(lambda: [0, 0])
for r in recs:
    lk[(r["sid"], r["lv"], r["i"], r["n_prefix"])][1] += 1
    lk[(r["sid"], r["lv"], r["i"], r["n_prefix"])][0] += r["cls"] == "lookup"
S["lookups_by_prompt"] = sorted([[k[0], k[1], f"grpo_train:{k[2]}", f"prefix_calls={k[3]}", v[0], v[1]] for k, v in lk.items() if v[0]],
                                key=lambda x: -x[4])
S["lookups_total"] = sum(r["cls"] == "lookup" for r in recs)
S["lookups_on_prompts_with_prefix"] = sum(r["cls"] == "lookup" for r in recs if r["n_prefix"] > 0)
S["lookup_scores"] = dict(Counter(r["score"] for r in recs if r["cls"] == "lookup"))

# adjudicated rows
S["adjudicated_samples"] = dict(Counter(r["cls"] for r in recs if r["adjudicated"]))

# reward means by step thirds
st = defaultdict(list)
for r in recs:
    st[r["step"]].append(r["score"])
S["mean_score_steps_0_14"] = round(float(np.mean([x for s_, v in st.items() if s_ < 15 for x in v])), 4)
S["mean_score_steps_15_29"] = round(float(np.mean([x for s_, v in st.items() if s_ >= 15 for x in v])), 4)
S["top_rate_by_step_bucket"] = {b: [sum(top(r) for r in recs if r["step"] // 10 == b), sum(1 for r in recs if r["step"] // 10 == b)]
                                for b in range(3)}
# examples the analyst cites
def per_prompt(i):
    return dict(Counter(r["cls"] for r in recs if r["i"] == i))


S["examples"] = {f"grpo_train:{i}": {"sid": grpo[i]["scenario_id"], "lv": grpo[i]["level"], "ps": grpo[i]["prompt_setting"],
                                     "classes": per_prompt(i), "kw_visible": vis[i]["res"] + vis[i]["sys"] + vis[i]["req"],
                                     "n_prefix": vis[i]["n_prefix"]}
                 for i in (4, 243, 494, 485, 262, 372, 435, 468, 21)}
p = dump(S, "v_scores_summary.json")
print(json.dumps(S, indent=1)[:9000])
print("->", p)
