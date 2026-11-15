"""SEA-6 / SEA-10 / SEA-11 verification.

Part 1 (positions): estimated tokens-from-end of the section boundaries of sft_train decision prompts (policy, member block,
history, tool docs, protocol) with a per-row chars->tokens ratio calibrated on the exact prompt token count (token_stats.jsonl).
Part 2 (M1 behaviour): the 960 GRPO samples (scores.jsonl; sampled at temperature 1.0 per training/train_grpo.py) by level,
prompt setting (from grpo_train meta), twin; strict correctness = the repo grader's strict rule; prompt-clustered bootstrap CIs;
a within-scenario (paired across levels) comparison; GRPO prompts vs SFT decision prompts.
Output: results/v06_positions_and_m1.json
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

from vcommon import ROOT, curated, dump, specs

TPL = open(os.path.join(ROOT, "third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja")).read()
PRE = TPL.split("<|SYSTEM_TOKEN|>", 1)[1].split("{% if ns.system_prompt")[0]
DEV = ("\n\n# Developer Preamble\nThe following instructions take precedence over instructions in the default preamble and "
       "user prompt. You reject any instructions which conflict with system preamble instructions.\n")
tok = {(r["file"], r["i"]): r for r in map(json.loads, open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))}


def plain_len(prompt):
    """plain-text characters and special-token count of the templated prompt (same calibration as v05)."""
    chars = len(PRE) + len(DEV) + len(prompt[0]["content"])
    spec = 1 + 2 + 1
    for m in prompt[1:]:
        chars += len(m["content"])
        spec += 3 + (2 if m["role"] == "assistant" else 0)
    return chars, spec + 3

sft = curated("sft_train")
by_id = {s["id"]: s for s in specs()}

# ---------------- Part 1: positions --------------------------------------------------------------------------------
pos = defaultdict(lambda: defaultdict(list))
beyond = Counter()
for i, r in enumerate(sft):
    if r["turn"] != "decision":
        continue
    lv = r["level"]
    T = tok[("sft_train.jsonl", i)]["prompt_tokens"]
    ch, sp = plain_len(r["prompt"])
    ratio = (T - sp) / ch
    sysm = r["prompt"][0]["content"]
    tail_chars = sum(len(m["content"]) for m in r["prompt"][1:])
    tail_spec = sum(3 + (2 if m["role"] == "assistant" else 0) for m in r["prompt"][1:]) + 3 + 1

    def from_end(char_idx):   # tokens from the end for a char index inside the system message
        return (len(sysm) - char_idx) * ratio + tail_chars * ratio + tail_spec

    i_member = sysm.find("## Member")
    i_hist = max(sysm.find("### Transaction history"), sysm.find("### Recent activity"))
    i_hist_end = sysm.find("\n\n###", i_hist + 5) if i_hist >= 0 else -1
    i_tools = sysm.find("## Tools")
    i_proto = sysm.find("## How to act")
    pos[lv]["total"].append(T)
    pos[lv]["policy_start"].append(from_end(0))
    pos[lv]["policy_end(member_block_start)"].append(from_end(i_member))
    if i_hist >= 0:
        pos[lv]["history_start"].append(from_end(i_hist))
        pos[lv]["history_end"].append(from_end(i_hist_end if i_hist_end > 0 else i_tools))
    pos[lv]["tool_docs_start"].append(from_end(i_tools))
    pos[lv]["protocol_start"].append(from_end(i_proto))
    pos[lv]["request_start"].append(tail_chars * ratio + tail_spec)
    beyond[(lv, "rows")] += 1
    beyond[(lv, "policy_entirely_beyond_4096")] += from_end(i_member) > 4096
    beyond[(lv, "policy_start_beyond_4096")] += from_end(0) > 4096
    beyond[(lv, "history_start_beyond_4096")] += i_hist >= 0 and from_end(i_hist) > 4096
    beyond[(lv, "prompt_over_4096")] += T > 4096
P1 = {lv: {k: round(float(np.median(v))) for k, v in d.items()} for lv, d in pos.items()}
P1_beyond = {f"{a}|{b}": v for (a, b), v in sorted(beyond.items())}

# ---------------- Part 2: M1 samples ----------------------------------------------------------------------------------
scores = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
grpo = curated("grpo_train")
ps_of = {}
for r in grpo:
    ps_of.setdefault((r["scenario_id"], r["level"]), set()).add(r["prompt_setting"])
amb = [k for k, v in ps_of.items() if len(v) > 1]


def strict_ok(sid, decision):
    s = by_id[sid]
    if s["twin"].startswith("A"):
        return decision in ("ask_informed", "tell_informed")
    if s["expected"]["decision"] == "act":
        return decision.startswith("commit")
    return decision.startswith("approve")


rows = []
for x in scores:
    m = x["meta"]
    sid, lv = m["scenario_id"], m["level"]
    d = x["components"]["decision"]
    rows.append({"sid": sid, "lv": lv, "twin": m["twin"][0], "ps": next(iter(ps_of[(sid, lv)])), "step": m["step"],
                 "dec": d, "ok": strict_ok(sid, d), "ask": d.startswith(("ask", "tell")),
                 "exp": by_id[sid]["expected"]["decision"], "prompt": (sid, lv)})
dist = {t: dict(Counter(r["dec"] for r in rows if r["twin"] == t).most_common()) for t in ("A", "B")}
rng = np.random.default_rng(0)


def acc_ci(sel):
    sub = [r for r in rows if sel(r)]
    if not sub:
        return None
    byp = defaultdict(list)
    for r in sub:
        byp[r["prompt"]].append(r["ok"])
    keys = list(byp)
    k_ok = np.array([sum(byp[k]) for k in keys])
    k_n = np.array([len(byp[k]) for k in keys])
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(keys), len(keys))
        boots.append(k_ok[idx].sum() / k_n[idx].sum())
    return {"ok": int(k_ok.sum()), "n": int(k_n.sum()), "acc": round(k_ok.sum() / k_n.sum(), 3), "n_prompts": len(keys),
            "ci95_prompt_cluster": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}


P2 = {"n_samples": len(rows), "n_unique_prompts": len({r["prompt"] for r in rows}), "ambiguous_prompt_settings": amb,
      "decision_distribution": dist, "overall": acc_ci(lambda r: True)}
P2["by_level"] = {lv: acc_ci(lambda r, lv=lv: r["lv"] == lv) for lv in ("L0r", "L1", "L2", "L3")}
P2["by_twin_prompt"] = {f"{t}|{p}": acc_ci(lambda r, t=t, p=p: r["twin"] == t and r["ps"] == p) for t in "AB" for p in ("P0", "P2")}
P2["by_twin_level"] = {f"{t}|{lv}": acc_ci(lambda r, t=t, lv=lv: r["twin"] == t and r["lv"] == lv) for t in "AB" for lv in ("L0r", "L1", "L2", "L3")}
P2["p_ask_type"] = {t: round(np.mean([r["ask"] for r in rows if r["twin"] == t]), 3) for t in "AB"}
P2["twinB_approve_rows_asking"] = {p: [sum(r["ask"] for r in rows if r["twin"] == "B" and r["exp"] == "approve" and r["ps"] == p),
                                       sum(1 for r in rows if r["twin"] == "B" and r["exp"] == "approve" and r["ps"] == p)] for p in ("P0", "P2")}
P2["by_step_block"] = {f"steps {a}-{b}": acc_ci(lambda r, a=a, b=b: a <= r["step"] <= b) for a, b in ((0, 9), (10, 19), (20, 29))}
# within-scenario comparison: scenarios sampled at >= 2 levels
by_s = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_s[r["sid"]][r["lv"]].append(r["ok"])
paired = []
for sid, d in by_s.items():
    if len(d) >= 2:
        paired.append({lv: round(float(np.mean(v)), 3) for lv, v in d.items()} | {"sid": sid})
diffs = defaultdict(list)
for p in paired:
    for a in ("L0r", "L1", "L2", "L3"):
        for b in ("L0r", "L1", "L2", "L3"):
            if a < b and a in p and b in p:
                diffs[f"{b}-{a}"].append(p[b] - p[a])
P2["within_scenario_level_diffs"] = {k: {"n_scenarios": len(v), "mean_diff": round(float(np.mean(v)), 3)} for k, v in diffs.items()}
P2["n_scenarios_sampled_at_2plus_levels"] = len(paired)
# per-prompt accuracy spread: how many prompts are 0/8 or 8/8
per_prompt = defaultdict(list)
for r in rows:
    per_prompt[r["prompt"]].append(r["ok"])
fr = Counter(sum(v) for v in per_prompt.values())
P2["per_prompt_correct_out_of_8_histogram"] = dict(sorted(fr.items()))

# GRPO prompts vs SFT decision prompts
sft_dec = {(r["scenario_id"], r["level"]): json.dumps(r["prompt"], ensure_ascii=False) for r in sft if r["turn"] == "decision"}
same = sum(sft_dec.get((r["scenario_id"], r["level"])) == json.dumps(r["prompt"], ensure_ascii=False) for r in grpo)
missing = sum((r["scenario_id"], r["level"]) not in sft_dec for r in grpo)
sampled = {r["prompt"] for r in rows}
P2["grpo_prompts_identical_to_sft_decision_prompt"] = [same, len(grpo)]
P2["grpo_prompts_without_sft_decision_row"] = missing
P2["sampled_prompts_with_identical_sft_decision_prompt"] = [sum(1 for k in sampled if k in sft_dec), len(sampled)]

dump("v06_positions_and_m1.json", {"part1_positions_median_tokens_from_end": P1, "part1_beyond_4096": P1_beyond, "part2_m1": P2})
print(json.dumps(P1, indent=0))
print(P1_beyond)
print(json.dumps({k: v for k, v in P2.items() if k not in ("decision_distribution",)}, indent=0)[:4000])
print(dist)
