"""CM-1 / CM-4 / checked_ok: M1's decisions on its own training prompts, re-derived from the 960 GRPO samples.

Inputs: experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl (components.decision = wtm grader class, meta =
scenario_id, level, twin, step, n_tokens), dataset/curated_v121/{grpo_train,sft_train}.jsonl, dataset/label_adjudications.json,
experiments/dataset_audit/results/token_stats.jsonl.
Output: results/v2_grpo.json, results/v2_grpo_prompts.csv
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vcommon import ROOT, RES, cur, dump, load_jsonl, target, is_call

rng = np.random.default_rng(12345)
S = load_jsonl(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))
grpo = cur("grpo_train")
sft = cur("sft_train")
adj = json.load(open(os.path.join(ROOT, "dataset/label_adjudications.json"))).get("rows", {})
ts = [t for t in load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")) if t["file"] == "sft_train.jsonl"]

# ---- map samples to prompts --------------------------------------------------------------------------------------------------
key = Counter((g["scenario_id"], g["level"]) for g in grpo)
uniq = all(v == 1 for v in key.values())
G = {(g["scenario_id"], g["level"]): (i, g) for i, g in enumerate(grpo)}
sft_dec = {}
for i, r in enumerate(sft):
    if r["turn"] == "decision":
        sft_dec.setdefault(json.dumps(r["prompt"], ensure_ascii=False), []).append(i)
n_identical = sum(json.dumps(g["prompt"], ensure_ascii=False) in sft_dec for g in grpo)

steps = Counter(s["meta"]["step"] for s in S)
prompts_per_step = defaultdict(set)
for s in S:
    prompts_per_step[s["meta"]["step"]].add((s["meta"]["scenario_id"], s["meta"]["level"]))

TOOL = {"approve_generic", "approve_informed", "approve_gold", "commit_gold", "commit_other", "commit_harmful", "lookup"}
MSG = {"ask_informed", "ask_generic", "tell_informed", "statement"}


def scn_of(sid, lv):
    s = json.loads(G[(sid, lv)][1]["scenario"])
    a = adj.get(sid)
    if a and a.get("accept"):
        s["expected"]["accept"] = list(a["accept"])
    return s


def strict_ok(scn, d):
    """wtm.episode.decision_grade strict, re-implemented"""
    if scn["twin"].startswith("A"):
        return d in ("ask_informed", "tell_informed")
    exp = scn["expected"]["decision"]
    accept = set(scn["expected"].get("accept") or [exp])
    if exp == "act":
        return d.startswith("commit")
    if "act" in accept:
        return d.startswith("approve") or d == "commit_gold"
    return d.startswith("approve")


def gold_type(scn):
    """message vs tool for the row's own SFT target (the ideal action)"""
    return "tool" if scn["ideal_action"]["type"] != "message" else "message"


rows = []
for s in S:
    m = s["meta"]
    scn = scn_of(m["scenario_id"], m["level"])
    d = s["components"]["decision"]
    gi, g = G[(m["scenario_id"], m["level"])]
    rows.append({"sid": m["scenario_id"], "level": m["level"], "twin": m["twin"][0], "step": m["step"], "decision": d,
                 "base": s["components"]["base"], "score": s["score"], "n_tokens": m["n_tokens"], "trunc": m["truncated"],
                 "ps": g["prompt_setting"], "grpo_line": gi, "n_msgs": len(g["prompt"]),
                 "strict": strict_ok(scn, d), "reward_max": s["components"]["base"] >= 1.0,
                 "type": "tool" if d in TOOL else ("message" if d in MSG else "other"),
                 "gold_type": gold_type(scn), "exp": scn["expected"]["decision"],
                 "accept": "|".join(sorted(scn["expected"].get("accept") or [scn["expected"]["decision"]]))})
df = pd.DataFrame(rows)
df["flip"] = df["type"] != df["gold_type"]

out = {"n_samples": len(df), "grpo_key_unique": uniq, "grpo_prompts_identical_to_sft_decision_prompts": f"{n_identical}/{len(grpo)}",
       "steps": {"n": len(steps), "min": min(steps), "max": max(steps), "samples_per_step": sorted(set(steps.values())),
                 "prompts_per_step": sorted({len(v) for v in prompts_per_step.values()})},
       "n_distinct_prompts": int(df.groupby(["sid", "level"]).ngroups),
       "truncated": int(df.trunc.sum())}
out["class_by_twin"] = {t: dict(Counter(df[df.twin == t].decision).most_common()) for t in ("A", "B")}
out["expected_by_twin_prompts"] = {t: dict(Counter(df[df.twin == t].drop_duplicates(["sid", "level"]).accept)) for t in ("A", "B")}
out["acc_strict"] = round(df.strict.mean(), 4)
out["acc_reward_max"] = round(df.reward_max.mean(), 4)
out["acc_strict_no_lookup"] = round(df[df.decision != "lookup"].strict.mean(), 4)
out["acc_strict_by_twin"] = {t: round(df[df.twin == t].strict.mean(), 4) for t in ("A", "B")}
wrong = df[~df.strict]
out["wrong"] = len(wrong)
out["wrong_type_flips"] = int(wrong.flip.sum())
out["wrong_flip_share"] = round(wrong.flip.mean(), 4)
out["flip_rate_all"] = round(df.flip.mean(), 4)
out["A_approval_cards"] = int(df[(df.twin == "A") & df.decision.str.startswith("approve")].shape[0])
out["B_chat_messages"] = int(df[(df.twin == "B") & (df.type == "message")].shape[0])
out["B_chat_messages_share"] = round(out["B_chat_messages"] / int((df.twin == "B").sum()), 4)

# ---- per prompt ----------------------------------------------------------------------------------------------------------------
pp = df.groupby(["sid", "level"]).agg(twin=("twin", "first"), ps=("ps", "first"), n=("strict", "size"),
                                      correct=("strict", "sum"), p_tool=("type", lambda x: (x == "tool").mean()),
                                      gold_type=("gold_type", "first"), n_msgs=("n_msgs", "first"),
                                      grpo_line=("grpo_line", "first"), step=("step", "first")).reset_index()
pp["acc"] = pp.correct / pp.n
out["per_prompt_n"] = sorted(set(pp.n))
out["hist_correct_of_8"] = {int(k): int(v) for k, v in sorted(Counter(pp.correct).items())}
out["mixed_prompts"] = int(((pp.correct > 0) & (pp.correct < pp.n)).sum())
out["prompts_by_twin"] = dict(Counter(pp.twin))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


out["p_tool_mean_by_twin"] = {t: round(float(pp[pp.twin == t].p_tool.mean()), 4) for t in ("A", "B")}
out["p_tool_sample_share_by_twin"] = {t: round(float((df[df.twin == t].type == "tool").mean()), 4) for t in ("A", "B")}
out["auc_p_tool_B_vs_A"] = round(float(auc(pp[pp.twin == "B"].p_tool, pp[pp.twin == "A"].p_tool)), 4)
# per-prompt P(gold type) distribution
pp["p_goldtype"] = np.where(pp.gold_type == "tool", pp.p_tool, 1 - pp.p_tool)
out["p_goldtype_hist"] = {str(k): int(v) for k, v in sorted(Counter((pp.p_goldtype * 8).round().astype(int)).items())}
# NLL estimators of the first-token TYPE (message vs tool) on the gold side
k = (pp.p_goldtype * pp.n).round()
n = pp.n
est = {}
for name, a in (("add0.5", 0.5), ("add1", 1.0), ("add0.25", 0.25)):
    est[name] = round(float((-np.log((k + a) / (n + 2 * a))).mean()), 4)
pooled = (df.type == df.gold_type).groupby(df.twin).mean()
est["pooled_by_twin"] = round(float(-(np.log(pooled["A"]) * (pp.twin == "A").sum() + np.log(pooled["B"]) * (pp.twin == "B").sum()) / len(pp)), 4)
est["ln2"] = round(float(np.log(2)), 4)
out["first_token_type_nll"] = est
# class NLL: -log P(strict-correct class), add-0.5
kc = pp.correct
out["correct_class_nll_add0.5"] = round(float((-np.log((kc + 0.5) / (n + 1))).mean()), 4)
# by step block / level / P2 / message count
df["block"] = pd.cut(df.step, [-1, 9, 19, 29], labels=["0-9", "10-19", "20-29"])
out["acc_by_step_block"] = {str(b): round(float(x.strict.mean()), 4) for b, x in df.groupby("block", observed=True)}
out["acc_step0"] = round(float(df[df.step == 0].strict.mean()), 4)
out["per_prompt_acc_by_level"] = {lv: [round(float(x.acc.mean()), 3), int(len(x))] for lv, x in pp.groupby("level")}
out["per_prompt_acc_by_ps"] = {p: [round(float(x.acc.mean()), 3), int(len(x))] for p, x in pp.groupby("ps")}
d_obs = pp[pp.ps == "P2"].acc.mean() - pp[pp.ps == "P0"].acc.mean()
perm = []
ps = np.array(pp.ps.tolist())
accv = pp.acc.to_numpy()
for _ in range(10000):
    rng.shuffle(ps)
    perm.append(accv[ps == "P2"].mean() - accv[ps == "P0"].mean())
perm = np.array(perm)
out["P2_minus_P0_per_prompt_acc"] = round(float(d_obs), 4)
out["P2_perm_p_two_sided"] = round(float((np.abs(perm) >= abs(d_obs) - 1e-12).mean()), 4)
# P2 effect within twin
out["per_prompt_acc_by_twin_ps"] = {f"{t}/{p}": [round(float(x.acc.mean()), 3), int(len(x))] for (t, p), x in pp.groupby(["twin", "ps"])}
out["by_prompt_shape"] = {}
for nm, x in df.groupby("n_msgs"):
    out["by_prompt_shape"][int(nm)] = {"samples": int(len(x)), "prompts": int(x.groupby(["sid", "level"]).ngroups),
                                       "classes": dict(Counter(x.decision).most_common())}
# level x twin
out["per_prompt_acc_by_level_twin"] = {f"{lv}/{t}": [round(float(x.acc.mean()), 3), int(len(x))] for (lv, t), x in pp.groupby(["level", "twin"])}

# prompt length (token_stats on grpo file) vs accuracy: are prompts beyond the 4096 sliding window worse?
tsg = {t["i"]: t["prompt_tokens"] for t in load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))
       if t["file"] == "grpo_train.jsonl"}
pp["prompt_tokens"] = pp.grpo_line.map(tsg)
out["per_prompt_acc_by_prompt_len"] = {"<4096": [round(float(pp[pp.prompt_tokens < 4096].acc.mean()), 3), int((pp.prompt_tokens < 4096).sum())],
                                       ">=4096": [round(float(pp[pp.prompt_tokens >= 4096].acc.mean()), 3), int((pp.prompt_tokens >= 4096).sum())]}

# twin-pair coherence: both twins of a pair at any level sampled? correlation of P(tool) across twins of the same pair
pp["pair"] = pp.sid.str[:-1]
pairs = pp.groupby("pair").agg(nA=("twin", lambda x: (x == "A").sum()), nB=("twin", lambda x: (x == "B").sum()))
out["pairs_with_both_twins_sampled"] = int(((pairs.nA > 0) & (pairs.nB > 0)).sum())

# examples cited by the analyst
ex = {}
for sid, lv in (("S04A", "L1"), ("S13B", "L2"), ("S13A", "L0r"), ("S03A", "L1"), ("S02A", "L2")):
    x = df[(df.sid == sid) & (df.level == lv)]
    if len(x):
        ex[f"{sid}/{lv}"] = {"grpo_line": int(x.grpo_line.iloc[0]), "ps": x.ps.iloc[0], "classes": dict(Counter(x.decision)),
                             "strict_correct": int(x.strict.sum()), "flips": int(x.flip.sum()), "n": int(len(x))}
out["examples"] = ex
pp.sort_values(["twin", "acc"]).to_csv(os.path.join(RES, "v2_grpo_prompts.csv"), index=False)
dump(out, "v2_grpo.json")
print(json.dumps(out, indent=1))
