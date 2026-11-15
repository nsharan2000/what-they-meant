"""Verifier step 3: what do the DPO pairs contrast, and can they be ordered without reading the prompt?

Inputs: results/v1_rows.jsonl (my replay), experiments/dataset_audit/results/token_stats.jsonl (exact Tiny Aya token counts),
        experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl (960 samples of the M1 policy on training prompts).
Models are numpy only:
  - prompt-free action-type Bradley-Terry: score(type); P(chosen > rejected) = sigmoid(s[tc] - s[tr])
  - context-aware action-type BT: score(row class, type), row class = A / B-approve / B-act
  - response-only logistic on feature differences (length, format, '?', ...)
All with 5-fold CV grouped by pair_id (twins in the same fold), averaged over 10 random fold assignments. A held-out pair counts
1 if the chosen side scores higher, 0.5 on an exact tie, else 0.
Output: results/v3_contrasts.json
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.episode import is_question  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = [json.loads(l) for l in open(os.path.join(OUT, "v1_rows.jsonl"))]
tok = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))]
tok = [t for t in tok if t["file"] == "dpo_train.jsonl"]
assert len(tok) == len(rows)
misaligned = sum(1 for t, r in zip(tok, rows) if (t["scenario_id"], t["level"], t["prompt_setting"]) != (r["scenario_id"], r["level"], r["prompt_setting"]))


def atype(kind, tool_kind, text, dec):
    """action type of one side: question / statement (chat), card, commit, lookup, invalid"""
    if kind == "invalid":
        return "invalid"
    if kind == "message":
        return "question" if is_question(text) else "statement"
    if tool_kind == "ui":
        return "card"
    if tool_kind == "commit":
        return "commit"
    if tool_kind in ("lookup", "verify"):
        return "lookup"
    return "invalid"


COMMIT_NAMES = None
for r in rows:
    r["ct"] = atype(r["chosen_parse_kind"], r["chosen_tool_kind"], r["chosen_text"], r["chosen_dec"])
    r["rt"] = atype(r["rej_parse_kind"], r["rej_tool_kind"], r["rej_text"], r["rej_dec"])
    r["cls"] = "A" if r["twin"] == "A" else ("B_act" if r["expected"] == "act" else "B_approve")
    t = tok[r["idx"]]
    r["c_tok"], r["r_tok"] = t["chosen_tokens"], t["rejected_tokens"]

# a tool call the parser could not read as a tool call: does its text name a commit tool? (malformed commit)
env_commit = defaultdict(set)
for e in {r["env_id"] for r in rows}:
    for tl in json.load(open(os.path.join(ROOT, "envs", e, "env.json")))["tools"]:
        if tl["kind"] == "commit":
            env_commit[e].add(tl["name"])
for r in rows:
    txt = r["rej_text"]
    named = [n for n in env_commit[r["env_id"]] if n in txt]
    r["rej_malformed_call"] = r["rt"] in ("invalid", "question", "statement") and ("tool_call" in txt or "<tool_" in txt or txt.strip().startswith("{"))
    r["rej_malformed_commit"] = r["rej_malformed_call"] and bool(named)


def axis(r):
    if r["rt"] == "commit" or (r["rt"] == "invalid" and r["rej_malformed_commit"]):
        return "X > direct commit (incl. malformed commit)"
    if r["rej_malformed_call"]:
        return "format: malformed call read as message/invalid"
    if r["cls"] == "A":
        if r["rt"] == "card":
            return "A: ask > approval card"
        if r["rt"] == "question":
            return "A: informed question > uninformed question"
        if r["rt"] == "statement":
            return "A: informed question > statement"
        return f"A: other ({r['rt']})"
    if r["rt"] == "question":
        return f"{r['cls']}: proceed > question"
    if r["rt"] == "statement":
        return f"{r['cls']}: proceed > statement"
    return f"{r['cls']}: other ({r['rt']})"


ax = Counter(axis(r) for r in rows)
ax_level = defaultdict(Counter)
for r in rows:
    ax_level[r["level"]][axis(r) == "X > direct commit (incl. malformed commit)"] += 1

# twin-level cross check: A's rejected type equals B's chosen type
tl = defaultdict(dict)
for r in rows:
    tl[(r["pair_id"], r["level"])][r["twin"]] = r
both = [v for v in tl.values() if "A" in v and "B" in v]
a_rej_eq_b_ch = sum(1 for v in both if v["A"]["rt"] == v["B"]["ct"])
b_rej_eq_a_ch = sum(1 for v in both if v["B"]["rt"] == v["A"]["ct"])


# ---- grouped CV models ----
pairs = sorted({r["pair_id"] for r in rows})
TYPES = ["question", "statement", "card", "commit", "lookup", "invalid"]


def folds(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    f = {pairs[p]: i % 5 for i, p in enumerate(perm)}
    return np.array([f[r["pair_id"]] for r in rows])


def fit_logistic(X, l2=1e-2, iters=3000, lr=0.5):
    """all rows have label 1 (chosen - rejected); fit w for sigmoid(X w) without intercept"""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = -(X.T @ (1 - p)) / len(X) + l2 * w
        w -= lr * g
    return w


def cv_score(X, seeds=range(10)):
    accs = []
    for sd in seeds:
        f = folds(sd)
        correct = np.zeros(len(rows))
        for k in range(5):
            tr, te = f != k, f == k
            w = fit_logistic(X[tr])
            s = X[te] @ w
            correct[te] = np.where(np.abs(s) < 1e-9, 0.5, (s > 0).astype(float))
        accs.append(correct.mean())
    return round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)


def onehot_diff(keyf):
    keys = sorted({keyf(r, "c") for r in rows} | {keyf(r, "r") for r in rows})
    ix = {k: i for i, k in enumerate(keys)}
    X = np.zeros((len(rows), len(keys)))
    for i, r in enumerate(rows):
        X[i, ix[keyf(r, "c")]] += 1
        X[i, ix[keyf(r, "r")]] -= 1
    return X, keys


X_free, k_free = onehot_diff(lambda r, s: r["ct"] if s == "c" else r["rt"])
X_ctx, k_ctx = onehot_diff(lambda r, s: (r["cls"], r["ct"] if s == "c" else r["rt"]))
w_free_full = fit_logistic(X_free)
ranking_free = sorted(zip(k_free, np.round(w_free_full, 3).tolist()), key=lambda x: -x[1])
same_type = sum(1 for r in rows if r["ct"] == r["rt"])


def feats(text, ntok):
    t = text
    s = t.strip()
    return {
        "log_tokens": math.log(max(1, ntok)),
        "has_qmark": float("?" in t),
        "ends_qmark_or_card": float(s.endswith("?") or s.endswith('?"}}</tool_call>')),
        "lead_ws": float(t[:1].isspace()) if t else 0.0,
        "trail_ws": float(t[-1:].isspace()) if t else 0.0,
        "log_newlines": math.log1p(t.count("\n")),
        "strict_one_line_call": float(bool(re.fullmatch(r'<tool_call>\{.*\}</tool_call>', t)) and "\n" not in t),
        "markdown": float("**" in t or "\n#" in t or t.startswith("#") or "```" in t),
        "null_arg": float(": null" in t),
        "tool_result_tag": float("<tool_result" in t),
    }


FEAT_SETS = {
    "length_only": ["log_tokens"],
    "qmark_only": ["has_qmark"],
    "format_only": ["lead_ws", "trail_ws", "log_newlines", "strict_one_line_call", "markdown", "null_arg", "tool_result_tag"],
    "format_plus_length": ["log_tokens", "lead_ws", "trail_ws", "log_newlines", "strict_one_line_call", "markdown", "null_arg",
                           "tool_result_tag"],
    "format_length_qmark": ["log_tokens", "has_qmark", "ends_qmark_or_card", "lead_ws", "trail_ws", "log_newlines",
                            "strict_one_line_call", "markdown", "null_arg", "tool_result_tag"],
}
F_c = [feats(r["chosen_text"], r["c_tok"]) for r in rows]
F_r = [feats(r["rej_text"], r["r_tok"]) for r in rows]
cv = {}
for name, cols in FEAT_SETS.items():
    X = np.array([[fc[c] - fr[c] for c in cols] for fc, fr in zip(F_c, F_r)])
    cv[name] = cv_score(X)
cv["action_type_prompt_free"] = cv_score(X_free)
cv["action_type_context_aware(A/B_approve/B_act)"] = cv_score(X_ctx)

# simple deterministic rules (no fitting)
def rule_acc(vals):
    return round(float(np.mean(vals)), 4)


q_rule = [1.0 if ("?" in r["chosen_text"]) > ("?" in r["rej_text"]) else (0.5 if ("?" in r["chosen_text"]) == ("?" in r["rej_text"]) else 0.0) for r in rows]
len_rule = [1.0 if r["c_tok"] < r["r_tok"] else (0.5 if r["c_tok"] == r["r_tok"] else 0.0) for r in rows]
ctx_rule = [1.0 if r["ct"] != r["rt"] else 0.5 for r in rows]   # an oracle that knows the row class orders every type-differing pair

# M1 policy samples on the same prompts (GRPO run, steps 0-29)
sc = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
key2row = {(r["scenario_id"], r["level"]): r for r in rows}
assert len(key2row) == len(rows)
m1 = defaultdict(Counter)
for s in sc:
    m1[(s["meta"]["scenario_id"], s["meta"]["level"])][s["components"]["decision"]] += 1
missing_keys = [k for k in m1 if k not in key2row]
CARD = ("approve_generic", "approve_informed", "approve_gold")
MSG = ("ask_generic", "ask_informed", "tell_informed", "statement")
a_card_prompts = [k for k, c in m1.items() if key2row[k]["twin"] == "A" and any(c[d] for d in CARD)]
a_card_samples = sum(sum(m1[k][d] for d in CARD) for k in a_card_prompts)
a_card_targeted = [k for k in a_card_prompts if key2row[k]["rt"] == "card"]
b_msg_prompts = [k for k, c in m1.items() if key2row[k]["twin"] == "B" and any(c[d] for d in MSG)]
b_msg_samples = sum(sum(m1[k][d] for d in MSG) for k in b_msg_prompts)
b_msg_targeted = [k for k in b_msg_prompts if key2row[k]["rt"] in ("question", "statement")]
b_msg_targeted_samples = sum(sum(m1[k][d] for d in MSG) for k in b_msg_targeted)
commit_samples = Counter()
for s in sc:
    d = s["components"]["decision"]
    if d.startswith("commit"):
        commit_samples[(s["meta"]["twin"][:1], d)] += 1
m1_dec = Counter((s["meta"]["twin"][:1], s["components"]["decision"]) for s in sc)
# does the pair's rejected type match the M1 policy's modal wrong type on that prompt?
modal_match = Counter()
for k, c in m1.items():
    r = key2row[k]
    wrong = Counter()
    for d, n in c.items():
        ok = (r["cls"] == "A" and d in ("ask_informed", "tell_informed")) or \
             (r["cls"] == "B_approve" and d in CARD) or (r["cls"] == "B_act" and d == "commit_gold")
        if not ok:
            t = "card" if d in CARD else "commit" if d.startswith("commit") else "lookup" if d == "lookup" else \
                "invalid" if d == "invalid" else "question" if d.startswith("ask") else "statement"
            wrong[t] += n
    if wrong:
        modal_match[(r["cls"], wrong.most_common(1)[0][0] == r["rt"])] += 1

out = {
    "token_stats_misaligned_rows": misaligned,
    "type_pairs_by_class": {c: dict(Counter(f"{r['ct']}>{r['rt']}" for r in rows if r["cls"] == c).most_common()) for c in ("A", "B_approve", "B_act")},
    "teaching_axes": dict(ax.most_common()),
    "commit_axis_by_level(True=X>commit)": {k: dict(v) for k, v in sorted(ax_level.items())},
    "malformed_rejected_calls": sum(r["rej_malformed_call"] for r in rows),
    "malformed_rejected_commits": sum(r["rej_malformed_commit"] for r in rows),
    "malformed_idx": [[r["idx"], r["scenario_id"], r["rt"], r["rej_dec"]] for r in rows if r["rej_malformed_call"]],
    "twin_levels_with_both": len(both), "A_rejected_type_equals_B_chosen_type": a_rej_eq_b_ch,
    "B_rejected_type_equals_A_chosen_type": b_rej_eq_a_ch,
    "same_action_type_pairs": same_type,
    "cv_accuracy_mean_std": cv,
    "prompt_free_type_ranking_fit_on_all": ranking_free,
    "rule_qmark": rule_acc(q_rule), "rule_shorter_wins": rule_acc(len_rule),
    "rule_context_oracle_on_types": rule_acc(ctx_rule),
    "chosen_has_qmark_by_class": {c: [sum("?" in r["chosen_text"] for r in rows if r["cls"] == c), sum(1 for r in rows if r["cls"] == c)] for c in ("A", "B_approve", "B_act")},
    "rejected_has_qmark": sum("?" in r["rej_text"] for r in rows),
    "rejected_longer_share": round(sum(r["r_tok"] > r["c_tok"] for r in rows) / len(rows), 4),
    "mean_tokens_chosen_rejected": [round(float(np.mean([r["c_tok"] for r in rows])), 1), round(float(np.mean([r["r_tok"] for r in rows])), 1)],
    "mean_tokens_observed_chosen_rejected": [round(float(np.mean([r["c_tok"] for r in rows if r["rej_source"] == "observed"])), 1),
                                              round(float(np.mean([r["r_tok"] for r in rows if r["rej_source"] == "observed"])), 1)],
    "rejected_over_200_tokens": sum(r["r_tok"] > 200 for r in rows), "rejected_max_tokens": max(r["r_tok"] for r in rows),
    "chosen_max_tokens": max(r["c_tok"] for r in rows),
    "format_counts": {
        "lead_ws_rej_vs_chosen": [sum(f["lead_ws"] for f in F_r), sum(f["lead_ws"] for f in F_c)],
        "trail_ws_rej_vs_chosen": [sum(f["trail_ws"] for f in F_r), sum(f["trail_ws"] for f in F_c)],
        "rej_tool_calls_multiline": [sum(1 for r in rows if r["rej_parse_kind"] == "tool_call" and "\n" in r["rej_text"].strip()),
                                     sum(1 for r in rows if r["rej_parse_kind"] == "tool_call")],
        "chosen_tool_calls_multiline": [sum(1 for r in rows if r["chosen_parse_kind"] == "tool_call" and "\n" in r["chosen_text"].strip()),
                                        sum(1 for r in rows if r["chosen_parse_kind"] == "tool_call")],
        "rej_not_strict_format": sum(not r["rej_strict_fmt"] for r in rows), "chosen_not_strict_format": sum(not r["chosen_strict_fmt"] for r in rows),
        "rej_null_args": sum(f["null_arg"] for f in F_r), "rej_tool_result_tag": sum(f["tool_result_tag"] for f in F_r),
    },
    "m1_cross_reference": {
        "keys_in_scores": len(m1), "keys_missing_from_dpo": missing_keys, "samples": len(sc),
        "decisions_by_twin": {f"{a}:{b}": n for (a, b), n in sorted(m1_dec.items())},
        "A_prompts_with_card_samples": len(a_card_prompts), "A_card_samples": a_card_samples,
        "A_card_prompts_whose_pair_rejects_a_card": len(a_card_targeted),
        "B_prompts_with_message_samples": len(b_msg_prompts), "B_message_samples": b_msg_samples,
        "B_message_prompts_whose_pair_rejects_a_message": len(b_msg_targeted), "their_samples": b_msg_targeted_samples,
        "commit_samples": {f"{a}:{b}": n for (a, b), n in sorted(commit_samples.items())},
        "pair_rejected_type_equals_M1_modal_wrong_type(class,match)": {f"{a}:{b}": n for (a, b), n in sorted(modal_match.items())},
    },
}
json.dump(out, open(os.path.join(OUT, "v3_contrasts.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
