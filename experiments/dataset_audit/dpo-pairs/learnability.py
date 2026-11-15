"""DPO-pairs audit, part 3: what can 1 epoch of DPO (31 steps, beta 0.1, LoRA r16, lr 5e-6) learn from these pairs?

(a) Update budget. With Adam a LoRA B entry moves at most ~lr per step, so sum(lr_t) bounds how far the adapter can travel. We take
    sum(lr) of M1 SFT and of M2 GRPO (first 25 steps) from their logged learning rates, the measured adapter sizes
    (experiments/training_eval/results/train_logs/adapter_effect_m2_ckpt25.json), and the DPO schedule that training/train_dpo.py
    gets from AlignTune (lr 5e-6 default, not overridden by run_pipeline.sh; lr_scheduler_type 'linear', warmup 0 per
    third_party/aligntune/src/aligntune/core/rl/trainer_base.py:181-196 and backends/trl/rl/dpo/dpo.py:107-109 -> CAVEATED,
    read from source, not from the run's log), and extrapolate the DPO adapter's size relative to SFT's.
(b) Shortcut test. A Bradley-Terry logistic model (numpy, no intercept, L2) scores each response from RESPONSE-ONLY features (no
    prompt) and is evaluated with 5-fold cross-validation grouped by pair_id: held-out share of pairs where score(chosen) >
    score(rejected). Feature sets: action type only (a prompt-independent action prior), format+length only, format+length+content
    (no action type), all response-only, and a twin-aware oracle (action type x twin, i.e. what reading the prompt correctly buys).
(c) Prompt-independent directions: for 'lower P(direct commit)', 'raise P(chat message)', 'raise P(approval card)', count pairs whose
    margin the direction raises vs lowers.
(d) Cross-reference with the M1 policy's own errors on the same training prompts (m2_grpo/scores.jsonl, 120 prompts x 8 samples):
    does the DPO pair for that (scenario, level) contrast the error M1 actually makes?
Outputs: results/learnability.json

    python3 experiments/dataset_audit/dpo-pairs/learnability.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TL = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs")


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


# ---------------- (a) update budget ----------------------------------------------------------------------------------------
sft_res = json.load(open(os.path.join(TL, "m1_sft", "result.json")))
sft_lr = [x["learning_rate"] for x in sft_res["log_history"] if "learning_rate" in x and "loss" in x]
grpo_res = json.load(open(os.path.join(TL, "m2_grpo", "result.json")))
grpo_lr = [x["learning_rate"] for x in grpo_res["log_history"] if "learning_rate" in x]
eff = json.load(open(os.path.join(TL, "adapter_effect_m2_ckpt25.json")))["adapters"]
sft_args = json.load(open(os.path.join(TL, "m1_sft", "args.json")))
n_pairs = sum(1 for _ in open(os.path.join(ROOT, "dataset", "curated_v121", "dpo_train.jsonl")))
dpo_steps = -(-n_pairs // 16)                       # batch 1 x grad-accum 16, 1 epoch, ceil
dpo_lr = 5e-6
dpo_sched_linear = [dpo_lr * (1 - t / dpo_steps) for t in range(dpo_steps)]    # HF linear, warmup 0
budget = {
    "sft": {"steps": len(sft_lr), "lr_peak": max(sft_lr), "sum_lr": sum(sft_lr), "total_dW_fro": eff["m1_sft_final"]["total_dW_fro"],
            "max_abs_B": eff["m1_sft_final"]["max_abs_B"]},
    "grpo_25": {"steps": 25, "lr_peak": max(grpo_lr[:25]), "sum_lr": sum(grpo_lr[:25]),
                "total_dW_fro": eff["m2_grpo_ckpt25"]["total_dW_fro"], "max_abs_B": eff["m2_grpo_ckpt25"]["max_abs_B"]},
    "dpo_planned": {"pairs": n_pairs, "steps": dpo_steps, "lr_peak": dpo_lr, "sum_lr_linear_no_warmup": sum(dpo_sched_linear),
                    "sum_lr_constant_upper_bound": dpo_lr * dpo_steps},
}
for k in ("sft", "grpo_25"):
    budget[k]["max_abs_B_over_sum_lr"] = budget[k]["max_abs_B"] / budget[k]["sum_lr"]
g_ratio_measured = budget["grpo_25"]["total_dW_fro"] / budget["sft"]["total_dW_fro"]
g_ratio_budget = budget["grpo_25"]["sum_lr"] / budget["sft"]["sum_lr"]
calib = g_ratio_measured / g_ratio_budget
budget["calibration"] = {"grpo_dW_ratio_measured": g_ratio_measured, "grpo_sum_lr_ratio": g_ratio_budget,
                         "dW_per_unit_lr_vs_sft": calib}
for name, s in (("linear_no_warmup", budget["dpo_planned"]["sum_lr_linear_no_warmup"]),
                ("constant_upper_bound", budget["dpo_planned"]["sum_lr_constant_upper_bound"])):
    budget["dpo_planned"][f"predicted_dW_ratio_vs_sft_{name}"] = calib * s / budget["sft"]["sum_lr"]
    budget["dpo_planned"][f"sum_lr_ratio_vs_sft_{name}"] = s / budget["sft"]["sum_lr"]
    budget["dpo_planned"][f"predicted_max_abs_B_{name}"] = budget["grpo_25"]["max_abs_B_over_sum_lr"] * s
# SFT: how far had SFT got when it had spent the same lr budget as the whole DPO run?
cum, step_at = 0.0, None
for i, x in enumerate(sft_lr):
    cum += x
    if cum >= budget["dpo_planned"]["sum_lr_linear_no_warmup"] and step_at is None:
        step_at = i + 1
sft_losses = [x["loss"] for x in sft_res["log_history"] if "loss" in x and "learning_rate" in x]
budget["sft_reference"] = {"sft_step_with_same_cum_lr_as_dpo": step_at, "sft_loss_steps_1_5": [round(v, 3) for v in sft_losses[:5]],
                           "sft_loss_mean_steps_1_3": round(float(np.mean(sft_losses[:3])), 3),
                           "sft_loss_mean_steps_8_12": round(float(np.mean(sft_losses[7:12])), 3),
                           "sft_cum_lr_first_10": [round(float(v), 7) for v in np.cumsum(sft_lr[:10])]}

# ---------------- (b) response-only Bradley-Terry shortcut test -------------------------------------------------------------
pairs = load(os.path.join(OUT, "pairs.jsonl"))
crow = pd.read_csv(os.path.join(OUT, "contrast_rows.csv"))
assert list(crow.idx) == [p["idx"] for p in pairs]
CANON = re.compile(r'<tool_call>\{"name": "[a-z_]+", "arguments": \{.*\}\}</tool_call>', re.S)
TYPES = ["message", "ui", "commit", "lookup", "bad"]


def act_type(a: str) -> str:
    if a.startswith("malformed"):
        return "bad"
    return {"message": "message", "ui": "ui", "commit": "commit", "lookup": "lookup"}.get(a, "bad")


def feats(text: str, a: str, n_tok: int) -> dict:
    t = text
    tool_like = "<tool_call" in t or '"arguments"' in t
    f = {f"type_{k}": float(act_type(a) == k) for k in TYPES}
    f.update({
        # format / length
        "lead_ws": float(t[:1].isspace()), "trail_ws": float(t[-1:].isspace()), "log_newlines": np.log1p(t.count("\n")),
        "multiline_tool": float(tool_like and "\n" in t.strip()), "canonical_call": float(bool(CANON.fullmatch(t))),
        "markdown": float(bool(re.search(r"\*\*|`|^#", t, re.M))), "null_args": float("null" in t), "log_tokens": np.log(n_tok),
        # content (no action type)
        "qmark": float("?" in t), "digit_share": sum(c.isdigit() for c in t) / max(1, len(t)),
        "currency": float(bool(re.search(r"[£$€₹]|\b(USD|GBP|EUR|INR)\b", t))),
        "greeting": float(bool(re.match(r"\W*(hi|hello|dear|hey)\b", t, re.I))),
        "claims_done": float(bool(re.search(r"\b(done|i'?ve|i have|has been|successfully|taken care)\b", t, re.I))),
        "confirm_word": float(bool(re.search(r"\b(confirm|sure|go ahead|proceed)\b", t, re.I))),
    })
    return f


SETS = {
    "action_type_only": [f"type_{k}" for k in TYPES],
    "format_length_only": ["lead_ws", "trail_ws", "log_newlines", "multiline_tool", "canonical_call", "markdown", "null_args",
                           "log_tokens"],
    "length_only": ["log_tokens"],
}
SETS["format_length_content_no_type"] = SETS["format_length_only"] + ["qmark", "digit_share", "currency", "greeting",
                                                                       "claims_done", "confirm_word"]
SETS["all_response_only"] = SETS["action_type_only"] + SETS["format_length_content_no_type"]

FC, FR = [], []
for p, c in zip(pairs, crow.itertuples()):
    FC.append(feats(p["chosen_text"], c.chosen_action, p["chosen_tokens"]))
    FR.append(feats(p["rejected_text"], c.rejected_action_true, p["rejected_tokens"]))
FC, FR = pd.DataFrame(FC), pd.DataFrame(FR)
twin = np.array([p["twin"] for p in pairs])
for k in TYPES:                                          # twin-aware oracle: action type x twin (needs the prompt)
    for tw in ("A", "B"):
        FC[f"{k}_x_{tw}"] = FC[f"type_{k}"] * (twin == tw)
        FR[f"{k}_x_{tw}"] = FR[f"type_{k}"] * (twin == tw)
SETS["oracle_type_x_twin"] = [f"{k}_x_{tw}" for k in TYPES for tw in ("A", "B")]
groups = np.array([int(hashlib.md5(p["pair_id"].encode()).hexdigest(), 16) % 5 for p in pairs])


def fit_bt(X, lam=1e-2, iters=50):
    """Logistic regression on difference vectors with symmetric augmentation, no intercept; Newton steps."""
    Xa = np.vstack([X, -X])
    y = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    Z = (Xa - 0) / sd
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w / sd


def cv(cols):
    Xd = (FC[cols].values - FR[cols].values).astype(float)
    correct = np.zeros(len(Xd))
    for k in range(5):
        tr, te = groups != k, groups == k
        w = fit_bt(Xd[tr])
        s = Xd[te] @ w
        correct[te] = (s > 1e-9) + 0.5 * (np.abs(s) <= 1e-9)
    w_all = fit_bt(Xd)
    return {"cv_pair_accuracy": round(float(correct.mean()), 3),
            "by_twin": {tw: round(float(correct[twin == tw].mean()), 3) for tw in ("A", "B")},
            "weights_full_fit": {c: round(float(v), 3) for c, v in zip(cols, w_all)}}, correct


shortcut = {}
correct_by_set = {}
for name, cols in SETS.items():
    shortcut[name], correct_by_set[name] = cv(cols)
# simple rules
rl = np.array([p["rejected_tokens"] > p["chosen_tokens"] for p in pairs])
shortcut["rule_rejected_is_longer"] = round(float(rl.mean()), 3)
qc = np.array(["?" in p["chosen_text"] for p in pairs])
qr = np.array(["?" in p["rejected_text"] for p in pairs])
shortcut["rule_question_mark_only"] = {"accuracy_ties_half": round(float(((qc & ~qr) + 0.5 * (qc == qr)).mean()), 3),
                                       "chosen_has_qmark": int(qc.sum()), "rejected_has_qmark": int(qr.sum()),
                                       "B_approve_chosen_has_qmark": int(sum(1 for p in pairs if p["expected"] == "approve"
                                                                             and "?" in p["chosen_text"])),
                                       "B_approve_n": int(sum(1 for p in pairs if p["expected"] == "approve"))}
shortcut["rule_rejected_is_commit_or_malformed_commit"] = round(float(crow.rejected_action_true.isin(
    ["commit", "malformed commit call read as message"]).mean()), 3)
fmt_bad = np.array([(p["rejected_lead_ws"] or p["rejected_trail_ws"] or not p["rejected_strict"] or
                     (p["rejected_kind"] == "tool_call" and p["rejected_newlines"] > 0) or
                     crow.rejected_action_true.iloc[i].startswith("malformed")) for i, p in enumerate(pairs)])
shortcut["rule_rejected_has_any_format_defect"] = round(float(fmt_bad.mean()), 3)
shortcut["format_only_correct_but_type_only_wrong"] = int(((correct_by_set["format_length_only"] == 1) &
                                                           (correct_by_set["action_type_only"] < 1)).sum())

# ---------------- (c) prompt-independent directions ------------------------------------------------------------------------
ca = crow.chosen_action.map(act_type)
ra = crow.rejected_action_true.map(act_type)
directions = {}
for name, k, sign in (("lower P(direct commit)", "commit", -1), ("raise P(chat message)", "message", +1),
                      ("raise P(approval card)", "ui", +1)):
    # margin = [log pi(c) - ...] - [log pi(r) - ...]; raising P(type k) raises the term of whichever side is of type k
    helps = int((((ca == k) & (ra != k)) if sign > 0 else ((ra == k) & (ca != k))).sum())
    hurts = int((((ra == k) & (ca != k)) if sign > 0 else ((ca == k) & (ra != k))).sum())
    directions[name] = {"pairs_margin_up": helps, "pairs_margin_down": hurts, "neutral": len(pairs) - helps - hurts}

# ---------------- (d) M1's own errors on the same prompts vs the DPO pair --------------------------------------------------
scores = load(os.path.join(TL, "m2_grpo", "scores.jsonl"))
m1 = collections.defaultdict(collections.Counter)
for s in scores:
    m1[(s["meta"]["scenario_id"], s["meta"]["level"])][s["components"]["decision"]] += 1
key2pair = {(p["scenario_id"], p["level"]): (p, c) for p, c in zip(pairs, crow.itertuples())}


def err_axis(twin_, dec):
    if twin_ == "A":
        if dec.startswith("approve"):
            return "A: approval card instead of ask"
        if dec.startswith("commit"):
            return "A: direct commit"
        if dec in ("ask_generic", "statement"):
            return "A: generic question / statement"
        return None if dec in ("ask_informed", "tell_informed") else f"A: {dec}"
    if dec in ("ask_generic", "ask_informed", "tell_informed", "statement"):
        return "B: message instead of proceeding"
    if dec.startswith("commit"):
        return None if dec == "commit_gold" else "B: direct commit"
    return None if dec.startswith("approve") else f"B: {dec}"


def pair_axis(p, c):
    r = act_type(c.rejected_action_true)
    if p["twin"] == "A":
        return {"ui": "A: approval card instead of ask", "commit": "A: direct commit", "message": "A: generic question / statement"}\
            .get(r, f"A: {r}")
    return {"message": "B: message instead of proceeding", "commit": "B: direct commit"}.get(r, f"B: {r}")


xref = collections.Counter()
xref_prompts = collections.defaultdict(set)
xref_examples = collections.defaultdict(list)
m1_err_total = collections.Counter()
for k, cnt in m1.items():
    if k not in key2pair:
        continue
    p, c = key2pair[k]
    pa = pair_axis(p, c)
    for dec, n in cnt.items():
        ax = err_axis(p["twin"], dec)
        if ax is None:
            continue
        m1_err_total[ax] += n
        hit = "pair contrasts this error" if ax == pa else "pair contrasts something else"
        xref[(ax, hit)] += n
        xref_prompts[(ax, hit)].add(k)
        if len(xref_examples[(ax, hit)]) < 3:
            xref_examples[(ax, hit)].append({"idx": p["idx"], "scenario_id": k[0], "level": k[1], "m1_samples": n,
                                             "dpo_rejected_decision": p["rejected_decision"]})
m1_prompts_in_dpo = sum(1 for k in m1 if k in key2pair)

res = {"update_budget": budget, "shortcut_bt": shortcut, "prompt_independent_directions": directions,
       "m1_error_crossref": {"m1_prompts": len(m1), "m1_prompts_with_dpo_pair": m1_prompts_in_dpo,
                             "m1_error_samples_by_axis": dict(m1_err_total),
                             "error_samples_by_axis_and_coverage": {f"{a} | {h}": n for (a, h), n in sorted(xref.items())},
                             "error_prompts_by_axis_and_coverage": {f"{a} | {h}": len(v) for (a, h), v in sorted(xref_prompts.items())},
                             "m1_decisions_all": dict(collections.Counter(s["components"]["decision"] for s in scores)),
                             "examples": {f"{a} | {h}": v for (a, h), v in sorted(xref_examples.items())}}}
json.dump(res, open(os.path.join(OUT, "learnability.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
print("LEARNABILITY_OK")
