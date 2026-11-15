"""Verifier step 8: is DPO-3's response-only "shortcut" independent of the action-type contrast (DPO-1), and does it order the
pairs that carry the ask-versus-proceed decision?

(a) incremental grouped-CV accuracy: action type alone vs action type + format/length features (numpy logistic, same CV as v3)
(b) the '?' rule and the shorter-wins rule on the decision-boundary pairs only:
    A: ask > approval card, B: proceed (card or commit) > question
(c) per-pair token share of the rejected side that is pure formatting (leading/trailing whitespace, newlines, indentation)
(d) first-token divergence: the pairs whose rejected side starts with whitespace, i.e. the first token DPO contrasts is "\n"
Output: results/v8_shortcut_vs_type.json
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.episode import is_question  # noqa: E402

rows = [json.loads(l) for l in open(os.path.join(OUT, "v1_rows.jsonl"))]
tok = [t for t in (json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))
       if t["file"] == "dpo_train.jsonl"]


def atype(kind, tool_kind, text):
    if kind == "invalid":
        return "invalid"
    if kind == "message":
        return "question" if is_question(text) else "statement"
    return {"ui": "card", "commit": "commit", "lookup": "lookup", "verify": "lookup"}.get(tool_kind, "invalid")


for r in rows:
    r["ct"] = atype(r["chosen_parse_kind"], r["chosen_tool_kind"], r["chosen_text"])
    r["rt"] = atype(r["rej_parse_kind"], r["rej_tool_kind"], r["rej_text"])
    r["cls"] = "A" if r["twin"] == "A" else ("B_act" if r["expected"] == "act" else "B_approve")
    r["c_tok"], r["r_tok"] = tok[r["idx"]]["chosen_tokens"], tok[r["idx"]]["rejected_tokens"]

pairs = sorted({r["pair_id"] for r in rows})


def folds(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    f = {pairs[p]: i % 5 for i, p in enumerate(perm)}
    return np.array([f[r["pair_id"]] for r in rows])


def fit(X, l2=1e-2, iters=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= lr * (-(X.T @ (1 - p)) / len(X) + l2 * w)
    return w


def cv(X, mask=None, seeds=range(10)):
    acc = []
    for sd in seeds:
        f = folds(sd)
        c = np.zeros(len(rows))
        for k in range(5):
            tr, te = f != k, f == k
            w = fit(X[tr])
            s = X[te] @ w
            c[te] = np.where(np.abs(s) < 1e-9, 0.5, (s > 0).astype(float))
        acc.append(c[mask].mean() if mask is not None else c.mean())
    return round(float(np.mean(acc)), 4)


TYPES = ["question", "statement", "card", "commit", "lookup", "invalid"]


def type_vec(t):
    v = np.zeros(len(TYPES))
    v[TYPES.index(t)] = 1
    return v


def fmt(text, ntok):
    s = text.strip()
    return np.array([math.log(max(1, ntok)), float(text[:1].isspace()), float(text[-1:].isspace()), math.log1p(text.count("\n")),
                     float(bool(re.fullmatch(r"<tool_call>\{.*\}</tool_call>", text)) and "\n" not in text),
                     float("**" in text or "```" in text), float(": null" in text)])


X_type = np.array([type_vec(r["ct"]) - type_vec(r["rt"]) for r in rows])
X_fmt = np.array([fmt(r["chosen_text"], r["c_tok"]) - fmt(r["rej_text"], r["r_tok"]) for r in rows])
X_q = np.array([[float("?" in r["chosen_text"]) - float("?" in r["rej_text"])] for r in rows])
boundary = np.array([(r["cls"] == "A" and r["rt"] == "card") or (r["cls"] != "A" and r["rt"] == "question") for r in rows])
commit_rej = np.array([r["rt"] == "commit" for r in rows])
same_type = np.array([r["ct"] == r["rt"] for r in rows])

q_rule = np.array([1.0 if ("?" in r["chosen_text"]) > ("?" in r["rej_text"]) else 0.5 if ("?" in r["chosen_text"]) == ("?" in r["rej_text"]) else 0.0 for r in rows])
len_rule = np.array([1.0 if r["c_tok"] < r["r_tok"] else 0.5 if r["c_tok"] == r["r_tok"] else 0.0 for r in rows])


def ws_tokens_share(text):
    # crude: characters that are whitespace-only formatting (leading/trailing ws + newline+indent runs) over all characters
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    inner = sum(len(m.group(0)) for m in re.finditer(r"\n[ \t]*", text.strip()))
    return (lead + trail + inner) / max(1, len(text))


res = {
    "n_boundary_pairs": int(boundary.sum()), "n_commit_rejected_pairs": int(commit_rej.sum()), "n_same_type_pairs": int(same_type.sum()),
    "cv_all_pairs": {"type": cv(X_type), "type+format/length": cv(np.hstack([X_type, X_fmt])),
                     "type+format/length+qmark": cv(np.hstack([X_type, X_fmt, X_q])), "format/length+qmark": cv(np.hstack([X_fmt, X_q]))},
    "cv_on_boundary_pairs": {"type(prompt-free)": cv(X_type, boundary), "format/length+qmark": cv(np.hstack([X_fmt, X_q]), boundary)},
    "cv_on_same_type_pairs": {"format/length+qmark": cv(np.hstack([X_fmt, X_q]), same_type)},
    "qmark_rule_on_boundary": round(float(q_rule[boundary].mean()), 4),
    "qmark_rule_on_commit_rejected": round(float(q_rule[commit_rej].mean()), 4),
    "shorter_rule_on_boundary": round(float(len_rule[boundary].mean()), 4),
    "shorter_rule_on_commit_rejected": round(float(len_rule[commit_rej].mean()), 4),
    "boundary_pairs_rej_tokens_mean": round(float(np.mean([r["r_tok"] for r, b in zip(rows, boundary) if b])), 1),
    "boundary_pairs_chosen_tokens_mean": round(float(np.mean([r["c_tok"] for r, b in zip(rows, boundary) if b])), 1),
    "rejected_starting_with_whitespace": sum(1 for r in rows if r["rej_text"][:1].isspace()),
    "rejected_formatting_char_share_mean": round(float(np.mean([ws_tokens_share(r["rej_text"]) for r in rows])), 4),
    "chosen_formatting_char_share_mean": round(float(np.mean([ws_tokens_share(r["chosen_text"]) for r in rows])), 4),
    "boundary_idx_examples": [[r["idx"], r["scenario_id"], r["level"], r["cls"], r["rt"]] for r, b in zip(rows, boundary) if b][:8],
}
json.dump(res, open(os.path.join(OUT, "v8_shortcut_vs_type.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
