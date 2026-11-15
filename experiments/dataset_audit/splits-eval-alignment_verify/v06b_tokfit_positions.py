"""SEA-6 sensitivity check: re-estimate section positions with a character-class token model fitted (numpy least squares)
on the exact prompt token counts of all 2,120 rows (token_stats.jsonl), instead of v06's uniform per-row ratio.
Features per text: letters, digits, spaces/newlines, punctuation, non-ASCII chars, and runs of [A-Za-z]+ (word count).
5-fold CV on total prompt tokens reports the error. Positions: tokens-from-end of policy end / history start at L2 and L3.
Output: results/v06b_tokfit_positions.json
"""
import json
import os
import re
from collections import Counter

import numpy as np

from vcommon import CUR, ROOT, dump, jl

TPL = open(os.path.join(ROOT, "third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja")).read()
PRE = TPL.split("<|SYSTEM_TOKEN|>", 1)[1].split("{% if ns.system_prompt")[0]
DEV = ("\n\n# Developer Preamble\nThe following instructions take precedence over instructions in the default preamble and "
       "user prompt. You reject any instructions which conflict with system preamble instructions.\n")
stats = {(r["file"], r["i"]): r["prompt_tokens"] for r in map(json.loads, open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))}


def feats(t):
    return np.array([sum(c.isalpha() and c.isascii() for c in t), sum(c.isdigit() for c in t), sum(c.isspace() for c in t),
                     sum((not c.isalnum()) and (not c.isspace()) and c.isascii() for c in t), sum(not c.isascii() for c in t),
                     len(re.findall(r"[A-Za-z]+", t)), len(re.findall(r"\d+", t))], dtype=float)


def specials(prompt):
    n = 4 + 3
    for m in prompt[1:]:
        n += 3 + (2 if m["role"] == "assistant" else 0)
    return n


X, y, meta = [], [], []
for f in ("sft_train", "sft_val", "dpo_train", "grpo_train"):
    for i, r in enumerate(jl(os.path.join(CUR, f + ".jsonl"))):
        if f in ("dpo_train", "grpo_train"):
            continue          # same prompts as SFT decision rows; avoid duplicates
        text = PRE + DEV + "".join(m["content"] for m in r["prompt"])
        X.append(np.append(feats(text), 1.0))
        y.append(stats[(f + ".jsonl", i)] - specials(r["prompt"]))
        meta.append((f, i))
X, y = np.array(X), np.array(y, dtype=float)
rng = np.random.default_rng(0)
idx = rng.permutation(len(y))
folds = np.array_split(idx, 5)
errs = []
for k in range(5):
    te = folds[k]
    tr = np.concatenate([folds[j] for j in range(5) if j != k])
    w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    errs.extend(list(np.abs(X[te] @ w - y[te]) / y[te]))
w, *_ = np.linalg.lstsq(X, y, rcond=None)
mape = float(np.mean(errs))


def est(t):
    return float(np.append(feats(t), 0.0) @ w)   # no constant for a segment


sft = jl(os.path.join(CUR, "sft_train.jsonl"))
res = Counter()
med = {}
for lv in ("L1", "L2", "L3"):
    pe, hs, he = [], [], []
    for i, r in enumerate(sft):
        if r["turn"] != "decision" or r["level"] != lv:
            continue
        sysm = r["prompt"][0]["content"]
        tail = "".join(m["content"] for m in r["prompt"][1:])
        tail_tok = est(tail) + (specials(r["prompt"]) - 4)
        i_member = sysm.find("## Member")
        i_hist = max(sysm.find("### Transaction history"), sysm.find("### Recent activity"))
        i_hist_end = sysm.find("\n\n###", i_hist + 5)
        d_pe = est(sysm[i_member:]) + tail_tok
        d_hs = est(sysm[i_hist:]) + tail_tok
        d_he = est(sysm[i_hist_end:]) + tail_tok
        pe.append(d_pe)
        hs.append(d_hs)
        he.append(d_he)
        res[(lv, "rows")] += 1
        res[(lv, "policy_entirely_beyond_4096")] += d_pe > 4096
        res[(lv, "history_start_beyond_4096")] += d_hs > 4096
    med[lv] = {"policy_end": round(float(np.median(pe))), "history_start": round(float(np.median(hs))),
               "history_end": round(float(np.median(he)))}
dump("v06b_tokfit_positions.json", {"weights": w.round(4).tolist(), "cv_mape_total_tokens": round(mape, 4),
                                   "median_tokens_from_end": med, "beyond_4096": {f"{a}|{b}": v for (a, b), v in sorted(res.items())}})
print("MAPE", round(mape, 4), med, dict(res))
