"""A calibrated character-class -> Tiny Aya token-count model (no tokenizer is available locally; numpy least squares).

Calibration targets are EXACT token counts measured with the real tokenizer + chat template on the Spark
(experiments/dataset_audit/results/token_stats.jsonl): every row's templated prompt length and every target's plain length.
Features of a text: letter runs, letters, digits, ASCII punctuation, whitespace runs, newlines, non-ASCII characters; a templated
prompt adds one constant per message plus a global constant (the generation prompt / BOS). The per-row residual is then removed by
rescaling (positions inside one prompt are estimated proportionally to the model, and the whole prompt matches the exact count).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from common import ROOT, jl, target_text

FEATS = ("letter_runs", "letters", "digits", "punct", "ws_runs", "newlines", "nonascii")
_LR = re.compile(r"[A-Za-z]+")
_WS = re.compile(r"\s+")


def feats(t: str) -> np.ndarray:
    letters = sum(c.isascii() and c.isalpha() for c in t)
    digits = sum(c.isdigit() and c.isascii() for c in t)
    nonascii = sum(not c.isascii() for c in t)
    punct = sum(c.isascii() and not c.isalnum() and not c.isspace() for c in t)
    return np.array([len(_LR.findall(t)), letters, digits, punct, len(_WS.findall(t)), t.count("\n"), nonascii], float)


def _views():
    cur = os.path.join(ROOT, "dataset", "curated_v121")
    return {f"{v}.jsonl": jl(os.path.join(cur, f"{v}.jsonl")) for v in ("sft_train", "sft_val", "dpo_train", "grpo_train")}


def calibration():
    """X (n x (len(FEATS)+2)), y exact tokens, kind (prompt/target), meta."""
    stats = jl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))
    V = _views()
    X, y, kind, meta = [], [], [], []
    seen_prompt = set()
    for s in stats:
        r = V[s["file"]][s["i"]]
        assert r.get("scenario_id") == s["scenario_id"] and r.get("level") == s["level"], (s["file"], s["i"])
        pk = json.dumps(r["prompt"], ensure_ascii=False)
        if pk not in seen_prompt:
            seen_prompt.add(pk)
            f = sum((feats(m["content"]) for m in r["prompt"]), np.zeros(len(FEATS)))
            X.append(np.concatenate([f, [len(r["prompt"]), 1.0]]))
            y.append(s["prompt_tokens"])
            kind.append("prompt")
            meta.append((s["file"], s["i"]))
        for key in ("completion", "chosen", "rejected"):
            if key in r and s.get(key + "_tokens") is not None:
                X.append(np.concatenate([feats(r[key][0]["content"]), [0.0, 0.0]]))
                y.append(s[key + "_tokens"])
                kind.append("target")
                meta.append((s["file"], s["i"], key))
    return np.array(X), np.array(y, float), np.array(kind), meta


def fit(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w


_W = None


def weights():
    global _W
    if _W is None:
        X, y, _, _ = calibration()
        _W = fit(X, y)
    return _W


def est_text(t: str) -> float:
    w = weights()
    return float(feats(t) @ w[: len(FEATS)])


def est_prompt(msgs) -> float:
    w = weights()
    return float(sum(est_text(m["content"]) for m in msgs) + w[len(FEATS)] * len(msgs) + w[len(FEATS) + 1])


def cv_report(k=5, seed=0):
    X, y, kind, _ = calibration()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    pred = np.zeros_like(y)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        w = fit(X[tr], y[tr])
        pred[f] = X[f] @ w
    out = {}
    for kd in ("prompt", "target"):
        m = kind == kd
        ape = np.abs(pred[m] - y[m]) / np.maximum(1, y[m])
        out[kd] = {"n": int(m.sum()), "mape": round(float(ape.mean()), 4), "p95_ape": round(float(np.quantile(ape, 0.95)), 4),
                   "mean_abs_err_tokens": round(float(np.abs(pred[m] - y[m]).mean()), 1),
                   "max_abs_err_tokens": round(float(np.abs(pred[m] - y[m]).max()), 1)}
    w = fit(X, y)
    out["weights"] = dict(zip(list(FEATS) + ["per_message", "constant"], [round(float(x), 4) for x in w]))
    return out


if __name__ == "__main__":
    from common import dump
    rep = cv_report()
    print(json.dumps(rep, indent=1))
    print(dump("tokmodel_cv.json", rep))
