"""V7 (verifier): one-view shortcut classifiers, re-implemented from scratch (numpy logistic regression, full-batch gradient
descent on standardised features, L2). Metric = within-twin accuracy: over matched (pair, level) twin rows of an evaluation set,
share where score(A) > score(B) (ties 0.5). CIs: bootstrap over source-post clusters (2,000 resamples).
Train rows = sft_train decision rows; val = sft_val decision rows; test / hard_test = cache/eval_rows.jsonl (v6).
CV on train: 5 folds grouped by pair_id (siblings of a held-out pair can be in training) and by source cluster (whole families out).

Views: request_bow (uni+bigrams of the request), result_structure (log1p counts of leaves/null/false/true/zero/empty + chars of the
last tool result), length_raw (log chars of the whole prompt incl. the P2 hint), length_nohint (hint stripped),
result_field_values (field=value tokens for null/bool/0/empty/short strings), hint (P2 flag).

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v7_classifiers.py -> results/v7_classifiers.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import HERE, T, curated, jl, leaves, nohint, parts, result_obj, save, splits, struct_counts, words  # noqa: E402

RNG = np.random.default_rng(20261114)


def bigr(t):
    w = words(t)
    return set(w) | {a + "_" + b for a, b in zip(w, w[1:])}


def fv_tokens(obj):
    out = set()
    if not isinstance(obj, (dict, list)):
        return out
    for p, v in leaves(obj):
        if v is None:
            out.add(f"{p}=null")
        elif isinstance(v, bool):
            out.add(f"{p}={str(v).lower()}")
        elif v in ([], {}):
            out.add(f"{p}=empty")
        elif isinstance(v, (int, float)) and v == 0:
            out.add(f"{p}=0")
        elif isinstance(v, str) and len(v.split()) <= 3 and not any(c.isdigit() for c in v):
            out.add(f"{p}={v.lower()}")
    return out


def views(r):
    sysm, req, calls, results = parts(r["prompt"])
    obj, raw = result_obj(results[-1]) if results else (None, "")
    sc = struct_counts(obj)
    full = sum(len(m["content"]) for m in r["prompt"])
    return {"request_bow": bigr(req),
            "result_structure": [math.log1p(sc[k]) for k in ("leaves", "null", "false", "true", "zero", "empty")] + [math.log1p(len(raw))],
            "length_raw": [math.log(full)],
            "length_nohint": [math.log(full - len(sysm) + len(nohint(sysm)))],
            "result_field_values": fv_tokens(obj),
            "hint": [1.0 if r["prompt_setting"] == "P2" else 0.0]}


def featurize(train_rows, rows, view):
    if isinstance(train_rows[0]["_v"][view], set):
        vocab = sorted({t for r in train_rows for t in r["_v"][view]})
        idx = {t: i for i, t in enumerate(vocab)}

        def f(rs):
            X = np.zeros((len(rs), len(vocab)))
            for i, r in enumerate(rs):
                for t in r["_v"][view]:
                    if t in idx:
                        X[i, idx[t]] = 1.0
            return X
    else:
        def f(rs):
            return np.array([r["_v"][view] for r in rs], dtype=float)
    Xtr, X = f(train_rows), f(rows)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (X - mu) / sd


def fit(X, y, lam=1.0, iters=1500, lr=0.1):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        w -= lr * (X.T @ g / n + lam * w / n)
        b -= lr * g.mean()
    return w, b


def within_twin(rows, scores):
    """list of (cluster, value) per matched (pair, level) twin"""
    d = defaultdict(dict)
    for r, s in zip(rows, scores):
        d[(r["pair_id"], r["level"])][T(r)] = (s, r["_cluster"])
    out = []
    for k, x in d.items():
        if "A" in x and "B" in x:
            sa, sb = x["A"][0], x["B"][0]
            out.append((x["A"][1], 1.0 if sa > sb + 1e-12 else (0.5 if abs(sa - sb) <= 1e-12 else 0.0)))
    return out


def ci(vals, n=2000):
    byc = defaultdict(list)
    for c, v in vals:
        byc[c].append(v)
    cl = sorted(byc)
    pt = float(np.mean([v for _, v in vals]))
    bs = []
    for _ in range(n):
        pick = RNG.choice(len(cl), len(cl))
        x = [v for k in pick for v in byc[cl[k]]]
        bs.append(np.mean(x))
    return {"acc": round(pt, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "n_twins": len(vals), "n_clusters": len(cl)}


def main():
    pair_split, cluster = splits()
    tr = [r for r in curated("sft_train") if r["turn"] == "decision" and T(r) in "AB"]
    va = [r for r in curated("sft_val") if r["turn"] == "decision" and T(r) in "AB"]
    ev = jl(os.path.join(HERE, "cache", "eval_rows.jsonl"))
    te = [r for r in ev if r["split"] == "test"]
    ht = [r for r in ev if r["split"] == "hard_test"]
    for rs in (tr, va):
        for r in rs:
            r["_cluster"] = cluster.get(r["pair_id"], r["pair_id"])
    for r in ev:
        r["_cluster"] = r["cluster"]
    for r in tr + va + ev:
        r["_v"] = views(r)
    out = {"n_rows": {"train": len(tr), "val": len(va), "test": len(te), "hard_test": len(ht)}}
    y = np.array([1.0 if T(r) == "A" else 0.0 for r in tr])
    pids = sorted({r["pair_id"] for r in tr})
    cls = sorted({r["_cluster"] for r in tr})
    RNG.shuffle(pids)
    RNG.shuffle(cls)
    fold_pair = {p: i % 5 for i, p in enumerate(pids)}
    fold_cl = {c: i % 5 for i, c in enumerate(cls)}
    for view in ("request_bow", "result_structure", "length_raw", "length_nohint", "result_field_values", "hint"):
        res = {}
        Xtr, _ = featurize(tr, tr, view)
        w, b = fit(Xtr, y)
        res["in_sample"] = ci(within_twin(tr, Xtr @ w + b))
        if Xtr.shape[1] == 1:
            res["coef_sign"] = float(np.sign(w[0]))
        for name, fold in (("cv_pair_grouped", lambda r: fold_pair[r["pair_id"]]), ("cv_cluster_grouped", lambda r: fold_cl[r["_cluster"]])):
            sc = np.zeros(len(tr))
            for k in range(5):
                trn = [r for r in tr if fold(r) != k]
                tst_idx = [i for i, r in enumerate(tr) if fold(r) == k]
                Xa, Xb = featurize(trn, [tr[i] for i in tst_idx], view)
                ya = np.array([1.0 if T(r) == "A" else 0.0 for r in trn])
                w2, b2 = fit(Xa, ya)
                sc[tst_idx] = Xb @ w2 + b2
            res[name] = ci(within_twin(tr, sc))
        for name, rs in (("val", va), ("test", te), ("hard_test", ht)):
            Xa, Xb = featurize(tr, rs, view)
            res[name] = ci(within_twin(rs, Xb @ w + b))
        out[view] = res
        print(view, res)
    # model-free: share of twins where A's quantity is larger (prompt chars, last-result chars), per set, rows and pairs
    mf = {}
    for name, rs in (("train", tr), ("val", va), ("test", te), ("hard_test", ht)):
        d = defaultdict(dict)
        for r in rs:
            d[(r["pair_id"], r["level"])][T(r)] = r
        e = {"prompt_nohint_same_setting": [0, 0, 0], "last_result_chars": [0, 0, 0]}
        per_pair = defaultdict(list)
        for (pid, lv), x in d.items():
            if "A" not in x or "B" not in x:
                continue
            a, b = x["A"], x["B"]
            la = math.exp(a["_v"]["length_nohint"][0])
            lb = math.exp(b["_v"]["length_nohint"][0])
            if a["prompt_setting"] == b["prompt_setting"]:
                e["prompt_nohint_same_setting"][0 if la > lb else (1 if lb > la else 2)] += 1
            ra = parts(a["prompt"])[3]
            rb = parts(b["prompt"])[3]
            ca = len(ra[-1]) if ra else 0
            cb = len(rb[-1]) if rb else 0
            e["last_result_chars"][0 if ca > cb else (1 if cb > ca else 2)] += 1
            per_pair[pid].append(ca - cb)
        e = {k: {"A>B": v[0], "B>A": v[1], "tie": v[2]} for k, v in e.items()}
        e["last_result_chars_pairs"] = {"A>B": sum(np.mean(v) > 0 for v in per_pair.values()),
                                        "B>A": sum(np.mean(v) < 0 for v in per_pair.values()),
                                        "tie": sum(np.mean(v) == 0 for v in per_pair.values())}
        mf[name] = e
    out["model_free_length"] = mf
    print(mf)
    save(out, "v7_classifiers.json")


if __name__ == "__main__":
    main()
