"""Step 3 (brief items 4-6): shortcut classifiers. Can a simple model tell twin A (ask) from twin B (act) from ONE non-causal or
partial view of the decision prompt? numpy only (L2-regularised logistic regression, Newton steps with the Woodbury identity).

Rows (decision turns only; pushback rows excluded):
  train      dataset/curated_v121/sft_train.jsonl decision rows (= the DPO / GRPO prompts)
  val        dataset/curated_v121/sft_val.jsonl decision rows
  test, hard_test, hard_val, hard_train   dataset/eval/<name>_*.jsonl specs rendered exactly like training rows
             (training/build_datasets.prefix_messages, levels L0r/L1/L2 + L3 when a noisy message exists, prompt P0)
Feature views (the P2 hint text is removed from every text view; only the 'hint' view sees it):
  request_bow, request_bow_idmasked (identifiers -> <id>), result_bow (last tool result), result_field_values (field=value for
  null/bool/0/empty/short strings), result_keys (JSON paths present), result_structure (counts of leaves/nulls/false/true/0/empty
  + chars), call_bow (the agent's own look-up call(s)), length (log prompt chars), env_level (one-hot), hint (P2 flag),
  member_bow (member block of the system prompt), cue_counts (cue words per component), full_bow (whole prompt).
Metrics: accuracy, AUC, within-twin accuracy (share of (pair, level) twin pairs where the A row scores higher than the B row; ties
0.5), cluster-bootstrap 95 % CIs (2,000 resamples of source-post clusters), grouped 5-fold CV on train (folds = source clusters).
Item 5 (request alone) is also measured at the scenario level (one row per spec).

python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s3_shortcut_clf.py
-> results/s3_classifiers.json, results/s3_request_only.json, results/s3_predictions.jsonl, cache/rendered_eval_rows.jsonl
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, CUE_RX, RES, ROOT, components, eval_specs, flatten, generic_path, jdump, jl_dump,  # noqa: E402
                    load_jsonl, load_rows, load_specs, member_block, parse_result, strip_hint, twin_letter, words)

IDENT = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*|@[\w.-]+|[$€£₹]?\w*\d[\w,.\-/:]*|\b[\w-]+\.(?:com|net|org|pro|in|store|dev|de|"
                   r"ph|io|co|uk|app)\b")
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEED = 20261114


# ----------------------------------------------------------------------------------------------------------------- rendering
def render_eval_rows():
    path = os.path.join(CACHE, "rendered_eval_rows.jsonl")
    if os.path.exists(path):
        return load_jsonl(path)
    from training.build_datasets import prefix_messages
    from wtm.spec import load_env, load_member
    out = []
    envs, mems = {}, {}
    for name in ("test", "hard_test", "hard_val", "hard_train"):
        for s in eval_specs(name):
            if twin_letter(s["twin"]) == "P":
                continue
            env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
            mem = mems.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
            for lv in ("L0r", "L1", "L2", "L3"):
                if lv == "L3" and not s.get("user_message_noisy"):
                    continue
                system, user, msgs = prefix_messages(env, mem, s, lv, "P0")
                out.append({"split": name, "scenario_id": s["id"], "pair_id": s["pair_id"], "twin": s["twin"],
                            "env_id": s["env_id"], "level": lv, "prompt_setting": "P0",
                            "cluster": s.get("source", {}).get("post_id", s["pair_id"]),
                            "prompt": [{"role": "system", "content": system}] + msgs})
    jl_dump(out, path)
    return out


# ----------------------------------------------------------------------------------------------------------------- features
def last_result(prompt):
    comps = components(prompt)
    if not comps["results"]:
        return None, ""
    name, obj = parse_result(comps["results"][-1])
    m = re.search(r">\n(.*)\n</tool_result>", comps["results"][-1], re.S)
    return obj, (m.group(1) if m else comps["results"][-1])


def bigrams(toks):
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


def field_value_tokens(obj):
    out = set()
    if not isinstance(obj, (dict, list)):
        return out
    for p, v in flatten(obj).items():
        g = generic_path(p)
        if v is None:
            out.add(f"{g}=null")
        elif isinstance(v, bool):
            out.add(f"{g}={str(v).lower()}")
        elif v == [] or v == {}:
            out.add(f"{g}=empty")
        elif isinstance(v, (int, float)) and v == 0:
            out.add(f"{g}=0")
        elif isinstance(v, str) and len(v.split()) <= 3 and not any(ch.isdigit() for ch in v):
            out.add(f"{g}={v.lower()}")
    return out


def structure_vec(obj, raw):
    if isinstance(obj, (dict, list)):
        f = flatten(obj)
        vals = list(f.values())
    else:
        vals = []
    return [math.log1p(len(vals)), math.log1p(sum(v is None for v in vals)), math.log1p(sum(v in ([], {}) for v in vals)),
            math.log1p(sum(v is False for v in vals)), math.log1p(sum(v is True for v in vals)),
            math.log1p(sum((not isinstance(v, bool)) and isinstance(v, (int, float)) and v == 0 for v in vals)),
            math.log1p(len(raw))]


def row_views(r):
    p = r["prompt"]
    comps = components(p)
    system = strip_hint(comps["system"])
    obj, raw = last_result(p)
    req = comps["request"]
    calls = " ".join(comps["calls"])
    mb = member_block(system)
    full = "\n".join([system, req] + [m["content"] for m in p[2:]])
    cue = []
    for text in (mb, req, calls, " ".join(comps["results"])):
        cue += [math.log1p(len(rx.findall(text))) for rx in CUE_RX.values()]
    return {
        "request_bow": set(bigrams(words(req))),
        "request_bow_idmasked": set(bigrams(words(IDENT.sub(" idtoken ", req)))),
        "result_bow": set(words(raw)),
        "result_field_values": field_value_tokens(obj),
        "result_keys": {generic_path(k) for k in flatten(obj)} if isinstance(obj, (dict, list)) else set(),
        "call_bow": set(bigrams(words(calls))),
        "member_bow": set(words(mb)),
        "full_bow": set(words(full)),
        "result_structure": structure_vec(obj, raw),
        "length": [math.log(sum(len(m["content"]) for m in p))],
        "env_level": {f"{r['env_id']}|{r['level']}", f"env={r['env_id']}", f"level={r['level']}"},
        "hint": [1.0 if r.get("prompt_setting") == "P2" else 0.0],
        "cue_counts": cue,
    }


SPARSE = ["request_bow", "request_bow_idmasked", "result_bow", "result_field_values", "result_keys", "call_bow", "member_bow",
          "full_bow", "env_level"]
DENSE = ["result_structure", "length", "hint", "cue_counts"]
VIEW_NOTE = {
    "request_bow": "member request words (partly causal: some decisive facts sit in the request)",
    "request_bow_idmasked": "member request with identifiers masked (wording only; non-causal unless the wording is the clash)",
    "result_bow": "last tool result words (causal view: most decisive facts live here)",
    "result_field_values": "field=value tokens for null/bool/0/empty/short strings in the last result (verdict-like raw fields)",
    "result_keys": "JSON paths present in the last result, no values (schema only; non-causal)",
    "result_structure": "counts of leaves/null/empty/false/true/0 + chars of the last result (non-causal)",
    "call_bow": "the agent's own look-up call text (teacher-forced prefix)",
    "length": "log prompt characters (non-causal)",
    "env_level": "environment x realism level one-hot (non-causal)",
    "hint": "P2 hint present (non-causal)",
    "member_bow": "member block of the system prompt (mostly identical between twins)",
    "cue_counts": "cue-word counts per component (non-causal heuristics)",
    "full_bow": "whole prompt unigrams (upper bound for a lexical model)",
}


def build_matrix(views, view, vocab=None, stats=None):
    if view in SPARSE:
        if vocab is None:
            df = Counter(t for v in views for t in v[view])
            vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= 2))}
        X = np.zeros((len(views), len(vocab) + 1), dtype=np.float64)
        for i, v in enumerate(views):
            for t in v[view]:
                j = vocab.get(t)
                if j is not None:
                    X[i, j] = 1.0
        X[:, -1] = 1.0
        return X, vocab, None
    D = np.array([v[view] for v in views], dtype=np.float64)
    if stats is None:
        mu, sd = D.mean(0), D.std(0)
        sd[sd == 0] = 1.0
        stats = (mu, sd)
    D = (D - stats[0]) / stats[1]
    return np.hstack([D, np.ones((len(views), 1))]), None, stats


# ----------------------------------------------------------------------------------------------------------------- model
def fit_lr(X, y, lam, iters=40):
    n, d = X.shape
    w = np.zeros(d)
    K = X @ X.T
    ln = lam * n

    def loss(w):
        z = X @ w
        return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w)
    cur = loss(w)
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) / n + lam * w                      # gradient of mean loss + penalty
        Wd = np.clip(p * (1 - p), 1e-6, None) / n
        # H = X^T diag(Wd) X + lam I ; solve with Woodbury in n dims
        M = np.diag(lam / Wd) + K
        step = (g - X.T @ np.linalg.solve(M, X @ g)) / lam
        t = 1.0
        while t > 1e-4:
            nw = w - t * step
            nl = loss(nw)
            if nl <= cur:
                break
            t /= 2
        if cur - nl < 1e-10:
            w, cur = nw, nl
            break
        w, cur = nw, nl
    _ = ln
    return w


def predict(X, w):
    return 1 / (1 + np.exp(-(X @ w)))


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_pairs(meta, s):
    """[(cluster, value)] for every (pair, level) with both twins; value 1 if A scores higher, 0.5 tie, 0 otherwise."""
    idx = defaultdict(dict)
    for m, sc in zip(meta, s):
        idx[(m["pair_id"], m["level"])][m["t"]] = (sc, m["cluster"])
    out = []
    for k, d in idx.items():
        if "A" in d and "B" in d:
            a, b = d["A"][0], d["B"][0]
            out.append((d["A"][1], 1.0 if a > b + 1e-12 else 0.5 if abs(a - b) <= 1e-12 else 0.0))
    return out


def metrics(meta, y, s, boot=2000, seed=SEED):
    y, s = np.asarray(y), np.asarray(s)
    acc = float(np.mean((s > 0.5) == (y == 1)))
    wp = within_pairs(meta, s)
    res = {"n": int(len(y)), "acc": round(acc, 3), "auc": round(auc(y, s), 3),
           "within_twin_acc": round(float(np.mean([v for _, v in wp])), 3) if wp else None, "n_twin_pairs": len(wp)}
    # cluster bootstrap
    clusters = sorted({m["cluster"] for m in meta})
    if boot and len(clusters) > 1:
        rng = np.random.default_rng(seed)
        cl_idx = defaultdict(list)
        for i, m in enumerate(meta):
            cl_idx[m["cluster"]].append(i)
        wp_by_cl = defaultdict(list)
        for c, v in wp:
            wp_by_cl[c].append(v)
        accs, wps = [], []
        for _ in range(boot):
            pick = rng.choice(len(clusters), len(clusters), replace=True)
            ii = [i for k in pick for i in cl_idx[clusters[k]]]
            accs.append(np.mean((s[ii] > 0.5) == (y[ii] == 1)))
            vv = [v for k in pick for v in wp_by_cl[clusters[k]]]
            if vv:
                wps.append(np.mean(vv))
        res["acc_ci95"] = [round(float(np.percentile(accs, 2.5)), 3), round(float(np.percentile(accs, 97.5)), 3)]
        if wps:
            res["within_twin_acc_ci95"] = [round(float(np.percentile(wps, 2.5)), 3), round(float(np.percentile(wps, 97.5)), 3)]
        res["n_clusters"] = len(clusters)
    return res


def folds_by_cluster(meta, k=5, key="cluster"):
    cl = sorted({m[key] for m in meta})
    fold_of = {c: int(hashlib.sha256(c.encode()).hexdigest(), 16) % k for c in cl}
    return [fold_of[m[key]] for m in meta]


def oof_predict(views, y, view, folds, lam):
    oof = np.zeros(len(y))
    for k in sorted(set(folds)):
        tr = [i for i, f in enumerate(folds) if f != k]
        te = [i for i, f in enumerate(folds) if f == k]
        Xtr, vocab, stats = build_matrix([views[i] for i in tr], view)
        Xte, _, _ = build_matrix([views[i] for i in te], view, vocab, stats)
        oof[te] = predict(Xte, fit_lr(Xtr, y[tr], lam))
    return oof


def model_free_asymmetry(metas, views):
    """share of (pair, level) twin pairs where the A row has MORE of x than the B row (ties counted separately)."""
    names = ["result_leaves", "result_nulls", "result_empties", "result_falses", "result_trues", "result_zeros", "result_chars"]
    out = {}
    for split in metas:
        idx = defaultdict(dict)
        for m, v in zip(metas[split], views[split]):
            idx[(m["pair_id"], m["level"])][m["t"]] = (m, v)
        pairs = [(d["A"], d["B"]) for d in idx.values() if "A" in d and "B" in d]
        tab = {}
        feats = {n: (lambda v, j=j: v["result_structure"][j]) for j, n in enumerate(names)}
        feats["prompt_chars"] = lambda v: v["length"][0]
        feats["cue_words_in_results"] = lambda v: sum(v["cue_counts"][3 * len(CUE_RX): 4 * len(CUE_RX)])
        for n, fn in feats.items():
            d = [fn(a[1]) - fn(b[1]) for a, b in pairs]
            tab[n] = {"A_more": sum(x > 1e-12 for x in d), "B_more": sum(x < -1e-12 for x in d),
                      "equal": sum(abs(x) <= 1e-12 for x in d), "n": len(d)}
            nz = tab[n]["A_more"] + tab[n]["B_more"]
            tab[n]["share_A_more_among_unequal"] = round(tab[n]["A_more"] / nz, 3) if nz else None
        # same prompt setting only for prompt length (the hint adds ~45 tokens)
        d = [a[1]["length"][0] - b[1]["length"][0] for a, b in pairs if a[0]["ps"] == b[0]["ps"]]
        tab["prompt_chars_same_setting"] = {"A_more": sum(x > 0 for x in d), "B_more": sum(x < 0 for x in d),
                                            "equal": sum(x == 0 for x in d), "n": len(d)}
        out[split] = tab
    return out


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    specs, origin = load_specs()
    sp = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    pair_cluster = {p: c for c, v in sp["clusters"].items() for p in v["pairs"]}

    def meta_of(r, split):
        return {"split": split, "scenario_id": r["scenario_id"], "pair_id": r["pair_id"], "t": twin_letter(r["twin"]),
                "level": r["level"], "ps": r.get("prompt_setting"), "env": r["env_id"],
                "cluster": pair_cluster.get(r["pair_id"]) or r.get("cluster") or r["pair_id"],
                "row": [r.get("_file"), r.get("_i")] if "_i" in r else [split, r["scenario_id"], r["level"]]}

    sets = {}
    for split, fname in (("train", "sft_train"), ("val", "sft_val")):
        rows = [r for r in load_rows(fname) if r["turn"] == "decision" and twin_letter(r["twin"]) != "P"]
        sets[split] = rows
    ev = render_eval_rows()
    for name in ("test", "hard_test", "hard_val", "hard_train"):
        sets[name] = [r for r in ev if r["split"] == name]
    metas = {k: [meta_of(r, k) for r in v] for k, v in sets.items()}
    views = {k: [row_views(r) for r in v] for k, v in sets.items()}
    ys = {k: np.array([1.0 if m["t"] == "A" else 0.0 for m in metas[k]]) for k in sets}
    print({k: len(v) for k, v in sets.items()}, {k: int(ys[k].sum()) for k in ys})

    results, preds = {}, []
    folds = folds_by_cluster(metas["train"])                 # held-out source clusters (new seed + all its variants)
    pair_folds = folds_by_cluster(metas["train"], key="pair_id")   # held-out pairs; sibling variants of the same seed stay in
    for view in SPARSE + DENSE:
        # lambda by grouped (cluster) CV AUC; ties -> the stronger penalty
        best = None
        cv_scores = {}
        for lam in sorted(LAMBDAS, reverse=True):
            oof = oof_predict(views["train"], ys["train"], view, folds, lam)
            a = auc(ys["train"], oof)
            cv_scores[lam] = (a, oof)
            if best is None or a > cv_scores[best][0] + 1e-9:
                best = lam
        Xtr, vocab, stats = build_matrix(views["train"], view)
        w = fit_lr(Xtr, ys["train"], best)
        pair_oof = oof_predict(views["train"], ys["train"], view, pair_folds, best)
        res = {"note": VIEW_NOTE[view], "lambda": best, "n_features": int(Xtr.shape[1] - 1),
               "cv_auc_by_lambda": {str(l): round(v[0], 4) for l, v in cv_scores.items()},
               "train_fit": metrics(metas["train"], ys["train"], predict(Xtr, w), boot=0),
               "train_grouped_cv": metrics(metas["train"], ys["train"], cv_scores[best][1]),
               "train_pair_grouped_cv": metrics(metas["train"], ys["train"], pair_oof)}
        for m, sc, sp_ in zip(metas["train"], cv_scores[best][1], pair_oof):
            preds.append({"view": view, "split": "train_oof", **{k: m[k] for k in ("scenario_id", "level", "ps", "t")},
                          "row": m["row"], "p_A_cluster_oof": round(float(sc), 4), "p_A_pair_oof": round(float(sp_), 4)})
        for split in ("val", "test", "hard_test", "hard_val", "hard_train"):
            X, _, _ = build_matrix(views[split], view, vocab, stats)
            s = predict(X, w)
            res[split] = metrics(metas[split], ys[split], s)
            if view in ("request_bow", "request_bow_idmasked", "result_bow", "result_field_values", "result_structure",
                        "length", "full_bow", "call_bow"):
                for m, sc in zip(metas[split], s):
                    preds.append({"view": view, **{k: m[k] for k in ("split", "scenario_id", "level", "ps", "t")},
                                  "row": m["row"], "p_A": round(float(sc), 4)})
        if vocab is not None:
            inv = {i: t for t, i in vocab.items()}
            order = np.argsort(w[:-1])
            res["top_A_features"] = [[inv[i], round(float(w[i]), 3)] for i in order[::-1][:15]]
            res["top_B_features"] = [[inv[i], round(float(w[i]), 3)] for i in order[:15]]
        results[view] = res

        def fm(x):
            return f"{x['auc']:.2f}/{x['within_twin_acc']}"
        print(f"{view:21s} lam={best:<7} fit={fm(res['train_fit'])} clusterCV={fm(res['train_grouped_cv'])} "
              f"pairCV={fm(res['train_pair_grouped_cv'])} val={fm(res['val'])} test={fm(res['test'])} "
              f"hard_test={fm(res['hard_test'])} hard_train={fm(res['hard_train'])}")
    jdump(results, os.path.join(RES, "s3_classifiers.json"))
    jl_dump(preds, os.path.join(RES, "s3_predictions.jsonl"))
    asym = model_free_asymmetry(metas, views)
    jdump(asym, os.path.join(RES, "s3_model_free_asymmetry.json"))
    for split, tab in asym.items():
        print(split, {k: (v.get("A_more"), v.get("B_more"), v.get("equal")) for k, v in tab.items()})

    # ---------------------------------------------------------------- item 5: request text alone, scenario level
    split_of = {p: k for k in ("train", "val", "test") for p in sp[k]}
    groups = defaultdict(list)
    for s in specs.values():
        if twin_letter(s["twin"]) == "P":
            continue
        g = split_of.get(s["pair_id"]) or origin[s["id"]]
        groups[g].append(s)
    req_out = {"pairs_with_different_request_by_split": {}, "scenario_level": {}}
    for g, ss in groups.items():
        by = defaultdict(dict)
        for s in ss:
            by[s["pair_id"]][twin_letter(s["twin"])] = s["user_message"]
        full = [d for d in by.values() if "A" in d and "B" in d]
        req_out["pairs_with_different_request_by_split"][g] = {"pairs": len(full), "different": sum(d["A"] != d["B"] for d in full),
                                                               "share": round(sum(d["A"] != d["B"] for d in full) / max(1, len(full)), 3)}
    def sv(s):
        return {"request_bow": set(bigrams(words(s["user_message"]))),
                "request_bow_idmasked": set(bigrams(words(IDENT.sub(" idtoken ", s["user_message"]))))}
    for view in ("request_bow", "request_bow_idmasked"):
        tr = groups["train"]
        Vtr = [sv(s) for s in tr]
        ytr = np.array([1.0 if twin_letter(s["twin"]) == "A" else 0.0 for s in tr])
        Xtr, vocab, _ = build_matrix(Vtr, view)
        mtr = [{"pair_id": s["pair_id"], "level": "-", "t": twin_letter(s["twin"]),
                "cluster": pair_cluster.get(s["pair_id"], s["pair_id"])} for s in tr]
        fo = folds_by_cluster(mtr)
        fp = folds_by_cluster(mtr, key="pair_id")
        best, besta, oof_best = None, None, None
        for lam in sorted(LAMBDAS, reverse=True):
            oof = oof_predict(Vtr, ytr, view, fo, lam)
            a = auc(ytr, oof)
            if besta is None or a > besta + 1e-9:
                best, besta, oof_best = lam, a, oof
        w = fit_lr(Xtr, ytr, best)
        out = {"lambda": best, "train_fit": metrics(mtr, ytr, predict(Xtr, w), boot=0),
               "train_grouped_cv": metrics(mtr, ytr, oof_best),
               "train_pair_grouped_cv": metrics(mtr, ytr, oof_predict(Vtr, ytr, view, fp, best))}
        inv = {i: t for t, i in vocab.items()}
        order = np.argsort(w[:-1])
        out["top_A_features"] = [[inv[i], round(float(w[i]), 3)] for i in order[::-1][:20]]
        out["top_B_features"] = [[inv[i], round(float(w[i]), 3)] for i in order[:20]]
        for g in ("val", "test", "hard_test", "hard_val", "hard_train"):
            ss = groups[g]
            X, _, _ = build_matrix([sv(s) for s in ss], view, vocab)
            y = np.array([1.0 if twin_letter(s["twin"]) == "A" else 0.0 for s in ss])
            m = [{"pair_id": s["pair_id"], "level": "-", "t": twin_letter(s["twin"]),
                  "cluster": pair_cluster.get(s["pair_id"]) or s.get("source", {}).get("post_id", s["pair_id"])} for s in ss]
            sc = predict(X, w)
            out[g] = metrics(m, y, sc)
            # within-twin accuracy on pairs whose requests differ (identical requests are ties by construction)
            by = defaultdict(dict)
            for mm, s_, v in zip(m, ss, sc):
                by[mm["pair_id"]][mm["t"]] = (v, s_["user_message"])
            diff = [1.0 if d["A"][0] > d["B"][0] else 0.5 if d["A"][0] == d["B"][0] else 0.0 for d in by.values()
                    if "A" in d and "B" in d and d["A"][1] != d["B"][1]]
            out[g]["within_twin_acc_on_pairs_with_different_requests"] = round(float(np.mean(diff)), 3) if diff else None
            out[g]["n_pairs_with_different_requests"] = len(diff)
        req_out["scenario_level"][view] = out
    jdump(req_out, os.path.join(RES, "s3_request_only.json"))
    print(json.dumps(req_out["pairs_with_different_request_by_split"]))
    for view, o in req_out["scenario_level"].items():
        print(view, "cv", o["train_grouped_cv"], {g: (o[g]["acc"], o[g]["within_twin_acc"], o[g]["within_twin_acc_on_pairs_with_different_requests"],
                                                      o[g]["n_pairs_with_different_requests"]) for g in ("val", "test", "hard_test", "hard_val", "hard_train")})
        print("  A:", o["top_A_features"][:12])
        print("  B:", o["top_B_features"][:12])


if __name__ == "__main__":
    main()
