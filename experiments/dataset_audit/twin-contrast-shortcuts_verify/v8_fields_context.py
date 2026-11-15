"""V8 (verifier): TCS-4/6/7 details.
  (a) near-pure field=value tokens in the last tool result (pairs where the token occurs in A's result only / B's only), train;
  (b) M1 ask rate vs an out-of-fold (cluster-grouped) result-structure score, within twin A and within twin B (Spearman +
      permutation p) -- re-derived with this verifier's own structure model, not the analyst's scores;
  (c) TCS-7: L0r tool-list differences, member-file differences, teacher-forced look-up differences, L3 noisy rewrites
      (train specs and test specs);
  (d) what makes twin A's last result longer in TEST (extra keys / list items present only in A), to judge whether the
      'A is longer' asymmetry is the decisive evidence itself.

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v8_fields_context.py -> results/v8_fields_context.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_twins import req_kind  # noqa: E402
from v7_classifiers import featurize, fit, fv_tokens, views  # noqa: E402
from vcommon import HERE, T, all_specs, curated, eval_specs, jl, leaves, parts, result_obj, save, splits  # noqa: E402


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spear(x, y):
    a, b = rank(x), rank(y)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def perm_p(x, y, n=5000, seed=1):
    r0 = spear(x, y)
    rng = np.random.default_rng(seed)
    c = sum(abs(spear(x, rng.permutation(y))) >= abs(r0) - 1e-12 for _ in range(n))
    return (c + 1) / (n + 1)


def main():
    out = {}
    pair_split, cluster = splits()
    sft = curated("sft_train")
    dec = [r for r in sft if r["turn"] == "decision" and T(r) in "AB"]
    m = defaultdict(dict)
    for r in dec:
        m[(r["pair_id"], r["level"])][T(r)] = r
    # (a) near-pure field=value tokens, counted per PAIR (a token counted for a pair if present in that twin's result at any level)
    tokA, tokB = defaultdict(set), defaultdict(set)
    no_result = set()
    for (pid, lv), d in m.items():
        if "A" not in d or "B" not in d:
            continue
        rsa, rsb = parts(d["A"]["prompt"])[3], parts(d["B"]["prompt"])[3]
        if not rsa or not rsb:
            no_result.add((pid, lv, len(rsa), len(rsb)))
            continue
        oa = result_obj(rsa[-1])[0]
        ob = result_obj(rsb[-1])[0]
        fa, fb = fv_tokens(oa), fv_tokens(ob)
        for t in fa - fb:
            tokA[t].add(pid)
        for t in fb - fa:
            tokB[t].add(pid)
    out["matched_rows_without_tool_result"] = sorted(no_result)
    names = ["payment_allowed=false", "payment_allowed=null", "merchant_name=null", "currency=null", "network=null", "network=ethereum"]
    out["field_value_tokens_pairs"] = {t: {"A_only_pairs": sorted(tokA.get(t, [])), "B_only_pairs": sorted(tokB.get(t, [])),
                                           "A_clusters": sorted({cluster[p] for p in tokA.get(t, [])}),
                                           "B_clusters": sorted({cluster[p] for p in tokB.get(t, [])})} for t in names}
    allt = set(tokA) | set(tokB)
    lop = sorted(((t, len(tokA.get(t, ())), len(tokB.get(t, ()))) for t in allt), key=lambda x: -abs(x[1] - x[2]))
    out["most_one_sided_tokens_pairs"] = [{"token": t, "A_pairs": a, "B_pairs": b} for t, a, b in lop[:20]]

    # (b) M1 ask rate vs OOF cluster-grouped result-structure score
    tr = dec
    for r in tr:
        r["_cluster"] = cluster.get(r["pair_id"])
        r["_v"] = views(r)
    cls = sorted({r["_cluster"] for r in tr})
    rng = np.random.default_rng(20261114)
    rng.shuffle(cls)
    fold = {c: i % 5 for i, c in enumerate(cls)}
    oof = np.zeros(len(tr))
    for k in range(5):
        trn = [r for r in tr if fold[r["_cluster"]] != k]
        idx = [i for i, r in enumerate(tr) if fold[r["_cluster"]] == k]
        Xa, Xb = featurize(trn, [tr[i] for i in idx], "result_structure")
        w, b = fit(Xa, np.array([1.0 if T(r) == "A" else 0.0 for r in trn]))
        oof[idx] = Xb @ w + b
    score = {(r["scenario_id"], r["level"]): oof[i] for i, r in enumerate(tr)}
    sc = jl(os.path.join(os.path.dirname(HERE), "..", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl"))
    MSG = {"ask_informed", "ask_generic", "tell_informed", "statement"}
    per = defaultdict(list)
    for s in sc:
        per[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"] in MSG)
    assoc = {}
    for t in "AB":
        ks = [k for k in per if k[0].endswith(t) and k in score]
        x = [score[k] for k in ks]
        y = [float(np.mean(per[k])) for k in ks]
        assoc[t] = {"n": len(ks), "spearman": round(spear(x, y), 3), "perm_p": round(perm_p(x, y), 4)}
    out["m1_ask_vs_oof_structure_score"] = assoc

    # (c) TCS-7 context differences, at the spec level (train pairs and test pairs)
    specs = all_specs()
    for s in eval_specs("test"):
        specs.setdefault(s["id"], s)
    byp = defaultdict(dict)
    for s in specs.values():
        if T(s) in "AB":
            byp[s["pair_id"]][T(s)] = s
    ctx = {}
    for split_name in ("train", "test"):
        pids = sorted(p for p, sp in pair_split.items() if sp == split_name and "A" in byp.get(p, {}) and "B" in byp.get(p, {}))
        l0 = [p for p in pids if set(byp[p]["A"]["l0_tools"]) != set(byp[p]["B"]["l0_tools"])]
        l0_detail = {p: {"A_only": sorted(set(byp[p]["A"]["l0_tools"]) - set(byp[p]["B"]["l0_tools"])),
                         "B_only": sorted(set(byp[p]["B"]["l0_tools"]) - set(byp[p]["A"]["l0_tools"]))} for p in l0}
        mem = [p for p in pids if byp[p]["A"]["member_id"] != byp[p]["B"]["member_id"]]
        pref = [p for p in pids if byp[p]["A"].get("prefix_calls") != byp[p]["B"].get("prefix_calls")]
        pref_detail = {p: [byp[p]["A"].get("prefix_calls"), byp[p]["B"].get("prefix_calls")] for p in pref}
        pref_kind = {p: req_kind(json.dumps(byp[p]["A"].get("prefix_calls")), json.dumps(byp[p]["B"].get("prefix_calls"))) for p in pref}
        noisy = [p for p in pids if byp[p]["A"].get("user_message_noisy") and byp[p]["B"].get("user_message_noisy")]
        noisy_diff = {}
        for p in noisy:
            a, b = byp[p]["A"], byp[p]["B"]
            clean = req_kind(a["user_message"], b["user_message"])
            nz = req_kind(a["user_message_noisy"], b["user_message_noisy"])
            if clean in ("identical", "identifier_only") and nz == "wording":
                noisy_diff[p] = [a["user_message_noisy"], b["user_message_noisy"]]
        ctx[split_name] = {"n_pairs": len(pids), "l0_tools_differ": l0_detail, "member_file_differs": mem,
                           "prefix_calls_differ": pref, "prefix_calls_differ_kind": pref_kind,
                           "prefix_detail": {p: pref_detail[p] for p in pref[:20]},
                           "noisy_wording_only_after_rewrite": noisy_diff}
    out["context_differences"] = ctx
    # share of matched train decision rows whose teacher-forced calls differ
    calls_diff = sum(parts(d["A"]["prompt"])[2] != parts(d["B"]["prompt"])[2] for d in m.values() if "A" in d and "B" in d)
    out["train_decision_twin_rows_calls_differ"] = {"n": calls_diff, "of": sum(1 for d in m.values() if "A" in d and "B" in d)}

    # (d) TEST: what is in A's last result that B's lacks (L1 rows)
    ev = [r for r in jl(os.path.join(HERE, "cache", "eval_rows.jsonl")) if r["split"] in ("test", "val") and r["level"] == "L1"]
    em = defaultdict(dict)
    for r in ev:
        em[(r["split"], r["pair_id"])][T(r)] = r
    rows = []
    for (sp, pid), d in sorted(em.items()):
        if "A" not in d or "B" not in d:
            continue
        if not parts(d["A"]["prompt"])[3] or not parts(d["B"]["prompt"])[3]:
            continue
        ra, rb = parts(d["A"]["prompt"])[3][-1], parts(d["B"]["prompt"])[3][-1]
        oa, ob = result_obj(ra)[0], result_obj(rb)[0]
        la = Counter(p for p, _ in leaves(oa)) if isinstance(oa, (dict, list)) else Counter()
        lb = Counter(p for p, _ in leaves(ob)) if isinstance(ob, (dict, list)) else Counter()
        extra_a = sorted((la - lb).elements())
        extra_b = sorted((lb - la).elements())
        rows.append({"split": sp, "pair": pid, "chars_A": len(ra), "chars_B": len(rb), "extra_paths_A": extra_a[:12],
                     "extra_paths_B": extra_b[:12], "n_extra_A": len(extra_a), "n_extra_B": len(extra_b)})
    out["eval_L1_result_extra_paths"] = rows
    out["test_L1_pairs_A_longer"] = sum(r["chars_A"] > r["chars_B"] for r in rows if r["split"] == "test")
    out["test_L1_pairs_A_has_extra_paths"] = sum(r["n_extra_A"] > r["n_extra_B"] for r in rows if r["split"] == "test")
    out["test_L1_pairs"] = sum(r["split"] == "test" for r in rows)
    save(out, "v8_fields_context.json")
    for k, v in out.items():
        if k not in ("eval_L1_result_extra_paths", "context_differences"):
            print(k, v)
    for k, v in out["context_differences"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "prefix_detail"})
    for r in rows:
        print(r["split"], r["pair"], r["chars_A"], r["chars_B"], "A+", r["extra_paths_A"][:6], "B+", r["extra_paths_B"][:6])


if __name__ == "__main__":
    main()
