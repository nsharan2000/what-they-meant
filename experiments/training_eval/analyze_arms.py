"""E1 (experiments/dataset_audit/REPORT.md): the stratified re-analysis of the held-out eval for every arm (2026-11-15).

Complements experiments/replication/analysis.py (paired pass + realism-gap decomposition, which it already does) with what the
audit asked for: decision rates per twin (ask / informed ask / card / commit / look-up first), D_ask = P(ask|A) - P(ask|B), and
within-scenario paired contrasts — the hint (P2 - P0) at every level and the levels (L1 - L0r, L2 - L1, L2 - L0r; L2s - L2 only
as an answer-hint diagnostic) — on D_ask and on paired pass, with 95 % bootstrap CIs over source-post clusters (test = 41 pairs
from 13 clusters; hard_test is always reported separately, never pooled). Strata: all pairs, and without the answer-bearing
pairs (V232, V082, H231, H201: a training target contains a held-out answer). 'act' items (twin B whose expected decision is to
act) are reported strict and lenient at P2 (the hint says "just show the approval card", which the act label calls a fail).
Optional Opus-judge verdicts (wtm/judge.py output: key -> behaviour_ok) give the judged paired pass; without them "judged" =
lenient, flagged in the output.

PRE-REGISTERED decision rule (audit E1): the prompt is NOT limiting a trained arm if both its P2-P0 effect and its L2-L1 effect on
D_ask have 95 % CIs that include 0.

  python3 experiments/training_eval/analyze_arms.py --results experiments/training_eval/results/eval_test.jsonl \
      [more result files, e.g. the Claude arms] --scenarios dataset/eval/test_*.jsonl --split test \
      [--judge <verdicts.jsonl> ...] --out experiments/training_eval/results/analysis/e1_test.json [--md <file.md>]
Arm labels: base (tiny-aya-global-local), the adapter label before the content hash (m1-9c892f3863 -> m1), claude:<x> -> claude-<x>.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
ANSWER_BEARING = {"V232", "V082", "H231", "H201"}
LEVELS = ("L0r", "L1", "L2", "L2s")
CONTRASTS = [("hint", "P0", "P2"), ("context", "L0r", "L1"), ("burial", "L1", "L2"), ("realism", "L0r", "L2"),
             ("salience_diag", "L2", "L2s")]


def arm_of(model: str) -> str:
    if model.startswith("claude:"):
        return "claude-" + model.split(":", 1)[1]
    name = model.split(":", 1)[1].split("@")[0] if model.startswith("vllm:") else model
    return "base" if name.startswith("tiny-aya-global") else name.rsplit("-", 1)[0]


def kind(dec: str) -> str:
    if dec.startswith(("ask", "tell")):
        return "ask"
    if dec.startswith("approve"):
        return "card"
    if dec.startswith("commit"):
        return "commit"
    return "other"


def lookup_first(row: dict) -> bool:
    ev = [e for e in row.get("events", []) if e.get("kind") in ("tool_call", "message", "invalid")]
    return bool(ev) and ev[0].get("kind") == "tool_call" and ev[0].get("tool_kind") in ("lookup", "verify")


def boot_ci(clusters: dict, stat, n: int, seed: int = 0):
    """95 % percentile CI of stat(list of pair ids) under resampling whole clusters."""
    rng = random.Random(seed)
    keys = list(clusters)
    vals = []
    for _ in range(n):
        ps = [p for k in (rng.choice(keys) for _ in keys) for p in clusters[k]]
        v = stat(ps)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--split", required=True, help="label for the output (test | hard_test); never pool the two")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--exclude-pairs", nargs="*", default=None,
                    help="pair ids for the 'excl' stratum (default: the answer-bearing pairs V232 V082 H231 H201)")
    ns = ap.parse_args()
    excluded_pairs = set(ns.exclude_pairs) if ns.exclude_pairs is not None else ANSWER_BEARING

    scn = {}
    for pat in ns.scenarios:
        for f in glob.glob(pat):
            for l in open(f):
                if l.strip():
                    s = json.loads(l)
                    scn[s["id"]] = s
    cluster_of = {s["pair_id"]: (s.get("source") or {}).get("post_id", s["pair_id"]) for s in scn.values()}
    judged = {}
    for f in ns.judge:
        for l in open(f):
            if l.strip():
                j = json.loads(l)
                v = j.get("judge") if isinstance(j.get("judge"), dict) else j   # wtm.judge rows: {"key", "judge": {...}}
                if isinstance(v, dict) and "behaviour_ok" in v:
                    judged[j["key"]] = bool(v["behaviour_ok"])
    rows = []
    for f in ns.results:
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    # cell[(arm, level, prompt)][pair_id][twin] = row
    cell = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["id"] not in scn:
            continue
        cell[(arm_of(r["model"]), r["level"], r["prompt"])][r["pair_id"]][r["twin"][0]] = r
    arms = sorted({k[0] for k in cell})

    def ok(r, how):
        g = r["decision_grade"]
        if how == "strict":
            return bool(g["success"])
        if how == "lenient":
            return bool(g["success_lenient"])
        return judged.get(r["key"], bool(g["success_lenient"]))

    def pair_vals(arm, lv, p, pid):
        d = cell[(arm, lv, p)].get(pid, {})
        if "A" not in d or "B" not in d:
            return None
        a, b = d["A"], d["B"]
        ka, kb = kind(a["decision_grade"]["decision"]), kind(b["decision_grade"]["decision"])
        return {"dask": (ka == "ask") - (kb == "ask"),
                **{f"pp_{h}": float(ok(a, h) and ok(b, h)) for h in ("strict", "lenient", "judged")}}

    out = {"split": ns.split, "files": ns.results, "n_judged_keys": len(judged), "judged_is_lenient_fallback": not judged,
           "excl_stratum_pairs": sorted(excluded_pairs), "cells": [], "contrasts": [], "act_items_P2": [],
           "decision_rule": "prompt not limiting an arm iff its hint (P2-P0) and burial (L2-L1) effects on D_ask both have 95 % CIs "
                            "that include 0", "rule_verdicts": {}}
    for stratum in ("all", "excl"):
        keep = (lambda pid: True) if stratum == "all" else (lambda pid: pid not in excluded_pairs)
        for arm in arms:
            for p in ("P0", "P1", "P2"):
                for lv in LEVELS:
                    c = cell.get((arm, lv, p))
                    if not c:
                        continue
                    pids = [pid for pid in c if keep(pid) and "A" in c[pid] and "B" in c[pid]]
                    A = [c[pid]["A"] for pid in pids]
                    B = [c[pid]["B"] for pid in pids]
                    if not pids:
                        continue
                    rate = lambda rs, f: round(sum(map(f, rs)) / len(rs), 4)   # noqa: E731
                    dk = lambda k: (lambda r: kind(r["decision_grade"]["decision"]) == k)   # noqa: E731
                    inf = lambda r: r["decision_grade"]["decision"] in ("ask_informed", "tell_informed")   # noqa: E731
                    vals = [pair_vals(arm, lv, p, pid) for pid in pids]
                    out["cells"].append({
                        "stratum": stratum, "arm": arm, "level": lv, "prompt": p, "n_pairs": len(pids),
                        "A": {"ask": rate(A, dk("ask")), "informed_ask": rate(A, inf), "card": rate(A, dk("card")),
                              "commit": rate(A, dk("commit")), "other": rate(A, dk("other")), "lookup_first": rate(A, lookup_first),
                              "success_strict": rate(A, lambda r: ok(r, "strict")), "success_lenient": rate(A, lambda r: ok(r, "lenient"))},
                        "B": {"ask": rate(B, dk("ask")), "card": rate(B, dk("card")), "commit": rate(B, dk("commit")),
                              "other": rate(B, dk("other")), "lookup_first": rate(B, lookup_first),
                              "needless_ask": rate(B, lambda r: bool(r["decision_grade"].get("needless_ask"))),
                              "success_strict": rate(B, lambda r: ok(r, "strict"))},
                        "D_ask": round(sum(v["dask"] for v in vals) / len(vals), 4),
                        "PP_strict": round(sum(v["pp_strict"] for v in vals) / len(vals), 4),
                        "PP_lenient": round(sum(v["pp_lenient"] for v in vals) / len(vals), 4),
                        "PP_judged": round(sum(v["pp_judged"] for v in vals) / len(vals), 4)})
            # paired contrasts
            for name, x, y in CONTRASTS:
                combos = ([(lv, x, lv, y) for lv in LEVELS] if name == "hint" else
                          [(x, p, y, p) for p in ("P0", "P1", "P2")])
                for l1, p1, l2, p2 in combos:
                    common = [pid for pid in cell.get((arm, l1, p1), {}) if keep(pid)
                              and pair_vals(arm, l1, p1, pid) and pair_vals(arm, l2, p2, pid)]
                    if len(common) < 3:
                        continue
                    clusters = defaultdict(list)
                    for pid in common:
                        clusters[cluster_of.get(pid, pid)].append(pid)
                    for metric in ("dask", "pp_lenient", "pp_judged", "pp_strict"):
                        def stat(ps, metric=metric):
                            if not ps:
                                return None
                            return sum(pair_vals(arm, l2, p2, q)[metric] - pair_vals(arm, l1, p1, q)[metric] for q in ps) / len(ps)
                        out["contrasts"].append({"stratum": stratum, "arm": arm, "contrast": name,
                                                 "from": f"{l1}/{p1}", "to": f"{l2}/{p2}", "metric": metric,
                                                 "n_pairs": len(common), "n_clusters": len(clusters),
                                                 "effect": round(stat(common), 4), "ci95": boot_ci(clusters, stat, ns.boot)})
            # act items at P2 (twin B expected to act): strict vs lenient
            if stratum == "all":
                for lv in LEVELS:
                    c = cell.get((arm, lv, "P2"), {})
                    acts = [d["B"] for d in c.values() if "B" in d and scn[d["B"]["id"]]["expected"]["decision"] == "act"]
                    if acts:
                        out["act_items_P2"].append({"arm": arm, "level": lv, "n": len(acts),
                                                    "strict": round(sum(ok(r, "strict") for r in acts) / len(acts), 4),
                                                    "lenient": round(sum(ok(r, "lenient") for r in acts) / len(acts), 4),
                                                    "decisions": dict(Counter(r["decision_grade"]["decision"] for r in acts))})
    for arm in arms:   # the pre-registered rule, 'all' stratum, primary prompt P0 for the burial effect, every level for the hint
        cs = [c for c in out["contrasts"] if c["arm"] == arm and c["stratum"] == "all" and c["metric"] == "dask"
              and (c["contrast"] == "hint" or (c["contrast"] == "burial" and c["from"].endswith("P0")))]
        if cs:
            limiting = [f"{c['contrast']} {c['from']}->{c['to']} {c['effect']:+.3f} {c['ci95']}" for c in cs
                        if c["ci95"] and not (c["ci95"][0] <= 0 <= c["ci95"][1])]
            out["rule_verdicts"][arm] = {"prompt_limiting": bool(limiting), "effects_excluding_0": limiting}
    with open(ns.out, "w") as fh:
        json.dump(out, fh, indent=1)
    if ns.md:
        lines = [f"# E1 — {ns.split} ({'judged' if judged else 'judged = lenient fallback, no verdicts'})", "",
                 "| arm | prompt | level | pairs | D_ask | PP strict | PP lenient | PP judged | A ask | A informed | A card | A commit | B card | B needless | look-up first (A) |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in out["cells"]:
            if c["stratum"] != "all":
                continue
            lines.append(f"| {c['arm']} | {c['prompt']} | {c['level']} | {c['n_pairs']} | {c['D_ask']:+.2f} | {c['PP_strict']:.2f} | "
                         f"{c['PP_lenient']:.2f} | {c['PP_judged']:.2f} | {c['A']['ask']:.2f} | {c['A']['informed_ask']:.2f} | "
                         f"{c['A']['card']:.2f} | {c['A']['commit']:.2f} | {c['B']['card']:.2f} | {c['B']['needless_ask']:.2f} | "
                         f"{c['A']['lookup_first']:.2f} |")
        lines += ["", "## Paired contrasts on D_ask (all pairs; 95 % cluster-bootstrap CI)", "",
                  "| arm | contrast | from → to | pairs | clusters | effect | CI |", "|---|---|---|---|---|---|---|"]
        for c in out["contrasts"]:
            if c["stratum"] == "all" and c["metric"] == "dask":
                lines.append(f"| {c['arm']} | {c['contrast']} | {c['from']} → {c['to']} | {c['n_pairs']} | {c['n_clusters']} | "
                             f"{c['effect']:+.3f} | {c['ci95']} |")
        lines += ["", "## Pre-registered rule: is the prompt limiting?", ""]
        for arm, v in out["rule_verdicts"].items():
            lines.append(f"- {arm}: {'YES — ' + '; '.join(v['effects_excluding_0']) if v['prompt_limiting'] else 'no (all CIs include 0)'}")
        open(ns.md, "w").write("\n".join(lines) + "\n")
    print(f"E1 {ns.split}: arms {arms}; cells {len(out['cells'])}; contrasts {len(out['contrasts'])}; judged keys {len(judged)} -> {ns.out}")
    print("E1_DONE")


if __name__ == "__main__":
    main()
