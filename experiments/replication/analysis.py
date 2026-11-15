"""Analysis for the replication study (preregistered in README.md). Never retype numbers: everything comes from results files.

python3 experiments/replication/analysis.py --results experiments/replication/results/grid_*.jsonl \
    --judge experiments/replication/results/judge_*.jsonl --scenarios dataset/scenarios/*.jsonl \
    --out experiments/replication/results/summary.json
Outputs per (model, prompt, level): n_pairs, PP (strict / lenient / judged), A-side success, harm, uninformed consent, needless ask,
strict-format rate, decision mix; per (model, prompt): RG and its decomposition with 95 % cluster-bootstrap CIs and exact McNemar p.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from wtm.spec import load_jsonl  # noqa: E402

GAPS = {"RG": ("L0r", "L2"), "context": ("L0r", "L1"), "burial": ("L1", "L2"), "length_only": ("L0r", "L0p"),
        "salience": ("L2s", "L2"), "hint": ("L0", "L0r"), "noise": ("L2", "L3")}


def outcome(row: dict, judged: dict | None, kind: str) -> bool:
    g = row["decision_grade"]
    if kind == "strict":
        return bool(g["success"])
    if kind == "lenient":
        return bool(g["success_lenient"])
    j = (judged or {}).get(row["key"])
    if j and isinstance(j, dict) and "behaviour_ok" in j:
        return bool(j["behaviour_ok"])
    return bool(g["success_lenient"])


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ns = ap.parse_args()
    files = [f for p in ns.results for f in glob.glob(p)]
    rows = [r for f in files for r in load_jsonl(f)]
    scn = {s["id"]: s for p in ns.scenarios for f in glob.glob(p) for s in load_jsonl(f)}
    judged = {}
    for p in ns.judge:
        for f in glob.glob(p):
            for j in load_jsonl(f):
                judged[j["key"]] = j.get("judge")
    # label adjudications (dataset/label_adjudications.json) widen what counts as correct for a few rows: re-grade those rows
    # from their stored decision class so frozen results and current specs are scored the same way
    from wtm.episode import decision_grade  # noqa: E402
    from wtm.labels import adjudicate, adjudications  # noqa: E402
    adj = adjudications()
    regraded = 0
    for r in rows:
        if r["id"] in adj and r["id"] in scn and r.get("decision_grade"):
            r["decision_grade"] = decision_grade(adjudicate(scn[r["id"]]), r["decision_grade"]["decision"])
            regraded += 1
    # one row per (model, prompt, level, id): keep the latest rep 0
    cell = {}
    for r in rows:
        if r.get("rep", 0) != 0 or r["id"] not in scn:
            continue
        cell[(r["model"], r["prompt"], r["level"], r["id"])] = r
    groups = defaultdict(list)
    for (m, p, lv, sid), r in cell.items():
        groups[(m, p, lv)].append(r)
    summary = {"by_cell": [], "gaps": [], "files": files, "n_judged": len(judged), "n_regraded_adjudicated": regraded}
    pair_out = {}   # (m,p,lv,kind) -> {pair_id: bool}
    for (m, p, lv), rs in sorted(groups.items()):
        by_pair = defaultdict(dict)
        for r in rs:
            by_pair[r["pair_id"]][r["twin"][0]] = r
        full = {pid: d for pid, d in by_pair.items() if "A" in d and "B" in d}
        res = {"model": m, "prompt": p, "level": lv, "n_rows": len(rs), "n_pairs": len(full)}
        for kind in ("strict", "lenient", "judged"):
            po = {pid: outcome(d["A"], judged, kind) and outcome(d["B"], judged, kind) for pid, d in full.items()}
            pair_out[(m, p, lv, kind)] = po
            res[f"PP_{kind}"] = round(sum(po.values()) / len(po), 4) if po else None
        A = [r for r in rs if r["twin"].startswith("A")]
        B = [r for r in rs if r["twin"].startswith("B")]
        rate = lambda xs, f: round(sum(1 for x in xs if f(x)) / len(xs), 4) if xs else None  # noqa: E731
        res.update({
            "A_success_strict": rate(A, lambda r: r["decision_grade"]["success"]),
            "A_success_lenient": rate(A, lambda r: r["decision_grade"]["success_lenient"]),
            "harm": rate(A, lambda r: r["decision_grade"]["harm"]),
            "uninformed_consent": rate(A, lambda r: r["decision_grade"].get("uninformed_consent")),
            "B_success": rate(B, lambda r: r["decision_grade"]["success"]),
            "needless_ask": rate(B, lambda r: r["decision_grade"]["needless_ask"]),
            "strict_format": round(sum(r.get("strict_format_rate", 0) for r in rs) / len(rs), 4),
            "decisions_A": dict(Counter(r["decision_grade"]["decision"] for r in A)),
            "decisions_B": dict(Counter(r["decision_grade"]["decision"] for r in B)),
            "mean_system_chars": round(sum(r.get("system_chars", 0) for r in rs) / len(rs)),
            "cost_usd": round(sum(r.get("cost_usd", 0) for r in rs), 4),
        })
        summary["by_cell"].append(res)
    # gaps with cluster bootstrap over source posts
    rng = random.Random(0)
    models_prompts = sorted({(m, p) for (m, p, lv) in groups})
    for (m, p) in models_prompts:
        for gname, (la, lb) in GAPS.items():
            for kind in ("strict", "judged"):
                a, b = pair_out.get((m, p, la, kind)), pair_out.get((m, p, lb, kind))
                if not a or not b:
                    continue
                common = sorted(set(a) & set(b))
                if not common:
                    continue
                clusters = defaultdict(list)
                for pid in common:
                    sid = next((s for s in scn if scn[s]["pair_id"] == pid), None)
                    clusters[scn[sid]["source"].get("post_id", pid) if sid else pid].append(pid)
                keys = list(clusters)
                diff = lambda ps: (sum(a[q] for q in ps) - sum(b[q] for q in ps)) / len(ps)  # noqa: E731
                point = diff(common)
                boots = []
                for _ in range(ns.boot):
                    ps = [q for k in (rng.choice(keys) for _ in keys) for q in clusters[k]]
                    boots.append(diff(ps))
                boots.sort()
                lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
                n10 = sum(1 for q in common if a[q] and not b[q])
                n01 = sum(1 for q in common if b[q] and not a[q])
                summary["gaps"].append({"model": m, "prompt": p, "gap": gname, "from": la, "to": lb, "kind": kind,
                                        "n_pairs": len(common), "value_pp": round(100 * point, 1),
                                        "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                                        "flip_pass_to_fail": n10, "flip_fail_to_pass": n01,
                                        "mcnemar_p": round(binom_two_sided(min(n10, n01), n10 + n01), 4)})
    json.dump(summary, open(ns.out, "w"), indent=1)
    print(f"cells={len(summary['by_cell'])} gaps={len(summary['gaps'])} judged={len(judged)} -> {ns.out}")


if __name__ == "__main__":
    main()
