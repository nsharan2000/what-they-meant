"""V4 (verifier): base Tiny Aya (experiments/training_eval/results/eval_base_v12.jsonl: greedy, test + hard sets, L0r/L1/L2/L2s x
P0/P2) -- twin discrimination by request kind (pair level, from the eval specs' clean user_message), pair-bootstrap CIs, and the
same quantity per split and with the WORDING pairs broken down by source family (to see whether one family drives it).

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v4_base.py -> results/v4_base.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_twins import req_kind  # noqa: E402
from vcommon import ROOT, T, eval_specs, jl, save  # noqa: E402

MSG = {"ask_informed", "ask_generic", "tell_informed", "statement"}


def main():
    out = {}
    specs = {}
    origin = {}
    for name in ("test", "val", "hard_test", "hard_val", "hard_train"):
        for s in eval_specs(name):
            specs[s["id"]] = s
            origin[s["pair_id"]] = name
    byp = defaultdict(dict)
    for s in specs.values():
        if T(s) in "AB":
            byp[s["pair_id"]][T(s)] = s
    rk = {p: req_kind(d["A"]["user_message"], d["B"]["user_message"]) for p, d in byp.items() if "A" in d and "B" in d}
    fam = {p: (d["A"].get("source", {}).get("post_id") or p) for p, d in byp.items() if "A" in d}   # source post = family
    rows = [r for r in jl(os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl")) if T(r) in "AB"]
    out["n_rows"] = len(rows)
    out["rows_by_split"] = dict(Counter(origin.get(r["pair_id"], "?") for r in rows))
    for r in rows:
        r["_ask"] = r["decision_grade"]["decision"] in MSG
        r["_rk"] = rk.get(r["pair_id"], "?")

    def disc(sel):
        a = [r["_ask"] for r in sel if T(r) == "A"]
        b = [r["_ask"] for r in sel if T(r) == "B"]
        return (np.mean(a) - np.mean(b)) if a and b else np.nan

    def boot(sel_rows, kind_a="wording", kind_b="identical", n=4000, seed=3):
        bp = defaultdict(list)
        for r in sel_rows:
            bp[r["pair_id"]].append(r)
        pids = sorted(bp)
        rng = np.random.default_rng(seed)
        da, db, dd = [], [], []
        for _ in range(n):
            pick = rng.choice(len(pids), len(pids))
            s = [x for k in pick for x in bp[pids[k]]]
            x = disc([r for r in s if r["_rk"] == kind_a])
            y = disc([r for r in s if r["_rk"] == kind_b])
            da.append(x)
            db.append(y)
            dd.append(x - y)
        q = lambda v: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]  # noqa: E731
        return {"disc_" + kind_a: round(float(disc([r for r in sel_rows if r["_rk"] == kind_a])), 3), "ci_a": q(da),
                "disc_" + kind_b: round(float(disc([r for r in sel_rows if r["_rk"] == kind_b])), 3), "ci_b": q(db),
                "difference_ci": q(dd), "n_pairs": len(pids)}
    out["all"] = boot(rows)
    out["test"] = boot([r for r in rows if origin.get(r["pair_id"]) == "test"])
    out["hard"] = boot([r for r in rows if origin.get(r["pair_id"], "").startswith("hard")])
    out["by_kind"] = {k: {"pairs": len({r["pair_id"] for r in rows if r["_rk"] == k}),
                          "askA": round(float(np.mean([r["_ask"] for r in rows if r["_rk"] == k and T(r) == "A"])), 3),
                          "askB": round(float(np.mean([r["_ask"] for r in rows if r["_rk"] == k and T(r) == "B"])), 3),
                          "correctA": round(float(np.mean([r["decision_grade"]["success"] for r in rows if r["_rk"] == k and T(r) == "A"])), 3)}
                      for k in ("identical", "identifier_only", "wording")}
    # per wording pair: where does the discrimination come from?
    per = []
    for p in sorted({r["pair_id"] for r in rows if r["_rk"] == "wording"}):
        sel = [r for r in rows if r["pair_id"] == p]
        per.append({"pair": p, "split": origin.get(p), "family": fam.get(p), "disc": round(float(disc(sel)), 3),
                    "A_msg": byp[p]["A"]["user_message"][:110], "B_msg": byp[p]["B"]["user_message"][:110],
                    "A_decisions": dict(Counter(r["decision_grade"]["decision"] for r in sel if T(r) == "A"))})
    out["wording_pairs"] = sorted(per, key=lambda x: -x["disc"])
    # family (source-post) robustness: per-family discrimination on wording pairs, and leave-one-family-out
    fams = sorted({fam[p] for p in {r["pair_id"] for r in rows if r["_rk"] == "wording"}})
    fd = {}
    for f in fams:
        sel = [r for r in rows if r["_rk"] == "wording" and fam.get(r["pair_id"]) == f]
        fd[f] = {"pairs": sorted({r["pair_id"] for r in sel}), "disc": round(float(disc(sel)), 3)}
    out["wording_disc_by_family"] = fd
    out["wording_disc_leave_one_family_out"] = {f: round(float(disc([r for r in rows if r["_rk"] == "wording" and fam.get(r["pair_id"]) != f])), 3)
                                                for f in fams}
    # family-cluster bootstrap of wording-minus-identical (resample source families, not pairs)
    bf = defaultdict(list)
    for r in rows:
        bf[fam.get(r["pair_id"], r["pair_id"])].append(r)
    fl = sorted(bf)
    rng = np.random.default_rng(9)
    dd = []
    for _ in range(4000):
        pick = rng.choice(len(fl), len(fl))
        s_ = [x for k in pick for x in bf[fl[k]]]
        dd.append(disc([r for r in s_ if r["_rk"] == "wording"]) - disc([r for r in s_ if r["_rk"] == "identical"]))
    out["family_bootstrap_wording_minus_identical"] = {"n_families": len(fl), "ci95": [round(float(np.nanpercentile(dd, 2.5)), 3),
                                                       round(float(np.nanpercentile(dd, 97.5)), 3)],
                                                       "share_le_0": round(float(np.nanmean(np.array(dd) <= 0)), 4)}
    out["wording_pairs_disc_gt0"] = sum(x["disc"] > 0 for x in per)
    out["wording_pairs_n"] = len(per)
    save(out, "v4_base.json")
    for k, v in out.items():
        if k != "wording_pairs":
            print(k, v)
    for x in out["wording_pairs"]:
        print(x["pair"], x["split"], x["disc"], "|", x["A_msg"], "||", x["B_msg"], x["A_decisions"])


if __name__ == "__main__":
    main()
