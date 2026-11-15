"""RP-1 'it also feeds the hard-set selection': programmatic sensitivity of the mined hard set (panel_mined, L2/P2, 6 models).

The real selection used Opus-judge verdicts (>= 4 of 6 models fail a pair). Here a programmatic proxy: a model fails a pair when
twin A is not lenient-correct (wtm grader 'success_lenient') or twin B is not strictly correct. Count pairs failed by >= 4 models
(a) as logged, (b) when a B-side needless question that follows an empty / unscripted look-up is NOT counted as a failure.
Also: overlap of the proxy with the published hard list (experiments/hardset/results/hardset_mined_6models.json).
Outputs: results/v11_hardset_sensitivity.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import pandas as pd

from vcommon import OUT, PANEL, ROOT, dump, load_jsonl, mname

EMPTY = {"No matching records found."}


def empty_before(events):
    for e in events:
        if e.get("kind") != "tool_call":
            continue
        if e.get("tool_kind") in ("ui", "commit"):
            break
        r = e.get("result")
        if isinstance(r, dict) and (r.get("result") in EMPTY or str(r.get("error", "")).startswith("Unknown tool")
                                    or (set(r) == {"total", "offset", "items"} and r.get("total") == 0)):
            return True
    return False


def main():
    rows = []
    for f in ("panel_mined_claude.jsonl", "panel_mined_open.jsonl", "panel_mined_fable.jsonl"):
        for i, r in enumerate(load_jsonl(os.path.join(PANEL, f))):
            tw = {"A_should_ask": "A", "B_should_act": "B"}.get(r["twin"])
            if tw is None:
                continue
            g = r["decision_grade"]
            dec = g["decision"]
            fail = (not g["success_lenient"]) if tw == "A" else (not g["success"])
            needless = tw == "B" and dec in ("ask_informed", "ask_generic", "tell_informed")
            rows.append({"model": mname(r["model"]), "pair": r["pair_id"], "twin": tw, "fail": fail,
                         "fail_excl_empty": fail and not (needless and empty_before(r.get("events", []))),
                         "needless": needless, "empty_before": empty_before(r.get("events", []))})
    df = pd.DataFrame(rows)
    per = df.groupby(["pair", "model"]).agg(fail=("fail", "max"), fail2=("fail_excl_empty", "max")).reset_index()
    cnt = per.groupby("pair").agg(n_models=("model", "nunique"), n_fail=("fail", "sum"), n_fail2=("fail2", "sum")).reset_index()
    full = cnt[cnt.n_models == 6]
    pub = json.load(open(os.path.join(ROOT, "experiments", "hardset", "results", "hardset_mined_6models.json")))
    hard_pub = set(pub["hard_pairs"])
    proxy = set(full[full.n_fail >= 4].pair)
    proxy2 = set(full[full.n_fail2 >= 4].pair)
    b = df[df.twin == "B"]
    out = {"pairs_with_all_6_models": int(len(full)), "published_hard": len(hard_pub),
           "proxy_hard_as_logged": len(proxy), "proxy_hard_excluding_needless_after_empty": len(proxy2),
           "proxy_overlap_with_published": len(proxy & hard_pub),
           "pairs_that_drop_out": sorted(proxy - proxy2),
           "B_needless_total": int(b.needless.sum()), "B_needless_after_empty": int((b.needless & b.empty_before).sum())}
    dump(out, "v11_hardset_sensitivity.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
