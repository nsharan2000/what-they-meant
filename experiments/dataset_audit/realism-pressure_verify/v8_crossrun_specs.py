"""VERIFY the 'only the hint differs' premise of RP-6's big-model P0 -> P2 contrast (cross-run: replication grid on the frozen
v1 specs vs the hard-set panel on the v1.2 specs).

For every cross-run pair (same model + scenario id, L2/P0 grid row vs L2/P2 panel row, rows listed in the v1.2/v1.2.1 fix reports
excluded as the analyst did) compare the scenario's tool_rules in the frozen spec (dataset/frozen_replication_v1 or
frozen_variants_v1) with the current spec (dataset/scenarios or scenarios_synth). Also check, on the logged episodes, that the
grid's tool results come from the frozen rules and the panel's from the current rules. Recompute the P0 -> P2 contrasts on the
pairs whose tool_rules did NOT change.
Outputs: results/v8_crossrun_specs.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from vcommon import GRID, OUT, PANEL, ROOT, boot, dump, ind, load_jsonl, mname


def load_specs(pattern):
    out = {}
    for f in glob.glob(os.path.join(ROOT, pattern)):
        for s in load_jsonl(f):
            out[s["id"]] = s
    return out


def rule_results(spec):
    res = set()
    for r in spec.get("tool_rules", []):
        for k in ("result_raw", "result"):
            if k in r:
                res.add(json.dumps(r[k], sort_keys=True))
    return res


def main():
    frozen = {**load_specs("dataset/frozen_replication_v1/*.jsonl"), **load_specs("dataset/frozen_variants_v1/*.jsonl")}
    current = {**load_specs("dataset/scenarios/*.jsonl"), **load_specs("dataset/scenarios_synth/*.jsonl")}
    changed_reported = {c["row"] for f in ("fix_v12_report.json", "fix_v121_report.json")
                        for c in json.load(open(os.path.join(ROOT, "dataset", f)))["changes"]}
    grid = {}
    for f in ("grid_open_P0.jsonl", "grid_claude.jsonl", "grid_open_variants.jsonl"):
        for i, r in enumerate(load_jsonl(os.path.join(GRID, f))):
            if r["level"] == "L2" and r["prompt"] == "P0":
                grid[(r["model"], r["id"])] = r
    recs = []
    prov = {"grid_results_from_frozen": 0, "grid_results_from_current_only": 0, "panel_results_from_current": 0,
            "panel_results_from_frozen_only": 0, "tool_results_checked_grid": 0, "tool_results_checked_panel": 0}
    for f in ("panel_mined_claude.jsonl", "panel_mined_open.jsonl"):
        for i, p in enumerate(load_jsonl(os.path.join(PANEL, f))):
            tw = {"A_should_ask": "A", "B_should_act": "B"}.get(p["twin"])
            g = grid.get((p["model"], p["id"]))
            if tw is None or g is None or p["id"] in changed_reported or "error" in (p["decision_grade"]["decision"],
                                                                                   g["decision_grade"]["decision"]):
                continue
            fz, cu = frozen.get(p["id"]), current.get(p["id"])
            if fz is None or cu is None:
                continue
            rules_changed = json.dumps(fz.get("tool_rules"), sort_keys=True) != json.dumps(cu.get("tool_rules"), sort_keys=True)
            fr, cr = rule_results(fz), rule_results(cu)
            for ep, key in ((g, "grid"), (p, "panel")):
                for e in ep["events"]:
                    if e.get("kind") != "tool_call" or e.get("tool_kind") not in ("lookup", "verify"):
                        continue
                    j = json.dumps(e.get("result"), sort_keys=True)
                    if j not in fr and j not in cr:
                        continue                                      # handler result (view / fallback), not a rule
                    prov[f"tool_results_checked_{key}"] += 1
                    if key == "grid":
                        prov["grid_results_from_frozen"] += int(j in fr)
                        prov["grid_results_from_current_only"] += int(j in cr and j not in fr)
                    else:
                        prov["panel_results_from_current"] += int(j in cr)
                        prov["panel_results_from_frozen_only"] += int(j in fr and j not in cr)
            a, b = ind(g["decision_grade"]["decision"], tw), ind(p["decision_grade"]["decision"], tw)
            rec = {"model": mname(p["model"]), "id": p["id"], "twin": tw, "rules_changed": rules_changed,
                   "panel_ref": f"{f}:{i}"}
            for k in a:
                rec[k + "_P0"], rec[k + "_P2"] = a[k], b[k]
            recs.append(rec)
    df = pd.DataFrame(recs)
    out = {"n_pairs": int(len(df)), "pairs_with_changed_tool_rules": int(df.rules_changed.sum()),
           "share_changed": round(float(df.rules_changed.mean()), 4), "provenance_check": prov,
           "changed_by_model": {m: [int(g.rules_changed.sum()), int(len(g))] for m, g in df.groupby("model")}}
    res = []
    for subset, sub in (("all", df), ("unchanged_rules_only", df[~df.rules_changed])):
        for (model, tw), g in sub.groupby(["model", "twin"]):
            for k in (["ask", "informed"] if tw == "A" else ["needless", "approve"]):
                d = (g[k + "_P2"] - g[k + "_P0"]).to_numpy(float)
                if len(d) < 5:
                    continue
                mu, lo, hi = boot(d, None, seed=81)
                res.append({"subset": subset, "model": model, "twin": tw, "metric": k, "n": int(len(g)),
                            "diff": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)]})
    out["contrasts"] = res
    s53 = df[df.id == "S53B"][["model", "rules_changed", "needless_P0", "needless_P2", "panel_ref"]].to_dict(orient="records")
    out["S53B"] = s53
    dump(out, "v8_crossrun_specs.json")
    print(json.dumps({k: out[k] for k in ("n_pairs", "pairs_with_changed_tool_rules", "share_changed", "provenance_check",
                                          "changed_by_model", "S53B")}, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
