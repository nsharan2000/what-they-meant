"""Missed-issue check (simulator fidelity at EVERY level): catch-all tool_rules (match: []) return one fixed result whatever the
arguments, e.g. add_to_cart('tomatoes') -> 'Garlic, 2 pcs'. Detect episodes in which the same look-up tool is called twice
with DIFFERENT arguments and returns the IDENTICAL non-empty result before the first decision ('echo'), by level, and the
B-needless rate with / without such an echo. Also count catch-all rules in the as-trained specs.
Outputs: results/v14_echo.json
"""
from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd

from vcommon import CUR, dump, episodes, ind, ref, load_jsonl

EMPTY = '{"result": "No matching records found."}'


def main():
    rows = []
    for e in episodes(keep_events=True):
        if e["dec"] == "error" or e["fam"] is None:
            continue
        seen = {}
        echo = False
        for ev in e.pop("events"):
            if ev.get("kind") != "tool_call":
                continue
            if ev.get("tool_kind") in ("ui", "commit"):
                break
            res = json.dumps(ev.get("result"), sort_keys=True)
            if res == EMPTY or '"error"' in res:
                continue
            key = ev.get("tool")
            args = json.dumps(ev.get("args"), sort_keys=True)
            for a2, r2 in seen.get(key, []):
                if a2 != args and r2 == res:
                    echo = True
            seen.setdefault(key, []).append((args, res))
        d = ind(e["dec"], e["twin"])
        rows.append({"level": e["level"], "twin": e["twin"], "model": e["model"], "echo": int(echo),
                     "needless": d.get("needless", 0), "ref": ref(e["path"], e["line"])})
    df = pd.DataFrame(rows)
    out = {"by_level": []}
    for (lv, tw), g in df.groupby(["level", "twin"]):
        rec = {"level": lv, "twin": tw, "n": int(len(g)), "share_echo": round(g.echo.mean(), 4)}
        if tw == "B":
            rec["needless_if_echo"] = round(g[g.echo == 1].needless.mean(), 4) if g.echo.sum() else None
            rec["needless_if_not"] = round(g[g.echo == 0].needless.mean(), 4)
        out["by_level"].append(rec)
    out["examples_B_needless_echo"] = df[(df.echo == 1) & (df.twin == "B") & (df.needless == 1)].head(5).ref.tolist()
    n_rules = n_catch = 0
    specs_with = 0
    for r in load_jsonl(os.path.join(CUR, "grpo_train.jsonl")):
        s = json.loads(r["scenario"])
        c = sum(1 for x in s.get("tool_rules", []) if not x.get("match"))
        n_rules += len(s.get("tool_rules", []))
        n_catch += c
        specs_with += int(c > 0)
    out["grpo_rows_with_catch_all_rule"] = specs_with
    out["grpo_rule_counts"] = {"rules": n_rules, "catch_all": n_catch}
    dump(out, "v14_echo.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
