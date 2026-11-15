"""Why the SFT loss can fall to ~0.1 while the decision stays a coin flip: under teacher forcing, the target's FIRST token selects the
twin. Group the SFT-train decision rows by the member's request (and by request + look-up call); count distinct targets per
group, and distinct targets once the first target token is also known (first word of a message, '<' + tool name of a call).
If (request, first token) almost always determines the whole target, every token after the first can be predicted from memory
of the 203 distinct conversations without reading the look-up result that actually decides between the twins.

    python3 experiments/dataset_audit/composition-memorization/selector.py
Output: results/selector.json
"""
from __future__ import annotations

import collections
import json

from common import dump, load_view, target_of
from wtm.spec import norm


def first_unit(t: str) -> str:
    if t.startswith("<tool_call>"):
        try:
            return "<call:" + json.loads(t.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0])["name"]
        except Exception:  # noqa: BLE001
            return "<call:?"
    return t.split()[0] if t.split() else ""


def main():
    rows = [r for r in load_view("sft_train") if r["turn"] == "decision"]
    out = {"decision_rows": len(rows)}
    for name, keyf in (("request", lambda r: norm(r["prompt"][1]["content"])),
                       ("request+lookup_call", lambda r: norm(r["prompt"][1]["content"]) + "||" + "||".join(
                           norm(m["content"]) for m in r["prompt"][2::2]))):
        g = collections.defaultdict(set)
        g1 = collections.defaultdict(set)
        g1_kind = collections.defaultdict(set)
        for r in rows:
            t = target_of(r)
            g[keyf(r)].add(t)
            g1[(keyf(r), first_unit(t))].add(t)
            g1_kind[(keyf(r), "tool" if t.startswith("<tool_call>") else "msg")].add(t)
        amb = {k: v for k, v in g.items() if len(v) > 1}
        rows_amb = sum(1 for r in rows if keyf(r) in amb)
        out[name] = {"groups": len(g), "groups_with_2plus_targets": len(amb), "rows_in_ambiguous_groups": rows_amb,
                     "rows_in_ambiguous_groups_share": round(rows_amb / len(rows), 3),
                     "after_first_unit_groups_with_2plus_targets": sum(1 for v in g1.values() if len(v) > 1),
                     "after_first_unit_rows_still_ambiguous": sum(1 for r in rows if len(g1[(keyf(r), first_unit(target_of(r)))]) > 1),
                     "after_kind_rows_still_ambiguous": sum(1 for r in rows if len(g1_kind[(keyf(r), 'tool' if target_of(r).startswith('<tool_call>') else 'msg')]) > 1),
                     "examples_ambiguous": [sorted(x[:90] for x in v) for v in list(amb.values())[:3]]}
    dump(out, "selector.json")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "examples_ambiguous"})
                      for k, v in out.items()}, indent=0))


if __name__ == "__main__":
    main()
