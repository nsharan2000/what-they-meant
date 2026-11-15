"""Misc checks: S49P (A_pushback) rows vs S49A; P2 hint share by twin on decision rows; distinct conversations by scenario.
Output: results/v11_misc.json"""
import json, os
from collections import Counter, defaultdict
from vcommon import ROOT, cur, dump, target, request
HINT = open(os.path.join(ROOT, "prompts/golden_short.md")).read().strip()
tr = cur("sft_train")
out = {}
p = [(i, r) for i, r in enumerate(tr) if r["scenario_id"] == "S49P"]
a = {(r["level"], r["turn"]): (i, r) for i, r in enumerate(tr) if r["scenario_id"] == "S49A"}
rec = []
for i, r in p:
    j = a.get((r["level"], r["turn"]))
    d = {"line": i, "level": r["level"], "ps": r["prompt_setting"], "turn": r["turn"], "S49A_line": j[0] if j else None}
    if j:
        ra = j[1]
        strip = lambda s: s.replace("\n\n" + HINT, "")
        d["S49A_ps"] = ra["prompt_setting"]
        d["system_equal_hint_stripped"] = strip(r["prompt"][0]["content"]) == strip(ra["prompt"][0]["content"])
        d["rest_equal"] = r["prompt"][1:] == ra["prompt"][1:]
        d["target_equal"] = target(r) == target(ra)
    rec.append(d)
out["S49P_rows"] = rec
dec = [r for r in tr if r["turn"] == "decision"]
c = Counter((r["twin"][0], r["prompt_setting"]) for r in dec)
out["decision_ps_by_twin"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(c.items())}
out["P2_share_A"] = round(c[("A", "P2")] / (c[("A", "P2")] + c[("A", "P0")]), 4)
out["P2_share_B"] = round(c[("B", "P2")] / (c[("B", "P2")] + c[("B", "P0")]), 4)
out["P_B_given_P2"] = round(c[("B", "P2")] / (c[("A", "P2")] + c[("B", "P2")]), 4)
out["P_B_given_P0"] = round(c[("B", "P0")] / (c[("A", "P0")] + c[("B", "P0")]), 4)
conv = defaultdict(set)
for r in dec:
    conv[json.dumps(r["prompt"][1:], ensure_ascii=False)].add(r["scenario_id"])
out["conversations_shared_by_2plus_scenarios"] = [sorted(v) for v in conv.values() if len(v) > 1]
out["distinct_scenario_request_pairs"] = len({(r["scenario_id"], request(r)) for r in dec})
dump(out, "v11_misc.json")
print(json.dumps(out, indent=1))
