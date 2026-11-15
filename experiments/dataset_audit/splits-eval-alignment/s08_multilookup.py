"""6 (continued). Multi-look-up items: in which canonical look-up does the A-vs-B evidence first appear?

Training rows almost always show ONE look-up before the decision; the held-out hard pairs often need 2-3. For every pair (A and B
twins) in the training scenarios and each evaluation file, run the canonical look-ups through wtm.sim.Simulator at L1 and record
the index of the first look-up whose result differs between the twins (0 = request differs / no look-up needed). If the evidence
first appears in look-up 2 or 3, a policy that decides after one look-up (what the training rows demonstrate) never sees it.

python3 -B experiments/dataset_audit/splits-eval-alignment/s08_multilookup.py -> results/s08_multilookup.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from common import dump, eval_files, scen_files, views
from wtm.render import render
from wtm.sim import Simulator
from wtm.spec import load_env, load_member

V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
groups = {"train(trained)": [by_id[x] for x in sorted({r["scenario_id"] for r in V["sft_train"]})]}
for name, rows in eval_files().items():
    groups.setdefault(name.rsplit("_", 1)[0], []).extend(rows)
envs, members = {}, {}


def results(s):
    env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
    mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
    _, user, tools = render(env, mem, s, "L1", "P0")
    sim = Simulator(env, mem, s, tools, level="L1")
    return user, [(c["tool"], json.dumps(c["arguments"], sort_keys=True),
                   json.dumps(sim.execute(c["tool"], c["arguments"]), sort_keys=True, ensure_ascii=False))
                  for c in s.get("prefix_calls") or []]


out = {}
for g, rows in groups.items():
    pairs = defaultdict(dict)
    for s in rows:
        if s["twin"][0] in "AB" and not s["twin"].startswith("A_pushback"):
            pairs[s["pair_id"]][s["twin"][0]] = s
    first = Counter()
    ex = []
    for pid, d in sorted(pairs.items()):
        if "A" not in d or "B" not in d:
            continue
        ua, ra = results(d["A"])
        ub, rb = results(d["B"])
        idx = None
        for k in range(max(len(ra), len(rb))):
            if k >= len(ra) or k >= len(rb) or ra[k] != rb[k]:
                idx = k + 1
                break
        if ua != ub and (idx is None or idx > 1):
            key = "request_differs" + ("" if idx is None else f"+first_result_diff_at_lookup{idx}")
        else:
            key = "no_result_diff" if idx is None else f"first_result_diff_at_lookup{idx}"
        first[key] += 1
        if (idx or 0) >= 2 and len(ex) < 6:
            ex.append({"pair_id": pid, "n_lookups": len(ra), "first_diff_lookup": idx, "lookups": [x[0] for x in ra],
                       "request_A": ua[:100], "request_B": ub[:100]})
    out[g] = {"pairs": sum(first.values()), "where_twins_first_differ": dict(first.most_common()), "examples": ex}
print(json.dumps({g: {k: v for k, v in d.items() if k != "examples"} for g, d in out.items()}, indent=1))
print(json.dumps(out["hard_test"]["examples"], indent=0, ensure_ascii=False)[:2000])
print(dump("s08_multilookup.json", out))
