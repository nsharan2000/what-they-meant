"""Why do twin A and B system prompts differ at L0r (16 % of pairs) but rarely at L1/L2? Section-level diff of the A vs B system
prompts (hint stripped) for every pair x level in sft_train decision rows and for the evaluation files at L0r: member card vs tool
list. A tool list that differs between twins would let the rendering itself reveal the label.

python3 -B experiments/dataset_audit/splits-eval-alignment/s14_l0r_twin_system_diff.py -> results/s14_l0r_twin_system_diff.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from common import dump, eval_files, scen_files, views
from wtm.render import golden_text, render
from wtm.spec import load_env, load_member

HINT = golden_text("P2")
V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}


def sections(s):
    i_t, i_p = s.find("## Tools\n"), s.find("## How to act\n")
    return {"before_tools": s[:i_t], "tools": s[i_t:i_p], "after_tools": s[i_p:].replace("\n\n" + HINT, "")}


out = {}
dec = defaultdict(dict)
for r in V["sft_train"]:
    if r["turn"] == "decision" and r["twin"][0] in "AB" and not r["twin"].startswith("A_pushback"):
        dec[(r["pair_id"], r["level"])][r["twin"][0]] = r
cnt = Counter()
ex = defaultdict(list)
for (pid, lv), d in sorted(dec.items()):
    if "A" not in d or "B" not in d:
        continue
    sa, sb = sections(d["A"]["prompt"][0]["content"]), sections(d["B"]["prompt"][0]["content"])
    diff = [k for k in sa if sa[k] != sb[k]]
    cnt[f"{lv}|pairs"] += 1
    for k in diff:
        cnt[f"{lv}|differs:{k}"] += 1
    if "tools" in diff:
        ta = {l.split("(")[0][2:] for l in sa["tools"].splitlines() if l.startswith("- ")}
        tb = {l.split("(")[0][2:] for l in sb["tools"].splitlines() if l.startswith("- ")}
        if len(ex[lv]) < 8:
            ex[lv].append({"pair_id": pid, "tools_only_in_A": sorted(ta - tb), "tools_only_in_B": sorted(tb - ta)})
out["sft_train_decision_rows"] = dict(sorted(cnt.items()))
out["tool_list_diff_examples"] = ex
# evaluation files at L0r
envs, members = {}, {}
ce = Counter()
eex = []
for name, rows in eval_files().items():
    g = name.rsplit("_", 1)[0]
    pairs = defaultdict(dict)
    for s in rows:
        if s["twin"][0] in "AB" and not s["twin"].startswith("A_pushback"):
            pairs[s["pair_id"]][s["twin"][0]] = s
    for pid, d in pairs.items():
        if "A" not in d or "B" not in d:
            continue
        rend = {}
        for tw in "AB":
            s = d[tw]
            env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
            mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
            rend[tw] = render(env, mem, s, "L0r", "P0")
        ce[f"{g}|pairs"] += 1
        ta, tb = set(rend["A"][2]), set(rend["B"][2])
        if ta != tb:
            ce[f"{g}|tool_list_differs"] += 1
            if len(eex) < 8:
                eex.append({"file": name, "pair_id": pid, "only_A": sorted(ta - tb), "only_B": sorted(tb - ta)})
out["eval_L0r"] = dict(sorted(ce.items()))
out["eval_L0r_examples"] = eex
print(json.dumps(out, indent=1))
print(dump("s14_l0r_twin_system_diff.json", out))
