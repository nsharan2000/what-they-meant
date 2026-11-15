"""6 (continued). Are the twins distinguishable BEFORE any look-up? And what does L2s add?

At evaluation (decision mode) the model first sees [system, request]. If the A and B twins render to the SAME first prompt, a policy
that decides without looking anything up (temperature 0) must give both twins the same answer, so it can never pass both. Measured
per evaluated level (L0r, L1, L2, L2s; P0) for the training scenarios and every evaluation file.
L2s appends the scenario's decisive_facts as "## Most relevant records"; we also count how many of those lines restate a canonical
look-up result (i.e. L2s hands the model the look-up outcome before it looks anything up — never seen in training).
Also: L0r rows whose twin-A target refers to relative time (today / tomorrow / weekday) while the L0r prompt has no date line.

python3 -B experiments/dataset_audit/splits-eval-alignment/s09_first_prompt_identity.py -> results/s09_first_prompt_identity.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from common import dump, eval_files, scen_files, toks, views
from wtm.render import render
from wtm.sim import Simulator
from wtm.spec import load_env, load_member

V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
groups = {"train(trained)": [by_id[x] for x in sorted({r["scenario_id"] for r in V["sft_train"]})]}
for name, rows in eval_files().items():
    groups.setdefault(name.rsplit("_", 1)[0], []).extend(rows)
envs, members = {}, {}


def em(s):
    return (envs.setdefault(s["env_id"], load_env(s["env_id"])),
            members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"])))


out = {}
for g, rows in groups.items():
    pairs = defaultdict(dict)
    for s in rows:
        if s["twin"][0] in "AB" and not s["twin"].startswith("A_pushback"):
            pairs[s["pair_id"]][s["twin"][0]] = s
    c = Counter()
    ex = defaultdict(list)
    ident = defaultdict(list)
    for pid, d in sorted(pairs.items()):
        if "A" not in d or "B" not in d:
            continue
        c["pairs"] += 1
        for lv in ("L0r", "L1", "L2", "L2s"):
            ea, ma = em(d["A"])
            eb, mb = em(d["B"])
            sa, ua, _ = render(ea, ma, d["A"], lv, "P0")
            sb, ub, _ = render(eb, mb, d["B"], lv, "P0")
            same = sa == sb and ua == ub
            c[f"{lv}:identical_first_prompt"] += same
            if same:
                ident[lv].append(pid)
            if not same and len(ex[lv]) < 3:
                ex[lv].append({"pair_id": pid, "system_differs": sa != sb, "request_differs": ua != ub})
    # L2s lines that restate a canonical look-up result
    restate = Counter()
    rex = []
    for s in rows:
        if s["twin"].startswith("A_pushback") or not s.get("decisive_facts"):
            continue
        env, mem = em(s)
        _, _, tools = render(env, mem, s, "L1", "P0")
        sim = Simulator(env, mem, s, tools, level="L1")
        res_text = " ".join(json.dumps(sim.execute(x["tool"], x["arguments"]), ensure_ascii=False) for x in s.get("prefix_calls") or [])
        rt = toks(res_text)
        rule_seed = " ".join(json.dumps(r.get("result"), ensure_ascii=False) for r in s.get("tool_rules", []))
        restate["rows"] += 1
        hit = False
        for f in s["decisive_facts"]:
            ft = toks(f)
            # a fact line 'restates' a look-up when most of its tokens occur in the canonical result or it is a seed result text
            if f and (f in rule_seed or (len(ft) >= 3 and len(ft & rt) / len(ft) >= 0.7)):
                hit = True
                if len(rex) < 4:
                    rex.append({"id": s["id"], "fact": f[:160]})
        restate["rows_with_a_fact_that_restates_a_lookup_result"] += hit
    out[g] = {**dict(c), "examples_not_identical": ex, "identical_first_prompt_pairs": dict(ident),
              "L2s_restates_lookup": dict(restate), "L2s_restate_examples": rex}

# L0r has no 'Today is' line: twin-A message targets that lean on relative time
rel = re.compile(r"\b(today|tomorrow|yesterday|tonight|this (morning|week|month)|next (week|month)|in \d+ days?|"
                 r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
l0 = []
for i, r in enumerate(V["sft_train"]):
    if r["turn"] == "decision" and r["level"] == "L0r":
        t = (r["completion"])[0]["content"]
        if rel.search(t):
            sysm = r["prompt"][0]["content"]
            l0.append({"i": i, "scenario_id": r["scenario_id"], "match": rel.search(t).group(0), "target": t[:160],
                       "prompt_has_iso_date": bool(re.search(r"\b20\d\d-\d\d-\d\d\b", sysm)),
                       "prompt_has_today_word": "today" in " ".join(m["content"] for m in r["prompt"]).lower()})
out["L0r_targets_with_relative_time"] = {"n": len(l0), "rows": l0[:12],
                                         "n_prompt_mentions_today": sum(x["prompt_has_today_word"] for x in l0)}
print(json.dumps({g: {k: v for k, v in d.items() if not k.endswith("examples") and k != "examples_not_identical"}
                  for g, d in out.items() if g != "L0r_targets_with_relative_time"}, indent=1))
print(json.dumps(out["L0r_targets_with_relative_time"], indent=0, ensure_ascii=False)[:2500])
print(dump("s09_first_prompt_identity.json", out))
