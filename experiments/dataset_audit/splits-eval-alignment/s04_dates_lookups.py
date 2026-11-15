"""4/6. Dates, look-up robustness and how the decision turn is reached at evaluation time.

(a) Dates: 'Today is 2026-11-12' (wtm/render.py TODAY) vs ISO dates inside the rendered prompts (system + look-up results) of the
    training decision rows and of the evaluation scenario renderings; dates after today = records from the future.
(b) Look-up robustness: at eval the model issues its OWN look-ups (wtm/run.py decision mode); training shows only the canonical
    prefix_calls. For every evaluated scenario, re-run each canonical look-up with plausible argument variants (lower case, first
    word only, last word only, 'limit' added) through wtm.sim.Simulator and record whether the result still equals the canonical one.
(c) Base Tiny Aya on the held-out eval (eval_base_v12.jsonl): how often its look-ups equal the canonical call, how often the
    result is an error / 'No matching records found.', how many episodes decide without any look-up.

python3 -B experiments/dataset_audit/splits-eval-alignment/s04_dates_lookups.py -> results/s04_dates_lookups.json
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from common import ROOT, dump, eval_files, jl, scen_files, views
from wtm.render import TODAY, render
from wtm.sim import Simulator
from wtm.spec import load_env, load_member

ISO = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
envs, members = {}, {}


def env_mem(s):
    env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
    mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
    return env, mem


out = {"today": TODAY}
# ---------------------------------------------------------------- (a) future dates
fut = defaultdict(Counter)
fut_ex = defaultdict(list)
for i, r in enumerate(V["sft_train"]):
    if r["turn"] != "decision":
        continue
    sysm = r["prompt"][0]["content"]
    rest = "\n".join(m["content"] for m in r["prompt"][1:])
    k = f"sft_train|{r['level']}"
    fut[k]["rows"] += 1
    fs = sorted({m.group(0) for m in ISO.finditer(sysm) if m.group(0) > TODAY and "Today is" not in sysm[max(0, m.start() - 9):m.start()]})
    fr = sorted({m.group(0) for m in ISO.finditer(rest) if m.group(0) > TODAY})
    fut[k]["rows_future_date_in_system"] += bool(fs)
    fut[k]["rows_future_date_in_request_or_results"] += bool(fr)
    if (fs or fr) and len(fut_ex[k]) < 4:
        fut_ex[k].append({"i": i, "scenario_id": r["scenario_id"], "system": fs[:4], "results": fr[:4]})
for name, rows in eval_files().items():
    grp = name.rsplit("_", 1)[0]
    for s in rows:
        env, mem = env_mem(s)
        for lv in ("L0r", "L1", "L2"):
            system, user, tools = render(env, mem, s, lv, "P0")
            sim = Simulator(env, mem, s, tools, level=lv)
            rest = user + "\n" + "\n".join(json.dumps(sim.execute(c["tool"], c["arguments"]), ensure_ascii=False)
                                           for c in s.get("prefix_calls") or [])
            k = f"eval:{grp}|{lv}"
            fut[k]["rows"] += 1
            fs = sorted({m.group(0) for m in ISO.finditer(system) if m.group(0) > TODAY})
            fr = sorted({m.group(0) for m in ISO.finditer(rest) if m.group(0) > TODAY})
            fut[k]["rows_future_date_in_system"] += bool(fs)
            fut[k]["rows_future_date_in_request_or_results"] += bool(fr)
            if (fs or fr) and len(fut_ex[k]) < 4:
                fut_ex[k].append({"id": s["id"], "system": fs[:4], "results": fr[:4]})
out["a_future_dates"] = {k: dict(v) for k, v in sorted(fut.items())}
out["a_future_date_examples"] = fut_ex
# which kind of line carries the future date: a completed transaction dated after today would be a defect; a bill due date or a
# subscription next_charge is legitimate
kinds = Counter()
kind_ex = defaultdict(list)
for i, r in enumerate(V["sft_train"]):
    if r["turn"] != "decision":
        continue
    sec = "?"
    for line in r["prompt"][0]["content"].split("\n"):
        if line.startswith("#"):
            sec = re.sub(r"\s*\(.*", "", line.strip())
        ds = [m.group(0) for m in ISO.finditer(line) if m.group(0) > TODAY]
        if not ds or "Today is" in line:
            continue
        lead = line[2:12] if line.startswith("- ") else ""
        k = ("transaction_dated_after_today" if sec.startswith(("### Transaction history", "### Recent activity")) and lead > TODAY
             else f"future_date_in {sec}")
        kinds[k] += 1
        if len(kind_ex[k]) < 2:
            kind_ex[k].append([i, r["scenario_id"], r["level"], line[:160]])
out["a_future_date_line_kinds_sft_train_decision_rows"] = dict(kinds)
out["a_future_date_line_kind_examples"] = kind_ex

# ---------------------------------------------------------------- (b) look-up robustness for evaluated scenarios
def variants(args):
    outs = []
    for key, val in args.items():
        if isinstance(val, str) and val.strip():
            ws = val.split()
            for tag, nv in (("lower", val.lower()), ("first_word", ws[0]), ("last_word", ws[-1])):
                if nv != val:
                    outs.append((f"{key}:{tag}", {**args, key: nv}))
    outs.append(("add_limit", {**args, "limit": 10}))
    return outs


rob = defaultdict(Counter)
rob_ex = defaultdict(list)
for name, rows in list(eval_files().items()) + [("train_trained", [by_id[x] for x in sorted({r["scenario_id"] for r in V["sft_train"]})])]:
    grp = name.rsplit("_", 1)[0] if name != "train_trained" else name
    for s in rows:
        env, mem = env_mem(s)
        _, _, tools = render(env, mem, s, "L1", "P0")
        for c in s.get("prefix_calls") or []:
            canon = json.dumps(Simulator(env, mem, s, tools, level="L1").execute(c["tool"], c["arguments"]), sort_keys=True)
            has_catch_all = any(r["tool"] == c["tool"] and not r.get("match") for r in s.get("tool_rules", []))
            rob[grp]["calls"] += 1
            rob[grp]["tool_has_catch_all_rule"] += has_catch_all
            for tag, a in variants(c["arguments"]):
                res = json.dumps(Simulator(env, mem, s, tools, level="L1").execute(c["tool"], a), sort_keys=True)
                rob[grp]["variants"] += 1
                same = res == canon
                rob[grp]["variant_same_result"] += same
                if not same:
                    kind = ("no_match" if "no matching records" in res.lower() else "error" if '"error"' in res[:12] else "other_result")
                    rob[grp][f"variant_diff_{kind}"] += 1
                    if len(rob_ex[grp]) < 6:
                        rob_ex[grp].append({"id": s["id"], "tool": c["tool"], "canonical_args": c["arguments"], "variant": tag,
                                            "variant_args": a, "result": res[:160]})
out["b_lookup_robustness"] = {k: dict(v) | {"share_variant_same": round(v["variant_same_result"] / max(1, v["variants"]), 3)}
                              for k, v in rob.items()}
out["b_examples"] = rob_ex

# ---------------------------------------------------------------- (c) what base Tiny Aya actually did at eval (v1.2 test run)
base = jl(os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl"))
bc = defaultdict(Counter)
for r in base:
    s = by_id.get(r["id"])
    canon = [(c["tool"], json.dumps(c["arguments"], sort_keys=True)) for c in (s.get("prefix_calls") or [])] if s else []
    lk = [e for e in r["events"] if e.get("tool_kind") in ("lookup", "verify")]
    k = r["level"]
    bc[k]["episodes"] += 1
    bc[k]["no_lookup_before_decision"] += not lk
    bc[k]["first_action_is_canonical_lookup"] += bool(lk) and r["events"][0].get("tool_kind") in ("lookup", "verify") and \
        (r["events"][0].get("tool"), json.dumps(r["events"][0].get("args"), sort_keys=True)) == (canon[0] if canon else None)
    bc[k]["first_lookup_same_tool_as_canonical"] += bool(lk) and bool(canon) and lk[0].get("tool") == canon[0][0]
    for e in lk:
        txt = json.dumps(e.get("result"), ensure_ascii=False).lower()
        bc[k]["lookups"] += 1
        bc[k]["lookup_result_no_match_or_error"] += ("no matching records" in txt) or txt.startswith('{"error"')
    bc[k]["narration_turns"] += sum(bool(e.get("narration")) for e in r["events"])
    bc[k]["decision_" + r["decision_grade"]["decision"]] += 1
out["c_base_eval_v12_lookup_behaviour"] = {k: dict(v) for k, v in sorted(bc.items())}
print(json.dumps(out["a_future_dates"], indent=0)[:2500])
print(json.dumps(out["b_lookup_robustness"], indent=0))
print(json.dumps(out["c_base_eval_v12_lookup_behaviour"], indent=0)[:3000])
print(dump("s04_dates_lookups.json", out))
