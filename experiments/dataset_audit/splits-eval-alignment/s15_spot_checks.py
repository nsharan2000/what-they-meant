"""Spot checks cited as examples in findings.json (saved so every quoted fact has a script and an output).

(a) held-out twin-B items whose expected decision is 'act', and how an approval card is graded on them (raw and adjudicated);
(b) the ruth_price world: V541B target and records vs the member file's lawn-mowing payments and the held-out items' ideals;
(c) environment-policy sentences about asking / confirming / approval cards (L1 template);
(d) sft_train row indices (0-based line numbers) of the training scenarios used as examples;
(e) held-out rows whose sub_domain or tags never occur in training.

python3 -B experiments/dataset_audit/splits-eval-alignment/s15_spot_checks.py -> results/s15_spot_checks.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from common import dump, eval_files, scen_files, views
from wtm.episode import decision_grade
from wtm.labels import adjudicate
from wtm.spec import load_env, load_member

V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
ev = eval_files()
out = {}

# (a)
acts = []
for name, rows in ev.items():
    for s in rows:
        if s["twin"].startswith("B") and s["expected"]["decision"] == "act":
            g_raw = decision_grade(s, "approve_generic")
            g_adj = decision_grade(adjudicate(s), "approve_generic")
            acts.append({"file": name, "id": s["id"], "ideal_tool": (s.get("ideal_action") or {}).get("name"),
                         "card_strict_raw": g_raw["success"], "card_lenient_raw": g_raw["success_lenient"],
                         "card_strict_adjudicated": g_adj["success"], "request": s["user_message"][:90]})
out["a_act_items"] = acts
out["a_counts"] = {g: sum(1 for a in acts if a["file"].rsplit("_", 1)[0] == g) for g in ("test", "hard_test", "val", "hard_val", "hard_train")}

# (b)
mem = load_member("wallet", "ruth_price")
lawn = [t for t in mem.get("transactions", []) if "lawn" in json.dumps(t, ensure_ascii=False).lower()]
out["b_ruth_price_lawn_transactions"] = {"n": len(lawn), "counterparties": sorted({t.get("counterparty") for t in lawn}),
                                         "amounts": sorted({t.get("amount") for t in lawn})}
v = by_id["V541B"]
out["b_V541B"] = {"target": v["ideal_action"], "scenario_records": v.get("scenario_records"), "pair_split": "train",
                  "lookup_rules": [r.get("result") for r in v.get("tool_rules", [])]}
out["b_heldout_ideals"] = {sid: (by_id[sid].get("ideal_action") or {}).get("text") or (by_id[sid].get("ideal_action") or {}).get("arguments")
                           for sid in ("S56A", "S56B", "V082A", "V082B", "V234A", "V234B")}

# (c)
pol = {}
for env_id in ("wallet", "bills", "shopping", "bizops"):
    t = load_env(env_id)["system_prompt"]["L1"]
    sents = re.split(r"(?<=[.!?])\s+|\n", t)
    pol[env_id] = [x.strip() for x in sents if re.search(r"\b(ask|question|clarif|confirm|approval card)\w*", x, re.I)]
out["c_policy_sentences"] = pol

# (d)
idx = defaultdict(list)
for i, r in enumerate(V["sft_train"]):
    if r["scenario_id"] in ("V492A", "V492B", "S06A", "S49B", "V474A", "V541B", "S14B", "V104B", "S12A", "S12B", "S54A", "S54B"):
        idx[r["scenario_id"]].append([i, r["level"], r["prompt_setting"], r["turn"]])
out["d_sft_train_rows"] = dict(idx)

# (e)
trained = {r["scenario_id"] for r in V["sft_train"]}
t_sub = {by_id[x].get("sub_domain") for x in trained}
t_tags = {t for x in trained for t in by_id[x].get("tags", [])}
new = defaultdict(list)
for name, rows in ev.items():
    g = name.rsplit("_", 1)[0]
    for s in rows:
        why = []
        if s.get("sub_domain") not in t_sub:
            why.append("sub_domain:" + str(s.get("sub_domain")))
        why += ["tag:" + t for t in s.get("tags", []) if t not in t_tags]
        if why:
            new[g].append([s["id"], why])
out["e_heldout_rows_with_untrained_subdomain_or_tag"] = dict(new)
out["e_counts"] = {g: {"rows": len(v_), "untrained_sub_domain": sum(any(w.startswith("sub_domain:") for w in x[1]) for x in v_),
                       "untrained_tag": sum(any(w.startswith("tag:") for w in x[1]) for x in v_)} for g, v_ in new.items()}
# (g) are canonical look-ups served by a scenario rule or by a member-data handler? (build_datasets' Simulator gets the base member,
#     the eval's Simulator gets member + scenario_records: only handler-served look-ups could differ)
from wtm.sim import _args_text  # noqa: E402
from wtm.spec import all_hit  # noqa: E402
served = {"rule": 0, "handler": 0}
handler_rows = []
trained_ids = {r["scenario_id"] for r in V["sft_train"]}
for s in by_id.values():
    tools = {t["name"]: t for t in load_env(s["env_id"])["tools"]}
    for c in s.get("prefix_calls") or []:
        hit = any(r["tool"] == c["tool"] and all_hit(_args_text(c["arguments"]), r.get("match", [])) for r in s.get("tool_rules", []))
        served["rule" if hit else "handler"] += 1
        if not hit:
            handler_rows.append([s["id"], c["tool"], tools.get(c["tool"], {}).get("handler", {}).get("type"), s["id"] in trained_ids])
out["g_prefix_lookup_serving"] = served
out["g_handler_served"] = handler_rows

# (f) member files of the twins whose system prompts differ (S12, S54) and of the shared-world examples
out["f_member_ids"] = {sid: by_id[sid]["member_id"] for sid in ("S12A", "S12B", "S54A", "S54B", "H231A", "V492A", "H201A", "S06A",
                                                               "V232A", "S49B", "S01A", "V474A", "V082A", "V541B")}
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
print(dump("s15_spot_checks.json", out))
