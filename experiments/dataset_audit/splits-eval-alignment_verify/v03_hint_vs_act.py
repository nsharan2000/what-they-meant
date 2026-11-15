"""SEA-3 verification: the P2 hint ('if everything matches, just show the approval card') vs 'act' labels.

(a) every twin-B row with expected 'act' in each eval file; grade an approval card with the repo grader (raw spec, and with
    label_adjudications applied) to see strict/lenient outcome;
(b) training: P2 decision rows by twin and target class (commit / approval card / message); P2 twin-B rows whose target is a
    direct commit; adjudicated rows (accept includes 'act') and their targets;
(c) the hint text itself.
Output: results/v03_hint_vs_act.json
"""
import json
import os
from collections import Counter

from vcommon import ROOT, curated, dump, eval_sets, specs, target_text, env
from wtm.episode import classify_decision, decision_grade
from wtm.labels import adjudicate
from wtm.protocol import parse_action

hint = open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()
ev = eval_sets()
A = {}
for name, rows in ev.items():
    acts = []
    for s in rows:
        if s["twin"].startswith("B") and s["expected"]["decision"] == "act":
            card_args = {"summary": "Proceed with the request as described?"}
            dec = classify_decision("ui", s, args=card_args)
            g_raw = decision_grade(s, dec)
            g_adj = decision_grade(adjudicate(s), dec)
            acts.append({"id": s["id"], "decision_for_card": dec, "strict_raw": g_raw["success"],
                         "lenient_raw": g_raw["success_lenient"], "strict_adj": g_adj["success"],
                         "ideal_type": (s.get("ideal_action") or {}).get("type"),
                         "ideal_name": (s.get("ideal_action") or {}).get("name")})
    # also rows whose adjudication accepts act
    adj_rows = [s["id"] for s in rows if s["twin"].startswith("B") and "act" in (adjudicate(s)["expected"].get("accept") or [])]
    A[name] = {"n_pairs": len({s["pair_id"] for s in rows}), "act_rows": acts, "n_act_rows": len(acts),
               "rows_with_adjudicated_accept_act": adj_rows}

# (b) training rows
sft = curated("sft_train")
by_id = {s["id"]: s for s in specs()}
cls = Counter()
p2_b_commit, p2_a_non_ask, adj_targets = [], [], []
for i, r in enumerate(sft):
    if r["turn"] != "decision":
        continue
    s = by_id[r["scenario_id"]]
    kinds = {t["name"]: t["kind"] for t in env(s["env_id"])["tools"]}
    act = parse_action(target_text(r), set(kinds))
    if act.kind == "message":
        tclass = "message"
    else:
        tclass = kinds.get(act.name, "?")
    cls[(r["prompt_setting"], r["twin"][0], s["expected"]["decision"], tclass)] += 1
    if r["prompt_setting"] == "P2" and r["twin"].startswith("B") and tclass == "commit":
        p2_b_commit.append([i, r["scenario_id"], r["level"], target_text(r)[:90]])
    if r["prompt_setting"] == "P2" and r["twin"].startswith("A"):
        dec = classify_decision("message", s, text=act.text) if act.kind == "message" else tclass
        if dec not in ("ask_informed",):
            p2_a_non_ask.append([i, r["scenario_id"], r["level"], dec, target_text(r)[:90]])
    if r["twin"].startswith("B") and "act" in (adjudicate(s)["expected"].get("accept") or []):
        adj_targets.append([i, r["scenario_id"], r["level"], r["prompt_setting"], tclass])

all_b_commit = sum(v for (p, t, e, c), v in cls.items() if t == "B" and c == "commit")
B = {"decision_rows_by(prompt,twin,expected,target_kind)": {"|".join(k): v for k, v in sorted(cls.items())},
     "p2_twinB_rows_with_commit_target": p2_b_commit, "n_p2_twinB_commit": len(p2_b_commit),
     "n_all_twinB_commit_targets(P0+P2)": all_b_commit,
     "p2_twinA_rows_not_ask_informed": p2_a_non_ask,
     "adjudicated_accept_act_training_rows(line,id,level,prompt,target_kind)": adj_targets}

dump("v03_hint_vs_act.json", {"hint": hint, "A_eval_act_items": A, "B_training": B})
print(hint)
for k, v in A.items():
    print(k, v["n_pairs"], v["n_act_rows"], [(x["id"], x["strict_raw"], x["lenient_raw"], x["strict_adj"]) for x in v["act_rows"]],
          "adj-accept-act:", v["rows_with_adjudicated_accept_act"])
print(json.dumps(B, indent=0)[:3000])
