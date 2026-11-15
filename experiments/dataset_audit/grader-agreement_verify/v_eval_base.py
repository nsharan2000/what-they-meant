"""GA-3 re-derivation on the base-model eval file and a read of every B-act commit that is not commit_gold.

Spec = the eval spec files (dataset/eval/*.jsonl), the ones run_eval.sh renders. Decision re-classified from the stored first
decision event with the current spec, then graded with wtm.episode.decision_grade (raw spec, as wtm/run.py does) and with the
spec adjudicated (wtm.labels.adjudicate, as the replication analysis does).
-> results/v_eval_base_summary.json, results/v_eval_base_bact_commits.jsonl
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

from vcommon import EP, ROOT, SP, adjudicate, dump, jl

specs = {}
for f in sorted(glob.glob(os.path.join(ROOT, "dataset/eval/*.jsonl"))):
    for s in jl(f):
        specs[s["id"]] = (os.path.basename(f), s)
rows = jl(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl"))


def first(events):
    for e in events or []:
        if e.get("kind") == "tool_call" and e.get("tool_kind") in ("ui", "commit"):
            return e
        if e.get("kind") == "message" and not e.get("narration"):
            return e
    return None


recs = []
for r in rows:
    fn, scn = specs[r["id"]]
    ev = first(r["events"])
    if ev is None:
        cls = r["decision_grade"]["decision"]
    elif ev["kind"] == "message":
        cls = EP.classify_decision("message", scn, text=ev.get("text") or "")
    else:
        cls = EP.classify_decision(ev["tool_kind"], scn, args=ev.get("args") or {})
    g = EP.decision_grade(scn, cls)
    ga = EP.decision_grade(adjudicate(scn), cls)
    exp = scn["expected"]["decision"]
    recs.append({"key": r["key"], "file": fn, "id": r["id"], "pair": r["pair_id"], "tw": r["twin"][0], "exp": exp,
                 "lv": r["level"], "ps": r["prompt"], "cls": cls, "stored": r["decision_grade"]["decision"],
                 "succ": g["success"], "succ_adj": ga["success"], "harm": g["harm"],
                 "tool": ev.get("tool") if ev else None, "args": ev.get("args") if ev else None})
S = {"n": len(recs), "class_changed_vs_stored": sum(r["cls"] != r["stored"] for r in recs),
     "files": dict(Counter(r["file"].split("_")[0] if not r["file"].startswith("hard") else "_".join(r["file"].split("_")[:2]) for r in recs))}
B = [r for r in recs if r["tw"] == "B"]
Bact = [r for r in B if r["exp"] == "act"]
S["B_n"] = len(B)
S["B_act_n"] = len(Bact)
S["B_act_classes"] = dict(Counter(r["cls"] for r in Bact))
S["B_act_strict_success"] = sum(r["succ"] for r in Bact)
S["B_act_strict_by_class"] = dict(Counter(r["cls"] for r in Bact if r["succ"]))
S["B_act_success_and_harm"] = sum(r["succ"] and r["harm"] for r in Bact)
S["B_success_current"] = sum(r["succ"] for r in B)


def alt_success(r):
    if r["exp"] == "act":
        return r["cls"] == "commit_gold"
    return r["succ"]


S["B_success_commit_gold_rule"] = sum(alt_success(r) for r in B)
S["B_success_adjudicated"] = sum(r["succ_adj"] for r in B)
pairs = defaultdict(dict)
for r in recs:
    pairs[(r["pair"], r["lv"], r["ps"])][r["tw"]] = r
full = [d for d in pairs.values() if "A" in d and "B" in d]
S["pairs"] = len(full)
S["paired_current"] = sum(d["A"]["succ"] and d["B"]["succ"] for d in full)
S["paired_commit_gold_rule"] = sum(d["A"]["succ"] and alt_success(d["B"]) for d in full)
S["A_success"] = sum(r["succ"] for r in recs if r["tw"] == "A")
# B-act specs: gold / harmful signatures
bact_specs = sorted({r["id"] for r in Bact})
S["B_act_specs"] = {sid: {"file": specs[sid][0], "gold_commits": specs[sid][1].get("gold_commits"),
                          "harmful_commits": specs[sid][1].get("harmful_commits"),
                          "ideal": specs[sid][1].get("ideal_action"),
                          "ideal_class": (EP.classify_decision("commit", specs[sid][1], args=(specs[sid][1].get("ideal_action") or {}).get("arguments"))
                                          if (specs[sid][1].get("ideal_action") or {}).get("type") == "tool_call" else None)}
                    for sid in bact_specs}
S["B_act_by_split"] = dict(Counter(f"{r['file'].rsplit('_', 1)[0]}|{r['cls']}" for r in Bact))
dump([{k: r[k] for k in ("key", "id", "lv", "ps", "cls", "tool", "args")} for r in Bact if r["cls"].startswith("commit")],
     "v_eval_base_bact_commits.jsonl")
# B cards on all B rows: target signature
S["B_card_targets"] = dict(Counter(f"{r['cls']}|{EP._target(specs[r['id']][1], json.dumps(r['args'] or {}, ensure_ascii=False))}"
                                   for r in B if r["cls"].startswith("approve")))
S["A_card_targets"] = dict(Counter(f"{r['cls']}|{EP._target(specs[r['id']][1], json.dumps(r['args'] or {}, ensure_ascii=False))}"
                                   for r in recs if r["tw"] == "A" and r["cls"].startswith("approve")))
S["A_classes"] = dict(Counter(r["cls"] for r in recs if r["tw"] == "A"))
S["B_classes"] = dict(Counter(r["cls"] for r in B))
dump(S, "v_eval_base_summary.json")
print(json.dumps({k: v for k, v in S.items() if k != "B_act_specs"}, indent=1))
for sid, d in S["B_act_specs"].items():
    print(sid, d["file"], "gold", d["gold_commits"], "| harmful", d["harmful_commits"], "| ideal", json.dumps(d["ideal"])[:200], "->", d["ideal_class"])
