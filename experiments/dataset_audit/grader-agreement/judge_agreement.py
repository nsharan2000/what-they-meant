"""Keyword 'informed' rule vs an independent reader: the Opus judge's named_the_discrepancy on stored transcripts.

Joins experiments/replication/results/judge_*_v3.jsonl (one verdict per transcript key; planted rows excluded because their
transcripts are not stored) with the transcripts in grid_*.jsonl, finds the FIRST DECISION event exactly as run_episode sets it
(first ui/commit tool call that executed, or first non-narration message), and re-classifies it with the CURRENT scenario spec
(wtm.episode.classify_decision), so the comparison is about the grader the training reward and the evaluation use today.
The judge is an LLM (claude-opus-5-5, judge v3) — a second opinion, not ground truth.

-> results/judge_agreement_rows.jsonl, results/judge_agreement_summary.json
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

from ga_common import OUT, ROOT, classify_decision, decision_grade, dump_json, dump_jsonl, load_specs

INFORMED = {"ask_informed", "tell_informed", "approve_informed"}


def first_decision(events):
    for e in events:
        if e.get("kind") == "tool_call" and e.get("tool_kind") in ("ui", "commit"):
            return e
        if e.get("kind") == "message" and not e.get("narration"):
            return e
    return None


def main():
    specs = load_specs()
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        for l in open(f):
            if l.strip():
                s = json.loads(l)
                specs.setdefault(s["id"], s)
    grid = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "replication", "results", "grid_*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                grid[r["key"]] = r
    judged = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "replication", "results", "judge_*_v3.jsonl"))):
        if "planted" in f:
            continue
        for l in open(f):
            if l.strip():
                j = json.loads(l)
                judged.setdefault(j["key"], (os.path.basename(f), j))
    rows, missing = [], Counter()
    for key, (src, j) in judged.items():
        g = grid.get(key)
        if g is None:
            missing["no_transcript"] += 1
            continue
        scn = specs.get(g["id"])
        if scn is None:
            missing["no_spec"] += 1
            continue
        ev = first_decision(g.get("events") or [])
        if ev is None:
            missing["no_decision_event"] += 1
            continue
        if ev["kind"] == "message":
            text = ev.get("text") or ""
            cls = classify_decision("message", scn, text=text)
        else:
            text = json.dumps(ev.get("args") or {}, ensure_ascii=False)
            cls = classify_decision(ev["tool_kind"], scn, args=ev.get("args") or {})
        jd = j.get("judge") or {}
        rows.append({"key": key, "judge_file": src, "model": g["model"], "id": g["id"], "twin": g["twin"][:1],
                     "level": g["level"], "prompt": g["prompt"], "kind": ev["kind"] if ev["kind"] == "message" else ev["tool_kind"],
                     "class_stored": (j.get("programmatic") or {}).get("decision"), "class_now": cls,
                     "kw_informed": cls in INFORMED, "success_now": decision_grade(scn, cls)["success"],
                     "judge_named": jd.get("named_the_discrepancy"), "judge_ok": jd.get("behaviour_ok"),
                     "judge_category": jd.get("category"), "rationale": (jd.get("rationale") or "")[:400], "text": text[:700],
                     "informed_any": scn["expected"].get("informed_any")})
    dump_jsonl(rows, f"{OUT}/judge_agreement_rows.jsonl")

    def confusion(rs):
        c = Counter((r["kw_informed"], r["judge_named"]) for r in rs)
        tp, fp = c[(True, True)], c[(True, False)]
        fn, tn = c[(False, True)], c[(False, False)]
        none = sum(v for (a, b), v in c.items() if b is None)
        return {"n": len(rs), "kw_yes_judge_yes": tp, "kw_yes_judge_no": fp, "kw_no_judge_yes": fn, "kw_no_judge_no": tn,
                "judge_null": none,
                "precision_kw_vs_judge": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall_kw_vs_judge": round(tp / (tp + fn), 4) if tp + fn else None,
                "agreement_on_judged": round((tp + tn) / (tp + fp + fn + tn), 4) if tp + fp + fn + tn else None}

    summ = {"n_judged_keys": len(judged), "n_joined": len(rows), "missing": dict(missing),
            "class_changed_since_stored": sum(r["class_now"] != r["class_stored"] for r in rows),
            "class_change_pairs": dict(Counter(f"{r['class_stored']}->{r['class_now']}" for r in rows
                                               if r["class_now"] != r["class_stored"]).most_common(12)),
            "A_messages": confusion([r for r in rows if r["twin"] == "A" and r["kind"] == "message"]),
            "A_messages_questions_only": confusion([r for r in rows if r["twin"] == "A" and r["class_now"] in ("ask_informed", "ask_generic")]),
            "A_approval_cards": confusion([r for r in rows if r["twin"] == "A" and r["kind"] == "ui"]),
            "B_messages": confusion([r for r in rows if r["twin"] == "B" and r["kind"] == "message"]),
            "by_model_A_messages": {m: confusion([r for r in rows if r["model"] == m and r["twin"] == "A" and r["kind"] == "message"])
                                    for m in sorted({r["model"] for r in rows})},
            "strict_success_vs_judge_ok": {}}
    for tw in ("A", "B"):
        rs = [r for r in rows if r["twin"] == tw and r["judge_ok"] is not None]
        c = Counter((bool(r["success_now"]), bool(r["judge_ok"])) for r in rs)
        summ["strict_success_vs_judge_ok"][tw] = {"n": len(rs), "both_ok": c[(True, True)], "grader_ok_judge_not": c[(True, False)],
                                                  "grader_not_judge_ok": c[(False, True)], "both_not": c[(False, False)]}
    for tw in ("A", "B"):
        rs = [r for r in rows if r["twin"] == tw and r["judge_ok"] is not None]
        summ["strict_success_vs_judge_ok"][tw]["grader_not_judge_ok_by_class|category"] = dict(Counter(
            f"{r['class_now']}|{r['judge_category']}" for r in rs if not r["success_now"] and r["judge_ok"]).most_common(10))
        summ["strict_success_vs_judge_ok"][tw]["grader_ok_judge_not_by_class|category"] = dict(Counter(
            f"{r['class_now']}|{r['judge_category']}" for r in rs if r["success_now"] and not r["judge_ok"]).most_common(10))
    # examples
    def ex(rs, n=6):
        return [{k: r[k] for k in ("key", "class_now", "judge_named", "judge_category", "text", "rationale")} for r in rs[:n]]
    A_msg = [r for r in rows if r["twin"] == "A" and r["kind"] == "message"]
    summ["examples_kw_generic_but_judge_named"] = ex([r for r in A_msg if not r["kw_informed"] and r["judge_named"] is True], 10)
    summ["examples_kw_informed_but_judge_not_named"] = ex([r for r in A_msg if r["kw_informed"] and r["judge_named"] is False], 10)
    summ["examples_A_approval_informed_judge_not_named"] = ex([r for r in rows if r["twin"] == "A" and r["kind"] == "ui"
                                                               and r["kw_informed"] and r["judge_named"] is False], 6)
    summ["examples_B_ask_informed"] = ex([r for r in rows if r["twin"] == "B" and r["class_now"] == "ask_informed"], 8)
    summ["B_ask_informed_judge_category"] = dict(Counter(r["judge_category"] for r in rows
                                                         if r["twin"] == "B" and r["class_now"] == "ask_informed"))
    summ["A_ask_generic_judge_category"] = dict(Counter(r["judge_category"] for r in rows
                                                        if r["twin"] == "A" and r["class_now"] == "ask_generic"))
    summ["A_ask_generic_judge_named"] = dict(Counter(str(r["judge_named"]) for r in rows
                                                     if r["twin"] == "A" and r["class_now"] == "ask_generic"))
    dump_json(summ, f"{OUT}/judge_agreement_summary.json")
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith("examples")}, indent=1)[:7000])


if __name__ == "__main__":
    main()
