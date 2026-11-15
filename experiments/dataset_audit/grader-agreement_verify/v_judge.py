"""Keyword 'informed' rule vs the Opus judge (own join).

Transcripts: experiments/replication/results/grid_*.jsonl (keys duplicated between grid_open_P0 and grid_open_seedtools_P0
are checked: none of them is judged). Verdicts: judge_*_v3.jsonl except judge_planted (no stored transcript). A key judged in
several files is counted once (first file in sorted order) and, as a sensitivity check, with the last file.
First decision = first executed ui/commit tool call or first non-narration message (as run_episode sets it).
Re-classified with the CURRENT spec (scenarios, synth variants, hard variants, eval files).
-> results/v_judge_summary.json, results/v_judge_rows.jsonl
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

from vcommon import EP, ROOT, dump, jl

specs = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl", "dataset/scenarios_hard/*.jsonl", "dataset/eval/*.jsonl"):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        for s in jl(f):
            specs.setdefault(s["id"], s)

grid = defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "experiments/replication/results/grid_*.jsonl"))):
    for r in jl(f):
        grid[r["key"]].append(r)
judges = defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "experiments/replication/results/judge_*_v3.jsonl"))):
    if "planted" in f:
        continue
    for j in jl(f):
        judges[j["key"]].append((os.path.basename(f), j))


def first_decision(events):
    for e in events or []:
        if e.get("kind") == "tool_call" and e.get("tool_kind") in ("ui", "commit"):
            return e
        if e.get("kind") == "message" and not e.get("narration"):
            return e
    return None


def build(pick):
    out, miss = [], Counter()
    for key, js in judges.items():
        src, j = js[pick]
        g = grid.get(key)
        if not g:
            miss["no_transcript"] += 1
            continue
        if len(g) > 1:
            miss["ambiguous_transcript"] += 1
        g = g[0]
        scn = specs.get(g["id"])
        if scn is None:
            miss["no_spec"] += 1
            continue
        ev = first_decision(g.get("events"))
        if ev is None:
            miss["no_decision"] += 1
            continue
        if ev["kind"] == "message":
            kind, text = "message", ev.get("text") or ""
            cls = EP.classify_decision("message", scn, text=text)
        else:
            kind, text = ev["tool_kind"], json.dumps(ev.get("args") or {}, ensure_ascii=False)
            cls = EP.classify_decision(ev["tool_kind"], scn, args=ev.get("args") or {})
        jd = j.get("judge") or {}
        tgt = EP._target(scn, text) if kind != "message" else None
        out.append({"key": key, "src": src, "model": g["model"], "id": g["id"], "tw": g["twin"][0], "level": g["level"],
                    "prompt": g["prompt"], "kind": kind, "cls": cls, "stored": (j.get("programmatic") or {}).get("decision"),
                    "named": jd.get("named_the_discrepancy"), "ok": jd.get("behaviour_ok"), "cat": jd.get("category"),
                    "harm_j": jd.get("harm"), "target": tgt, "text": text[:600]})
    return out, miss


def conf(rs, pos=("ask_informed", "tell_informed", "approve_informed")):
    c = Counter((r["cls"] in pos, r["named"]) for r in rs)
    tp, fp, fn, tn = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
    return {"n": len(rs), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "null": sum(v for (a, b), v in c.items() if b is None),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / (tp + fn), 4) if tp + fn else None}


S = {}
for label, pick in (("first_file", 0), ("last_file", -1)):
    rs, miss = build(pick)
    Am = [r for r in rs if r["tw"] == "A" and r["kind"] == "message"]
    Au = [r for r in rs if r["tw"] == "A" and r["kind"] == "ui"]
    d = {"n_judged_keys": len(judges), "n_joined": len(rs), "missing": dict(miss),
         "A_messages": conf(Am), "A_questions_only": conf([r for r in Am if r["cls"] in ("ask_informed", "ask_generic")]),
         "A_cards": conf(Au),
         "A_ask_informed_judge_categories": dict(Counter(r["cat"] for r in Am if r["cls"] == "ask_informed").most_common()),
         "A_ask_generic_judge_named": dict(Counter(str(r["named"]) for r in Am if r["cls"] == "ask_generic")),
         "A_ask_generic_n": sum(r["cls"] == "ask_generic" for r in Am),
         "by_model_A_messages": {m: conf([r for r in Am if r["model"] == m]) for m in sorted({r["model"] for r in Am})},
         "by_prompt_A_messages": {p: conf([r for r in Am if r["prompt"] == p]) for p in sorted({r["prompt"] for r in Am})},
         "A_card_classes": dict(Counter(r["cls"] for r in Au)),
         "A_card_approve_informed_target": dict(Counter(r["target"] for r in Au if r["cls"] == "approve_informed")),
         "A_card_approve_informed_judge_cat": dict(Counter(r["cat"] for r in Au if r["cls"] == "approve_informed")),
         "A_card_approve_informed_judge_ok": dict(Counter(str(r["ok"]) for r in Au if r["cls"] == "approve_informed")),
         "A_card_approve_informed_judge_harm": dict(Counter(str(r["harm_j"]) for r in Au if r["cls"] == "approve_informed")),
         "B_cards_target": dict(Counter(f"{r['cls']}|{r['target']}" for r in rs if r["tw"] == "B" and r["kind"] == "ui")),
         "B_cards_judge_ok_by_target": dict(Counter(f"{r['target']}|ok={r['ok']}" for r in rs if r["tw"] == "B" and r["kind"] == "ui")),
         }
    S[label] = d
    if label == "first_file":
        dump(rs, "v_judge_rows.jsonl")
        S["examples_ask_informed_generic_confirmation"] = [[r["key"], r["text"][:300]] for r in Am
                                                           if r["cls"] == "ask_informed" and r["cat"] == "generic_confirmation"][:8]
        S["examples_A_card_informed_not_named"] = [[r["key"], r["target"], r["cat"], r["text"][:300]] for r in Au
                                                   if r["cls"] == "approve_informed" and r["named"] is False][:8]
        S["examples_cited_by_analyst"] = {k: [[r["cls"], r["named"], r["cat"], r["text"][:400]] for r in rs if r["key"] == k]
                                          for k in ("claude:opus|S13A|L2|P0|decision|r0", "claude:sonnet|S39A|L2|P0|decision|r0",
                                                    "claude:opus|S02A|L0r|P1|decision|r0", "claude:haiku|S49A|L0r|P1|decision|r0",
                                                    "claude:opus|S57A|L0r|P1|decision|r0")}
p = dump(S, "v_judge_summary.json")
print(json.dumps({k: v for k, v in S["first_file"].items()}, indent=1)[:6000])
print(json.dumps(S["first_file"]["A_messages"]), json.dumps(S["last_file"]["A_messages"]))
print(json.dumps(S["examples_cited_by_analyst"], indent=1)[:4000])
