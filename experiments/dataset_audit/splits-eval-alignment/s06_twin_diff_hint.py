"""5/6 (continued). Where the A-vs-B signal sits, and whether the P2 hint agrees with the targets it is trained with.

(a) Twin diff: for every pair x level where sft_train has BOTH the A and the B decision row, diff the two prompts (P2 hint removed):
    which messages differ (system / request / look-up call / look-up result), how many tokens differ (tokmodel estimate), and the
    token offset from the end of the earliest difference. This is the evidence that separates 'ask' from 'proceed'.
(b) Hint vs targets: the P2 hint says "If anything clashes, ask me one short question that names it; if everything matches, just
    show the approval card." Count P2 training targets that contradict it: twin-B targets that commit directly (not a card), and
    twin-A targets that are not graded as a question (wtm.episode.classify_decision on the target itself).
(c) M1 policy samples on training prompts (m2_grpo/scores.jsonl) by prompt setting x twin, and twin-B 'act' rows by decision.

python3 -B experiments/dataset_audit/splits-eval-alignment/s06_twin_diff_hint.py -> results/s06_twin_diff_hint.json
"""
from __future__ import annotations

import difflib
import json
import os
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, dump, jl, scen_files, views
from tokmodel import FEATS, est_text, weights
from wtm.episode import classify_decision, decision_grade
from wtm.labels import adjudicate
from wtm.protocol import parse_action
from wtm.render import golden_text

HINT = golden_text("P2")
V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
stats = {(s["file"], s["i"]): s for s in jl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))}
w = weights()
PER_MSG, CONST = w[len(FEATS)], w[len(FEATS) + 1]
out = {}


def strip_hint(prompt):
    p = json.loads(json.dumps(prompt))
    p[0]["content"] = p[0]["content"].replace("\n\n" + HINT, "").replace(HINT, "")
    return p


# ---------------------------------------------------------------- (a) twin diff
dec = defaultdict(dict)
for i, r in enumerate(V["sft_train"]):
    if r["turn"] == "decision" and r["twin"][0] in "AB" and not r["twin"].startswith("A_pushback"):
        dec[(r["pair_id"], r["level"])][r["twin"][0]] = (i, r)
res = defaultdict(lambda: defaultdict(list))
examples = []
for (pid, lv), d in sorted(dec.items()):
    if "A" not in d or "B" not in d:
        continue
    (ia, a), (ib, b) = d["A"], d["B"]
    pa, pb = strip_hint(a["prompt"]), strip_hint(b["prompt"])
    where = []
    if len(pa) != len(pb):
        where.append("n_messages")
    diff_tok, first_off = 0.0, None
    # the A prompt's own token layout (exact total from token_stats; rescaled estimate inside)
    msg_tok = [est_text(m["content"]) + PER_MSG for m in pa]
    total_est = CONST + sum(msg_tok)
    exact = stats[("sft_train.jsonl", ia)]["prompt_tokens"]
    if a["prompt_setting"] == "P2":
        exact -= est_text("\n\n" + HINT)
    scale = exact / total_est
    start_tok = [CONST]
    for t in msg_tok[:-1]:
        start_tok.append(start_tok[-1] + t)
    for k in range(min(len(pa), len(pb))):
        x, y = pa[k]["content"], pb[k]["content"]
        if x == y:
            continue
        role = ["system", "request"][k] if k < 2 else ("lookup_call" if pa[k]["role"] == "assistant" else "lookup_result")
        where.append(role)
        sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            diff_tok += est_text(x[i1:i2]) if i2 > i1 else 0.0
            off = (total_est - (start_tok[k] + est_text(x[:i1]))) * scale
            first_off = off if first_off is None else max(first_off, off)
    rec = {"pair_id": pid, "level": lv, "A_i": ia, "B_i": ib, "where": sorted(set(where)), "diff_tokens_in_A": round(diff_tok),
           "earliest_diff_offset_from_end": round(first_off) if first_off is not None else None, "A_total_tokens": round(exact)}
    res[lv]["recs"].append(rec)
summary = {}
for lv, d in sorted(res.items()):
    recs = d["recs"]
    wc = Counter("+".join(r["where"]) for r in recs)
    offs = [r["earliest_diff_offset_from_end"] for r in recs if r["earliest_diff_offset_from_end"] is not None]
    summary[lv] = {"pairs_with_both_twins": len(recs), "where_A_and_B_differ": dict(wc.most_common()),
                   "share_differ_only_in_lookup_result": round(sum(r["where"] == ["lookup_result"] for r in recs) / len(recs), 3),
                   "share_system_prompt_differs": round(sum("system" in r["where"] for r in recs) / len(recs), 3),
                   "diff_tokens_p50": int(np.median([r["diff_tokens_in_A"] for r in recs])),
                   "diff_share_of_prompt_p50": round(float(np.median([r["diff_tokens_in_A"] / r["A_total_tokens"] for r in recs])), 4),
                   "earliest_diff_offset_p10_p50_p90": [round(float(np.quantile(offs, q))) for q in (0.1, 0.5, 0.9)] if offs else None,
                   "pairs_whose_earliest_diff_is_beyond_4096": sum(o > 4096 for o in offs),
                   "pairs_identical_after_hint_strip": sum(not r["where"] for r in recs)}
    examples += [r for r in recs if r["earliest_diff_offset_from_end"] and r["earliest_diff_offset_from_end"] > 4096][:3]
out["a_twin_diff_by_level"] = summary
out["a_examples_diff_beyond_window"] = examples
out["a_examples_identical"] = [r for lv in res for r in res[lv]["recs"] if not r["where"]][:5]
out["a_examples_lookup_result_only_L2"] = [r for r in res["L2"]["recs"] if r["where"] == ["lookup_result"]][:5]

# ---------------------------------------------------------------- (b) hint vs targets
tk_cache = {}
hint_cmp = Counter()
hint_ex = defaultdict(list)
for v in ("sft_train", "sft_val", "dpo_train"):
    for i, r in enumerate(V[v]):
        if r.get("turn", "decision") != "decision":
            continue
        s = by_id[r["scenario_id"]]
        t = (r.get("completion") or r.get("chosen"))[0]["content"]
        act = parse_action(t, {"request_approval", "send_payment"} | {x["tool"] for x in s.get("tool_rules", [])} |
                           {g.get("tool", "") for g in (s.get("gold_commits") or [])})
        tw = r["twin"][0] if not r["twin"].startswith("A_pushback") else "P"
        if act.kind == "message":
            cls = classify_decision("message", s, text=act.text)
        else:
            cls = "card" if act.name == "request_approval" else "commit_or_other_tool"
        key = f"{v}|{r['prompt_setting']}|{tw}|{cls}"
        hint_cmp[key] += 1
        contradicts = r["prompt_setting"] == "P2" and ((tw == "B" and cls != "card") or (tw in "AP" and cls not in ("ask_informed",)))
        if contradicts and len(hint_ex[v]) < 6:
            hint_ex[v].append({"i": i, "scenario_id": r["scenario_id"], "level": r["level"], "class": cls, "target": t[:180],
                               "expected": s["expected"].get("decision"), "accept": adjudicate(s)["expected"].get("accept")})
out["b_target_class_by_prompt_setting"] = dict(sorted(hint_cmp.items()))
out["b_p2_rows_where_target_contradicts_hint"] = hint_ex
# word counts of twin-A message targets vs "one short question"
wc = [len(by_id[r["scenario_id"]]["ideal_action"]["text"].split()) for r in V["sft_train"]
      if r["turn"] == "decision" and r["twin"].startswith("A") and by_id[r["scenario_id"]]["ideal_action"]["type"] == "message"]
qm = [by_id[r["scenario_id"]]["ideal_action"]["text"].count("?") for r in V["sft_train"]
      if r["turn"] == "decision" and r["twin"].startswith("A") and by_id[r["scenario_id"]]["ideal_action"]["type"] == "message"]
out["b_twinA_message_targets_words_p10_p50_p90"] = [int(np.quantile(wc, q)) for q in (0.1, 0.5, 0.9)]
out["b_twinA_message_targets_question_marks"] = dict(Counter(qm))
# twin-B expected decisions in training vs test
exp_b = Counter()
for name, ids in (("train_trained", {r["scenario_id"] for r in V["sft_train"]}),):
    for sid in ids:
        s = by_id[sid]
        if s["twin"].startswith("B"):
            a = s["ideal_action"]
            exp_b[f"{name}|expected={s['expected']['decision']}|target={'card' if a.get('name') == 'request_approval' else a.get('type') + ':' + str(a.get('name'))}"] += 1
out["b_twinB_expected_and_target_kind"] = dict(exp_b)

# ---------------------------------------------------------------- (c) M1 samples by prompt setting
ps_of = {(r["scenario_id"], r["level"]): r["prompt_setting"] for r in V["grpo_train"]}
agg = defaultdict(lambda: [0, 0])
dist = defaultdict(Counter)
for sc in jl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")):
    m = sc["meta"]
    s = by_id[m["scenario_id"]]
    d = sc["components"]["decision"]
    ok = d not in ("lookup", "invalid", "truncated") and bool(decision_grade(adjudicate(s), d)["success"])
    ps = ps_of.get((m["scenario_id"], m["level"]), "?")
    tw = m["twin"][0]
    for key in (f"{ps}|{tw}", f"{ps}|{tw}|{m['level']}"):
        agg[key][0] += ok
        agg[key][1] += 1
    if tw == "B":
        dist[f"B|expected={s['expected']['decision']}|{ps}"][d] += 1
    else:
        dist[f"A|{ps}"][d] += 1
out["c_m1_samples_by_prompt_setting"] = {k: {"correct": a, "n": n, "rate": round(a / n, 3)} for k, (a, n) in sorted(agg.items())}
out["c_m1_decision_distribution"] = {k: dict(v.most_common()) for k, v in sorted(dist.items())}
print(json.dumps(out["a_twin_diff_by_level"], indent=1))
print(json.dumps({k: v for k, v in out.items() if k.startswith(("b_target", "b_twin", "c_"))}, indent=1)[:6000])
print(dump("s06_twin_diff_hint.json", out))
