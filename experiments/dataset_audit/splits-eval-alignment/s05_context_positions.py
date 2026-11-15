"""5. Context length and WHERE things sit relative to the generation point (sliding_window 4096 on 3 of 4 layers).

Part 1  rows whose prompt + target exceed 4,096 / 8,192 tokens (exact counts, experiments/dataset_audit/results/token_stats.jsonl).
Part 2  token offsets FROM THE END of the prompt (= distance from the first generated token) of: the template preamble, the system
        rules (environment policy), the member block (and its transaction history), tool docs, protocol, hint, request, look-up
        calls/results, and the decisive evidence (scenario_records lines; identifier-like strings from decisive_facts, gold/harmful
        commit signatures, informed_any). Positions are estimated with tokmodel.py (CV error 0.65 % of prompt length) and rescaled
        per row so the whole prompt equals the exact token count. Rows: every DECISION row of sft_train (and the same for eval
        renderings of the test split at L0r/L1/L2/L2s, P0, with the canonical look-ups).
Part 3  behaviour vs position: the M1 policy's samples on training decision prompts (m2_grpo/scores.jsonl, step 0-29) and the base
        model's held-out eval (eval_base_v12.jsonl) by level; within L2, accuracy when the decisive evidence is beyond the window.

python3 -B experiments/dataset_audit/splits-eval-alignment/s05_context_positions.py -> results/s05_context_positions.json
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, dump, eval_files, jl, scen_files, views
from tokmodel import est_text, weights, FEATS
from wtm.episode import decision_grade, run_episode
from wtm.labels import adjudicate
from wtm.render import _txn_line, golden_text, render
from wtm.spec import load_env, load_member

W = 4096
V = views()
stats = jl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
out = {}

# ------------------------------------------------------------------------------------------------ Part 1: over-length rows
over = defaultdict(Counter)
over_ex = defaultdict(list)
for s in stats:
    f = s["file"].replace(".jsonl", "")
    p = s["prompt_tokens"]
    totals = []
    if "completion_tokens" in s:
        totals.append(("prompt+completion", p + s["completion_tokens"]))
    if "chosen_tokens" in s:
        totals += [("prompt+chosen", p + s["chosen_tokens"]), ("prompt+rejected", p + s["rejected_tokens"])]
    if f == "grpo_train":
        totals.append(("prompt+256(max_completion)", p + 256))
    for name, t in totals:
        key = f"{f}|{name}"
        over[key]["n"] += 1
        over[key][f"gt4096|{s['level']}"] += t > 4096
        over[key][f"n|{s['level']}"] += 1
        over[key]["gt4096"] += t > 4096
        over[key]["gt8192"] += t > 8192
        if t > 8192 and len(over_ex[key]) < 5:
            over_ex[key].append({"file": s["file"], "i": s["i"], "scenario_id": s["scenario_id"], "level": s["level"],
                                 "turn": s.get("turn"), "tokens": t})
    over["prompt_only|" + f]["n"] += 1
    over["prompt_only|" + f]["gt4096"] += p > 4096
    over["prompt_only|" + f][f"gt4096|{s['level']}"] += p > 4096
    over["prompt_only|" + f][f"n|{s['level']}"] += 1
out["part1_over_length"] = {k: dict(v) for k, v in over.items()}
out["part1_examples_gt8192"] = over_ex
max_rows = sorted(stats, key=lambda s: -(s["prompt_tokens"] + max(s.get("completion_tokens", 0), s.get("rejected_tokens", 0),
                                                                     s.get("chosen_tokens", 0))))[:5]
out["part1_longest_rows"] = [{"file": s["file"], "i": s["i"], "scenario_id": s["scenario_id"], "level": s["level"],
                              "prompt_tokens": s["prompt_tokens"], "target_tokens": max(s.get("completion_tokens", 0),
                              s.get("rejected_tokens", 0), s.get("chosen_tokens", 0))} for s in max_rows]

# ------------------------------------------------------------------------------------------------ Part 2: positions
w = weights()
PER_MSG, CONST = w[len(FEATS)], w[len(FEATS) + 1]
HINT = golden_text("P2")
IDENT = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w.@])@\w[\w.-]*\w|\b[A-Z]{2,5}-?\d{3,}[A-Z0-9-]*\b|\b\d{3}-\d{4}\b|"
                   r"\b\d{4}-\d{2}-\d{2}\b|\b[\w.-]+@[a-z]+\b")


def evidence_strings(scn):
    """identifier-like strings that carry the decision: decisive facts, commit signatures, informed_any, scenario records."""
    pool = list(scn.get("decisive_facts") or []) + list(scn.get("decisive_strings") or [])
    for key in ("gold_commits", "harmful_commits"):
        for c in scn.get(key) or []:
            pool += c.get("match", [])
    pool += scn["expected"].get("informed_any", [])
    found = set()
    for t in pool:
        for m in IDENT.finditer(t):
            if len(m.group(0)) >= 5:
                found.add(m.group(0))
    return sorted(found)


def record_lines(scn, level):
    """rendered text of each scenario record (+ l2_extra at buried levels) as it appears in the member block."""
    recs = list(scn.get("scenario_records") or [])
    lines = []
    for x in recs:
        rec = x["record"]
        if x["field"] == "transactions":
            lines.append(_txn_line(rec))
        elif isinstance(rec, str):
            lines.append(rec)
        else:
            lines.append(json.dumps(rec, ensure_ascii=False))
    return lines


SECTION_ORDER = ["policy", "member", "member.history", "tools", "protocol", "most_relevant", "hint"]


def system_sections(sysm):
    """char spans of the system message's sections."""
    spans = {}
    i_mem = sysm.find("## Member\n")
    i_tools = sysm.find("## Tools\n")
    i_proto = sysm.find("## How to act\n")
    i_rel = sysm.find("## Most relevant records")
    i_hint = sysm.rfind(HINT) if HINT in sysm else -1
    end = len(sysm)
    if i_mem >= 0:
        spans["policy"] = (0, i_mem)
        spans["member"] = (i_mem, i_tools)
        m = re.search(r"### (Recent activity|Transaction history)[^\n]*\n", sysm[i_mem:i_tools])
        if m:
            h0 = i_mem + m.start()
            nxt = re.search(r"\n### ", sysm[h0 + 4:i_tools])
            spans["member.history"] = (h0, h0 + 4 + nxt.start() if nxt else i_tools)
    else:   # L0 template: one-line intro, the seed member card, tools, protocol
        spans["policy"] = (0, max(0, sysm.find("\n\n")))
        spans["member"] = (max(0, sysm.find("\n\n")), i_tools)
    p_end = min(x for x in (i_rel, i_hint, end) if x >= 0 and x > i_proto)
    spans["tools"] = (i_tools, i_proto)
    spans["protocol"] = (i_proto, p_end)
    if i_rel >= 0:
        spans["most_relevant"] = (i_rel, i_hint if i_hint > i_rel else end)
    if i_hint >= 0:
        spans["hint"] = (i_hint, i_hint + len(HINT))
    return spans


def positions(prompt, exact_tokens, scn, level):
    """token offset from the end for every section start/end and every evidence occurrence (estimated, rescaled)."""
    # token estimate per message incl. its template wrapper; the template preamble + BOS = CONST at the very start
    sysm = prompt[0]["content"]
    msg_tok = [est_text(m["content"]) + PER_MSG for m in prompt]
    total_est = CONST + sum(msg_tok)
    scale = exact_tokens / total_est
    # cumulative tokens from the start to the start of message k
    starts = [CONST]
    for t in msg_tok[:-1]:
        starts.append(starts[-1] + t)

    def off_sys(ch):         # offset from the end of a char position inside the system message
        before = CONST + est_text(sysm[:ch])
        return (total_est - before) * scale

    res = {"total": exact_tokens, "preamble_start": total_est * scale, "preamble_end": (total_est - CONST) * scale}
    for name, (a, b) in system_sections(sysm).items():
        res[name + "_start"], res[name + "_end"] = off_sys(a), off_sys(b)
    res["request_start"] = (total_est - starts[1]) * scale
    if len(prompt) > 2:
        res["lookups_start"] = (total_est - starts[2]) * scale
    ev = []
    low_sys = sysm.lower()
    rest = [m["content"].lower() for m in prompt[1:]]
    for sstr in evidence_strings(scn):
        k = sstr.lower()
        in_rest = any(k in t for t in rest)
        occ = [mm.start() for mm in re.finditer(re.escape(k), low_sys)]
        ev.append({"s": sstr, "in_request_or_results": in_rest, "sys_offsets": [round(off_sys(o)) for o in occ]})
    recs = []
    for line in record_lines(scn, level):
        o = sysm.find(line)
        recs.append(round(off_sys(o)) if o >= 0 else None)
    res["evidence"] = ev
    res["record_offsets"] = recs
    return res


tok_of = {(s["file"], s["i"]): s["prompt_tokens"] for s in stats}
pos_rows = []
for i, r in enumerate(V["sft_train"]):
    if r["turn"] != "decision":
        continue
    scn = by_id[r["scenario_id"]]
    p = positions(r["prompt"], tok_of[("sft_train.jsonl", i)], scn, r["level"])
    p.update({"i": i, "scenario_id": r["scenario_id"], "level": r["level"], "prompt_setting": r["prompt_setting"],
              "twin": r["twin"][0]})
    pos_rows.append(p)


def q(v, qs=(0.1, 0.5, 0.9)):
    v = [x for x in v if x is not None]
    return [round(float(np.quantile(v, x))) for x in qs] if v else None


def summarise(rows):
    by = defaultdict(list)
    for p in rows:
        by[p["level"]].append(p)
    summ = {}
    for lv, ps in sorted(by.items()):
        d = {"n": len(ps), "total_tokens_p10_p50_p90": q([p["total"] for p in ps])}
        for key in ["preamble_start", "policy_start", "policy_end", "member_start", "member.history_start", "member.history_end",
                    "member_end", "tools_start", "tools_end", "protocol_start", "most_relevant_start", "hint_start",
                    "request_start", "lookups_start"]:
            vals = [p[key] for p in ps if key in p]
            if vals:
                d[key + "_p10_p50_p90"] = q(vals)
        d["share_policy_entirely_beyond_window"] = round(sum(p["policy_end"] > W for p in ps) / len(ps), 3)
        d["share_policy_partly_beyond_window"] = round(sum(p["policy_start"] > W for p in ps) / len(ps), 3)
        hs = [p for p in ps if "member.history_start" in p]
        if hs:
            d["share_history_start_beyond_window"] = round(sum(p["member.history_start"] > W for p in hs) / len(hs), 3)
            d["share_history_entirely_beyond_window"] = round(sum(p["member.history_end"] > W for p in hs) / len(hs), 3)
        recs = [o for p in ps for o in p["record_offsets"] if o is not None]
        d["scenario_record_lines"] = len(recs)
        d["scenario_record_offsets_p10_p50_p90"] = q(recs)
        d["share_scenario_record_lines_beyond_window"] = round(sum(o > W for o in recs) / len(recs), 3) if recs else None
        # evidence that sits ONLY in the system prompt (not in the request or look-up results)
        so_rows, so_beyond = 0, 0
        for p in ps:
            only_sys = [e for e in p["evidence"] if not e["in_request_or_results"] and e["sys_offsets"]]
            if only_sys:
                so_rows += 1
                nearest = min(min(e["sys_offsets"]) for e in only_sys)
                so_beyond += nearest > W
        d["rows_with_system_only_evidence"] = so_rows
        d["rows_whose_system_only_evidence_is_all_beyond_window"] = so_beyond
        summ[lv] = d
    return summ


out["part2_sft_train_decision_rows_by_level"] = summarise(pos_rows)
out["part2_note"] = ("offsets are tokens from the end of the prompt (incl. the generation prompt); > 4096 means outside the "
                     "sliding window of 3 of every 4 layers at the first generated token; template preamble = fitted constant "
                     f"{CONST:.0f} tokens before the system message")
# examples: L2 rows whose decisive record lines sit beyond the window
ex = [p for p in pos_rows if p["level"] == "L2" and any(o is not None and o > W for o in p["record_offsets"])]
out["part2_examples_L2_records_beyond_window"] = [{"sft_train_i": p["i"], "scenario_id": p["scenario_id"], "level": p["level"],
                                                   "total_tokens": p["total"], "record_offsets": p["record_offsets"],
                                                   "policy_end": round(p["policy_end"]), "tools_start": round(p["tools_start"])}
                                                  for p in ex[:6]]
out["part2_n_L2_rows_with_a_record_beyond_window"] = len(ex)

# eval renderings (test split, canonical look-ups through the eval loop) at L0r/L1/L2/L2s, P0 — same estimator, no exact count:
# the estimate itself is used (CV error 0.65 %).


class Res:
    def __init__(self, text):
        self.text, self.is_error, self.usage, self.cost_usd, self.raw_error = text, False, {}, 0.0, ""


class Oracle:
    def __init__(self, scn):
        self.s = scn

    def chat(self, system, msgs):
        k = sum(m["role"] == "assistant" for m in msgs)
        pcs = self.s.get("prefix_calls") or []
        if k < len(pcs):
            return Res('<tool_call>' + json.dumps({"name": pcs[k]["tool"], "arguments": pcs[k]["arguments"]}, ensure_ascii=False)
                       + '</tool_call>')
        a = self.s["ideal_action"]
        return Res(a["text"] if a["type"] == "message" else '<tool_call>' + json.dumps(
            {"name": a["name"], "arguments": a.get("arguments", {})}, ensure_ascii=False) + '</tool_call>')


envs, members = {}, {}
eval_pos = []
for name, rows in eval_files().items():
    if not name.startswith(("test_", "hard_test_")):
        continue
    for s in rows:
        env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
        mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
        for lv in ("L0r", "L1", "L2", "L2s"):
            system, user, tools = render(env, mem, s, lv, "P0")
            res = run_episode(env, mem, s, Oracle(s), system, user, tools, mode="decision", max_steps=14, level=lv)
            prompt = [{"role": "system", "content": system}] + res["messages"][:-1]
            est = CONST + sum(est_text(m["content"]) + PER_MSG for m in prompt)
            p = positions(prompt, est, s, lv)
            p.update({"file": name, "scenario_id": s["id"], "level": lv, "twin": s["twin"][0]})
            eval_pos.append(p)
out["part2_eval_test_and_hard_test_P0_by_level"] = summarise(eval_pos)

# ------------------------------------------------------------------------------------------------ Part 3: behaviour vs position
scores = jl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl"))
grpo = V["grpo_train"]
grpo_pos = {}
for i, r in enumerate(grpo):
    p = positions(r["prompt"], tok_of[("grpo_train.jsonl", i)], by_id[r["scenario_id"]], r["level"])
    grpo_pos[(r["scenario_id"], r["level"])] = p


def correct(scn, decision):
    if decision in ("lookup", "invalid", "truncated"):
        return False
    return bool(decision_grade(adjudicate(scn), decision)["success"])


agg = defaultdict(lambda: [0, 0])
agg_win = defaultdict(lambda: [0, 0])
per_prompt = defaultdict(lambda: [0, 0])
for sc in scores:
    m = sc["meta"]
    scn = by_id[m["scenario_id"]]
    ok = correct(scn, sc["components"]["decision"])
    tw = m["twin"][0]
    for key in (m["level"], f"{m['level']}|{tw}", f"all|{tw}"):
        agg[key][0] += ok
        agg[key][1] += 1
    per_prompt[(m["scenario_id"], m["level"])][0] += ok
    per_prompt[(m["scenario_id"], m["level"])][1] += 1
    p = grpo_pos.get((m["scenario_id"], m["level"]))
    if p is not None and m["level"] in ("L1", "L2", "L3"):
        only_sys = [e for e in p["evidence"] if not e["in_request_or_results"] and e["sys_offsets"]]
        if only_sys:
            beyond = min(min(e["sys_offsets"]) for e in only_sys) > W
            k = f"{m['level']}|{tw}|system_only_evidence_{'beyond' if beyond else 'within'}_window"
        else:
            k = f"{m['level']}|{tw}|evidence_in_request_or_results"
        agg_win[k][0] += ok
        agg_win[k][1] += 1
out["part3_m1_policy_samples_on_training_prompts_strict_success"] = {
    k: {"correct": a, "n": n, "rate": round(a / n, 3)} for k, (a, n) in sorted(agg.items())}
out["part3_m1_policy_by_evidence_position"] = {
    k: {"correct": a, "n": n, "rate": round(a / n, 3)} for k, (a, n) in sorted(agg_win.items())}
out["part3_distinct_prompts_sampled"] = len(per_prompt)
out["part3_prompts_never_correct"] = sorted([list(k) + [v[1]] for k, v in per_prompt.items() if v[0] == 0])[:40]
out["part3_n_prompts_never_correct"] = sum(v[0] == 0 for v in per_prompt.values())

base = jl(os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl"))
bagg = defaultdict(lambda: [0, 0])
calib = []
for r in base:
    g = r["decision_grade"]
    scn = by_id.get(r["id"])
    ok = bool(decision_grade(adjudicate(scn), g["decision"])["success"]) if scn else bool(g["success"])
    for key in (f"{r['level']}|{r['prompt']}", f"{r['level']}|{r['prompt']}|{r['twin'][0]}"):
        bagg[key][0] += ok
        bagg[key][1] += 1
    if r["tokens"]["calls"] == 1 and r["messages"]:
        sysm = None
    # independent check of the token estimator on eval prompts: first call = [system, first user message]
    if r["tokens"]["calls"] == 1:
        scn_ = by_id.get(r["id"])
        if scn_:
            env = envs.setdefault(scn_["env_id"], load_env(scn_["env_id"]))
            mem = members.setdefault((scn_["env_id"], scn_["member_id"]), load_member(scn_["env_id"], scn_["member_id"]))
            system, user, _ = render(env, mem, scn_, r["level"], r["prompt"])
            est = CONST + est_text(system) + est_text(user) + 2 * PER_MSG
            calib.append((r["tokens"]["prompt"], est, len(system) == r["system_chars"]))
out["part3_base_tiny_aya_eval_v12_strict_success_adjudicated"] = {
    k: {"correct": a, "n": n, "rate": round(a / n, 3)} for k, (a, n) in sorted(bagg.items())}
if calib:
    c = np.array([(a, b) for a, b, _ in calib], float)
    ape = np.abs(c[:, 1] - c[:, 0]) / c[:, 0]
    out["tokmodel_independent_check_on_eval_first_calls"] = {
        "n": len(calib), "system_chars_match_current_render": sum(x for _, _, x in calib),
        "mape": round(float(ape.mean()), 4), "p95_ape": round(float(np.quantile(ape, 0.95)), 4),
        "mean_signed_err_tokens": round(float((c[:, 1] - c[:, 0]).mean()), 1)}
print(json.dumps({k: v for k, v in out.items() if k.startswith(("part1_over", "part2_sft", "part2_eval", "part3_m1", "part3_base", "tokmodel"))},
                 indent=1)[:12000])
print(dump("s05_context_positions.json", out))
