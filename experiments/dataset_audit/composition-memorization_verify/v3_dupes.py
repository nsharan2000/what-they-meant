"""CM-1 selector, CM-2 duplicate renderings, twin contrast, request sharing vs M1 accuracy (independent re-derivation).

Inputs: dataset/curated_v121/{sft_train,sft_val,grpo_train}.jsonl, token_stats.jsonl, prompts/golden_short.md,
results/v2_grpo_prompts.csv (from v2_grpo.py).
Output: results/v3_dupes.json, results/v3_twin_pairs.csv
"""
import difflib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vcommon import ROOT, RES, cur, dump, load_jsonl, target, request, is_call, call_name

HINT = open(os.path.join(ROOT, "prompts/golden_short.md")).read().strip()
ts = load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))
out = {}


def strip_hint(sysmsg):
    return sysmsg.replace("\n\n" + HINT, "").replace(HINT, "")


for fname in ("sft_train", "sft_val"):
    rows = cur(fname)
    st = [t for t in ts if t["file"] == fname + ".jsonl"]
    tg = [target(r) for r in rows]
    dec = [i for i, r in enumerate(rows) if r["turn"] == "decision"]
    look = [i for i, r in enumerate(rows) if r["turn"] != "decision"]
    o = {}
    o["distinct_targets"] = len(set(tg))
    o["distinct_decision_targets"] = len({tg[i] for i in dec})
    o["distinct_lookup_targets"] = len({tg[i] for i in look})
    o["targets_both_lookup_and_decision"] = len({tg[i] for i in dec} & {tg[i] for i in look})
    cnt_dec = Counter(tg[i] for i in dec)
    cnt_look = Counter(tg[i] for i in look)
    o["decision_target_repeats"] = {"mean": round(np.mean(list(cnt_dec.values())), 3), "median": float(np.median(list(cnt_dec.values()))),
                                    "max": max(cnt_dec.values())}
    o["lookup_target_repeats"] = {"mean": round(np.mean(list(cnt_look.values())), 3), "median": float(np.median(list(cnt_look.values()))),
                                  "max": max(cnt_look.values())}
    top = cnt_look.most_common(1)[0]
    o["top_lookup_target"] = {"target": top[0], "rows": top[1],
                              "scenarios": len({rows[i]["scenario_id"] for i in look if tg[i] == top[0]}),
                              "pairs": len({rows[i]["pair_id"] for i in look if tg[i] == top[0]}),
                              "first_lines": [i for i in look if tg[i] == top[0]][:5]}
    o["rows_repeating_a_target_within_epoch"] = len(rows) - len(set(tg))
    o["exposures_2ep_per_decision_target"] = round(2 * len(dec) / len(cnt_dec), 2)
    o["exposures_2ep_per_lookup_target"] = round(2 * len(look) / len(cnt_look), 2)
    # decision targets shared across twins / scenarios
    by_t = defaultdict(set)
    for i in dec:
        by_t[tg[i]].add(rows[i]["scenario_id"])
    o["decision_targets_shared_by_2plus_scenarios"] = sum(len(v) > 1 for v in by_t.values())
    o["decision_targets_shared_across_twins"] = sum(len({s[-1] for s in v}) > 1 for v in by_t.values())
    # one target per scenario
    per_scn = defaultdict(set)
    for i in dec:
        per_scn[rows[i]["scenario_id"]].add(tg[i])
    o["scenarios"] = len(per_scn)
    o["scenarios_with_one_decision_target"] = sum(len(v) == 1 for v in per_scn.values())
    # distinct conversations (messages after system) among decision rows, and with the L3 request normalised away
    conv = Counter(json.dumps(rows[i]["prompt"][1:], ensure_ascii=False) for i in dec)
    o["decision_rows"] = len(dec)
    o["distinct_decision_conversations_after_system"] = len(conv)
    # do renderings of one scenario differ only in the system prompt? (compare prompt[2:] = lookup call + result across levels)
    tail_by_scn = defaultdict(set)
    req_by_scn = defaultdict(set)
    for i in dec:
        tail_by_scn[rows[i]["scenario_id"]].add(json.dumps(rows[i]["prompt"][2:], ensure_ascii=False))
        req_by_scn[rows[i]["scenario_id"]].add(request(rows[i]))
    o["scenarios_with_identical_lookup_and_result_across_levels"] = sum(len(v) == 1 for v in tail_by_scn.values())
    o["scenarios_with_1_request"] = sum(len(v) == 1 for v in req_by_scn.values())
    o["scenarios_with_2_requests"] = sum(len(v) == 2 for v in req_by_scn.values())
    lv_by_scn = defaultdict(list)
    for i in dec:
        lv_by_scn[rows[i]["scenario_id"]].append(rows[i]["level"])
    o["decision_renderings_per_scenario"] = dict(Counter(len(v) for v in lv_by_scn.values()))
    # ---- selector: request -> targets; request + first token -> targets ----------------------------------------------
    g_req = defaultdict(set)
    g_req_tok = defaultdict(set)
    g_req_unit = defaultdict(set)
    for i in dec:
        rq = request(rows[i])
        ft = st[i]["completion_first5"][0]
        unit = ("TOOL:" + call_name(tg[i])) if is_call(tg[i]) else ("MSG:" + tg[i].split()[0])
        g_req[rq].add(tg[i])
        g_req_tok[(rq, ft)].add(tg[i])
        g_req_unit[(rq, unit)].add(tg[i])
    amb_req = [i for i in dec if len(g_req[request(rows[i])]) > 1]
    amb_tok = [i for i in dec if len(g_req_tok[(request(rows[i]), st[i]["completion_first5"][0])]) > 1]
    amb_unit = [i for i in dec if len(g_req_unit[(request(rows[i]), ("TOOL:" + call_name(tg[i])) if is_call(tg[i]) else ("MSG:" + tg[i].split()[0]))]) > 1]
    o["selector"] = {"rows_whose_request_has_2plus_targets": len(amb_req), "share": round(len(amb_req) / len(dec), 4),
                     "rows_ambiguous_after_request_plus_first_bpe_token": len(amb_tok),
                     "share_tok": round(len(amb_tok) / len(dec), 4),
                     "rows_ambiguous_after_request_plus_first_word_or_toolname": len(amb_unit),
                     "share_unit": round(len(amb_unit) / len(dec), 4),
                     "ambiguous_tok_scenarios": sorted({rows[i]["scenario_id"] for i in amb_tok})}
    # Is the request shared with the TWIN (same pair, other twin)?
    req_twins = defaultdict(set)
    for i in dec:
        req_twins[(rows[i]["pair_id"], request(rows[i]))].add(rows[i]["twin"][0])
    shared = [i for i in dec if len(req_twins[(rows[i]["pair_id"], request(rows[i]))]) == 2]
    o["rows_whose_request_is_shared_with_twin"] = len(shared)
    # ---- twin contrast at the same level ------------------------------------------------------------------------------
    by_pl = defaultdict(dict)
    for i in dec:
        r = rows[i]
        by_pl[(r["pair_id"], r["level"])][r["twin"][0]] = i
    recs = []
    for (pid, lv), d in sorted(by_pl.items()):
        if "A" not in d or "B" not in d:
            continue
        a, b = rows[d["A"]], rows[d["B"]]
        pa, pb = a["prompt"], b["prompt"]
        rec = {"pair": pid, "level": lv, "lineA": d["A"], "lineB": d["B"], "psA": a["prompt_setting"], "psB": b["prompt_setting"],
               "n_msgs_A": len(pa), "n_msgs_B": len(pb)}
        rec["sys_equal_raw"] = pa[0]["content"] == pb[0]["content"]
        rec["sys_equal_hint_stripped"] = strip_hint(pa[0]["content"]) == strip_hint(pb[0]["content"])
        rec["request_equal"] = request(a) == request(b)
        if len(pa) == len(pb) and len(pa) >= 4:
            rec["calls_equal"] = all(pa[k]["content"] == pb[k]["content"] for k in range(2, len(pa), 2))
            rec["results_equal"] = all(pa[k]["content"] == pb[k]["content"] for k in range(3, len(pa), 2))
            ra = "".join(pa[k]["content"] for k in range(3, len(pa), 2))
            rb = "".join(pb[k]["content"] for k in range(3, len(pb), 2))
            sm = difflib.SequenceMatcher(None, ra, rb, autojunk=False)
            ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
            rec["result_diff_runs"] = len(ops)
            rec["result_diff_chars"] = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
            rec["result_chars"] = max(len(ra), len(rb))
        else:
            rec["calls_equal"] = rec["results_equal"] = None
        tot = sum(len(m["content"]) for m in pa)
        rec["prompt_chars_A"] = tot
        rec["only_result_differs_hint_stripped"] = bool(rec["sys_equal_hint_stripped"] and rec["request_equal"] and rec.get("calls_equal")
                                                        and rec.get("results_equal") is False)
        rec["only_result_differs_raw"] = bool(rec["sys_equal_raw"] and rec["request_equal"] and rec.get("calls_equal")
                                              and rec.get("results_equal") is False)
        rec["identical_prompts"] = pa == pb
        recs.append(rec)
    tp = pd.DataFrame(recs)
    tp.to_csv(os.path.join(RES, f"v3_twin_pairs_{fname}.csv"), index=False)
    o["twin_pair_levels"] = len(tp)
    o["twin_pair_levels_only_result_differs_hint_stripped"] = int(tp.only_result_differs_hint_stripped.sum())
    o["share_only_result_hint_stripped"] = round(tp.only_result_differs_hint_stripped.mean(), 4)
    o["twin_pair_levels_only_result_differs_raw"] = int(tp.only_result_differs_raw.sum())
    o["twin_pair_levels_ps_differs"] = int((tp.psA != tp.psB).sum())
    o["twin_pair_levels_sys_differs_beyond_hint"] = int((~tp.sys_equal_hint_stripped).sum())
    o["twin_pair_levels_request_differs"] = int((~tp.request_equal).sum())
    o["twin_pair_levels_identical"] = int(tp.identical_prompts.sum())
    o["twin_pair_levels_msgcount_differs"] = int((tp.n_msgs_A != tp.n_msgs_B).sum())
    sel = tp[tp.only_result_differs_hint_stripped]
    if len(sel):
        o["only_result_pairs_result_diff_chars_median"] = float(sel.result_diff_chars.median())
        o["only_result_pairs_result_diff_runs_median"] = float(sel.result_diff_runs.median())
        o["only_result_pairs_diff_share_of_prompt_chars_median"] = round(float((sel.result_diff_chars / sel.prompt_chars_A).median()), 4)
        o["only_result_pairs_result_chars_median"] = float(sel.result_chars.median())
    out[fname] = o

# ---- message count of decision prompts by twin (a composition shortcut?) --------------------------------------------------
rows = cur("sft_train")
dec = [r for r in rows if r["turn"] == "decision"]
out["decision_prompt_msgcount_by_twin"] = {t: dict(Counter(len(r["prompt"]) for r in dec if r["twin"][0] == t)) for t in "AB"}
out["prefix_calls_by_twin_scenarios"] = {}
grpo = cur("grpo_train")
pc = defaultdict(set)
for g in grpo:
    s = json.loads(g["scenario"])
    pc[s["twin"][0]].add((s["id"], len(s.get("prefix_calls") or [])))
out["prefix_calls_by_twin_scenarios"] = {t: dict(Counter(n for _, n in v)) for t, v in pc.items()}
# pairs whose twins differ in look-up count
nlook = {}
for g in grpo:
    s = json.loads(g["scenario"])
    nlook[s["id"]] = len(s.get("prefix_calls") or [])
diffpairs = sorted({sid[:-1] for sid in nlook if sid.endswith("A") and sid[:-1] + "B" in nlook and nlook[sid] != nlook[sid[:-1] + "B"]})
out["pairs_where_twins_differ_in_lookup_count"] = diffpairs

# ---- M1 accuracy (GRPO samples) by whether the decision row's request is shared with its twin -------------------------------
pp = pd.read_csv(os.path.join(RES, "v2_grpo_prompts.csv"))
req_twins = defaultdict(set)
for r in dec:
    req_twins[(r["pair_id"], request(r))].add(r["twin"][0])
gl = {i: g for i, g in enumerate(grpo)}
pp["req_shared_with_twin"] = pp.grpo_line.map(lambda i: len(req_twins[(gl[i]["pair_id"], request(gl[i]))]) == 2)
out["m1_acc_by_request_shared_with_twin"] = {str(k): [round(float(x.acc.mean()), 3), int(len(x))] for k, x in pp.groupby("req_shared_with_twin")}
out["m1_acc_by_request_shared_with_twin_and_twin"] = {f"{k[0]}/{k[1]}": [round(float(x.acc.mean()), 3), int(len(x))]
                                                      for k, x in pp.groupby(["req_shared_with_twin", "twin"])}
# examples of request-unique pairs (what differs?)
ex = []
seen = set()
for r in dec:
    if len(req_twins[(r["pair_id"], request(r))]) == 1 and r["pair_id"] not in seen and r["level"] == "L1":
        other = [x for x in dec if x["pair_id"] == r["pair_id"] and x["twin"][0] != r["twin"][0] and x["level"] == "L1"]
        if other:
            seen.add(r["pair_id"])
            ex.append({"pair": r["pair_id"], "A" if r["twin"][0] == "A" else "B": request(r),
                       "other": request(other[0])})
out["request_unique_examples_L1"] = ex[:12]
out["n_pairs_request_unique_L1"] = len(seen)
dump(out, "v3_dupes.json")
print(json.dumps(out, indent=1, ensure_ascii=False)[:9000])
