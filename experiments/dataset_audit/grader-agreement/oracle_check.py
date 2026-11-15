"""Oracle check: do the training targets, the GRPO reward and the evaluation grader agree?

For every SFT target (train + val, decision AND look-up turns), every DPO chosen and rejected, and every GRPO row's ideal
completion(s), compute
  * the GRPO reward exactly as training/train_grpo.py calls it: wtm.reward.reward(text, scenario-as-trained, env tool_kinds)
  * the evaluation decision exactly as wtm/run.py produces it: render() + run_episode(mode="decision") with a replay driver
    (row's own look-up calls, then the text), graded on the raw spec (run.py) and on the adjudicated spec (analysis.py).
Also checks: GRPO tool_kinds == env tool kinds; spec drift (GRPO-embedded scenario vs current spec file); render parity
(eval-time system prompt + conversation before the target == the training prompt).

PYTHONDONTWRITEBYTECODE=1 python3 experiments/dataset_audit/grader-agreement/oracle_check.py
-> results/oracle_rows.jsonl (one line per scored text), results/oracle_summary.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from ga_common import (FILES, OUT, adjudicate, any_hit, decision_grade, dump_json, dump_jsonl, eval_replay, expected_kind,
                       kw_hit, load_rows, load_specs, norm, reward, tool_kinds_of, top_class_ok, trained_specs)

SPEC_KEYS = ("expected", "gold_commits", "harmful_commits", "ideal_action", "prefix_calls", "tool_rules", "user_message",
             "scenario_records", "l2_extra_records", "l0_tools", "l0_member_block", "user_message_noisy", "twin")


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def action_text(a):
    return a["text"] if a["type"] == "message" else call_text(a["name"], a.get("arguments", {}))


def informed_hits(scn, text):
    return [k for k in scn["expected"].get("informed_any", []) if kw_hit(text, k)]


def reasons_for(scn, text, rclass, eclass, rep, role):
    """Why a target does not get the top class / why reward and eval disagree."""
    why = []
    tw = expected_kind(scn)
    if rclass != eclass:
        why.append(f"reward class {rclass} != eval class {eclass} (eval saw the target as {rep['target_event_kind']}; "
                   f"target_is_decision={rep['target_is_decision']})")
    if tw == "A":
        if rclass == "tell_informed":
            why.append("names the discrepancy but has no '?' and no QUESTION_RE phrase -> tell_informed (reward 0.5, not 1.0)")
        elif rclass in ("ask_generic", "statement"):
            why.append(f"no informed_any keyword found (keywords: {scn['expected'].get('informed_any')})")
        elif rclass.startswith("approve"):
            why.append("approval card on a should-ask twin")
        elif rclass == "lookup":
            why.append("look-up call: reward -0.1, eval treats it as a free step (no decision)")
    elif tw == "B_act":
        if rclass.startswith("approve"):
            why.append("approval card where the spec expects a direct commit (B_ACT gives 0.5)")
        elif rclass in ("commit_other", "commit_harmful"):
            why.append(f"commit args do not match gold_commits {scn.get('gold_commits')}")
    else:
        if rclass.startswith("commit"):
            acc = adjudicate(scn)["expected"].get("accept")
            why.append("direct commit where the spec expects an approval card"
                       + (f" (adjudication accepts {acc}: eval-after-adjudication counts commit_gold as correct, reward gives -0.5)"
                          if acc and "act" in acc else ""))
        elif rclass.startswith(("ask", "tell", "statement")):
            why.append("chat message on a should-act twin")
    return why


def score_text(scn, row, text, prefix, role, kinds):
    r, parts = reward(text, scn, kinds)
    rep = eval_replay(scn, row["level"], row["prompt_setting"], prefix, text)
    eclass = rep["eval_decision"]
    adj = adjudicate(scn)
    g_adj = decision_grade(adj, eclass)
    out = {"scenario_id": row["scenario_id"], "twin": scn["twin"], "expect": expected_kind(scn),
           "accept_adjudicated": adj["expected"].get("accept"), "level": row["level"], "prompt_setting": row["prompt_setting"],
           "turn": row.get("turn", "decision"), "role": role, "text": text,
           "reward": r, "reward_class": parts["decision"], "reward_parts": parts,
           "eval_class": eclass, "eval_success_raw": rep["eval_grade"]["success"],
           "eval_lenient_raw": rep["eval_grade"]["success_lenient"], "eval_success_adjudicated": g_adj["success"],
           "eval_harm": rep["eval_grade"]["harm"], "target_event_kind": rep["target_event_kind"],
           "target_is_decision": rep["target_is_decision"], "prefix_tool_kinds": rep["prefix_tool_kinds"],
           "informed_hits": informed_hits(scn, text) if role != "lookup_target" else None,
           "n_informed_keywords": len(scn["expected"].get("informed_any", []))}
    out["top_class_reward"] = top_class_ok(scn, parts["decision"])
    out["top_class_eval"] = top_class_ok(scn, eclass)
    out["classes_agree"] = parts["decision"] == eclass
    out["reasons"] = reasons_for(scn, text, parts["decision"], eclass, rep, role)
    # render parity: eval conversation before the target vs the training prompt
    return out, rep


def parity(row, rep):
    p = row["prompt"]
    sys_ok = rep["eval_system"] == p[0]["content"]
    em = rep["eval_msgs_before_target"] or []
    tm = [{"role": m["role"], "content": m["content"]} for m in p[1:]]
    msgs_ok = [(m["role"], m["content"]) for m in em] == [(m["role"], m["content"]) for m in tm]
    diff = None
    if not msgs_ok:
        for i, (a, b) in enumerate(zip(em, tm)):
            if (a["role"], a["content"]) != (b["role"], b["content"]):
                diff = {"msg_index": i + 1, "eval": a["content"][:600], "train": b["content"][:600]}
                break
        if diff is None:
            diff = {"len_eval": len(em), "len_train": len(tm)}
    return sys_ok, msgs_ok, diff


def main():
    specs = load_specs()
    tspecs, tsrc = trained_specs(specs)
    rows_out, parity_rows, drift = [], [], {}
    # spec drift: GRPO-embedded scenario vs current spec file
    for sid, s in tspecs.items():
        cur = specs.get(sid)
        if cur is None:
            drift[sid] = ["missing from current spec files"]
            continue
        d = [k for k in SPEC_KEYS if json.dumps(s.get(k), sort_keys=True) != json.dumps(cur.get(k), sort_keys=True)]
        if d:
            drift[sid] = d
    kinds_mismatch = []
    for f in FILES:
        rows = load_rows(f)
        for i, row in enumerate(rows):
            scn = tspecs[row["scenario_id"]]
            kinds = tool_kinds_of(scn["env_id"])
            prefix = [m["content"] for m in row["prompt"] if m["role"] == "assistant"]
            base = {"file": f, "line": i}
            if f in ("sft_train", "sft_val"):
                role = "decision_target" if row["turn"] == "decision" else "lookup_target"
                todo = [(role, row["completion"][0]["content"])]
            elif f == "dpo_train":
                todo = [("chosen", row["chosen"][0]["content"]), ("rejected", row["rejected"][0]["content"])]
            else:
                emb = json.loads(row["scenario"])
                if json.loads(row["tool_kinds"]) != kinds:
                    kinds_mismatch.append([f, i, row["scenario_id"]])
                scn = emb
                todo = [("ideal_action", action_text(emb["ideal_action"]))]
                im = emb["expected"].get("ideal_message")
                if im and im != todo[0][1]:
                    todo.append(("ideal_message_as_chat", im))
            for role, text in todo:
                o, rep = score_text(scn, row, text, prefix, role, kinds)
                o.update(base)
                o["spec_source"] = tsrc.get(row["scenario_id"])
                rows_out.append(o)
                if role in ("decision_target", "lookup_target", "chosen", "ideal_action"):
                    s_ok, m_ok, diff = parity(row, rep)
                    parity_rows.append({"file": f, "line": i, "scenario_id": row["scenario_id"], "level": row["level"],
                                        "prompt_setting": row["prompt_setting"], "turn": row.get("turn", "decision"),
                                        "system_equal": s_ok, "messages_equal": m_ok, "first_diff": diff})
    dump_jsonl(rows_out, f"{OUT}/oracle_rows.jsonl")
    dump_jsonl(parity_rows, f"{OUT}/render_parity.jsonl")

    # ---------- summary ----------
    def dist(xs):
        return dict(sorted(Counter(xs).items(), key=lambda kv: -kv[1]))

    summ = {"n_scored_texts": len(rows_out), "tool_kinds_mismatch_grpo_vs_env": kinds_mismatch,
            "spec_drift_grpo_embedded_vs_current": drift, "spec_source_counts": dict(Counter(tsrc.values())),
            "by_file_role_expect": {}}
    groups = defaultdict(list)
    for o in rows_out:
        groups[(o["file"], o["role"], o["expect"])].append(o)
    for (f, role, ex), os_ in sorted(groups.items()):
        rs = [o["reward"] for o in os_]
        summ["by_file_role_expect"][f"{f}|{role}|{ex}"] = {
            "n": len(os_), "reward_class": dist(o["reward_class"] for o in os_), "eval_class": dist(o["eval_class"] for o in os_),
            "reward_values": dist(o["reward"] for o in os_), "reward_mean": round(sum(rs) / len(rs), 4),
            "reward_min": min(rs), "reward_max": max(rs),
            "top_class_reward": sum(o["top_class_reward"] for o in os_), "top_class_eval": sum(o["top_class_eval"] for o in os_),
            "eval_success_raw": sum(bool(o["eval_success_raw"]) for o in os_),
            "eval_success_adjudicated": sum(bool(o["eval_success_adjudicated"]) for o in os_),
            "classes_agree": sum(o["classes_agree"] for o in os_),
            "target_is_decision": sum(o["target_is_decision"] for o in os_)}
    # disagreements on positive targets (decision targets, DPO chosen, GRPO ideal_action)
    pos_roles = ("decision_target", "chosen", "ideal_action")
    dis = [o for o in rows_out if o["role"] in pos_roles and not (o["top_class_reward"] and o["top_class_eval"]
                                                                   and o["classes_agree"])]
    summ["positive_target_disagreements"] = {
        "n": len(dis), "n_positive_targets": sum(o["role"] in pos_roles for o in rows_out),
        "by_file": dict(Counter(o["file"] for o in dis)),
        "by_reward_class": dict(Counter((o["expect"], o["reward_class"]) for o in dis).most_common()),
        "distinct_scenarios": sorted({o["scenario_id"] for o in dis}),
        "rows": [{k: o[k] for k in ("file", "line", "scenario_id", "level", "prompt_setting", "role", "expect", "reward",
                                    "reward_class", "eval_class", "eval_success_raw", "eval_success_adjudicated",
                                    "accept_adjudicated", "reasons", "text")} for o in dis]}
    # reward class vs eval class mismatches anywhere
    mm = [o for o in rows_out if not o["classes_agree"]]
    summ["reward_vs_eval_class_mismatch"] = {
        "n": len(mm), "pairs": dict(Counter(f"{o['role']}: reward={o['reward_class']} eval={o['eval_class']} "
                                            f"target_event={o['target_event_kind']}" for o in mm).most_common()),
        "examples": [{k: o[k] for k in ("file", "line", "scenario_id", "level", "prompt_setting", "role", "reward_class",
                                         "eval_class", "target_event_kind", "text")} for o in mm[:12]]}
    # DPO margin check
    dpo = defaultdict(dict)
    for o in rows_out:
        if o["file"] == "dpo_train":
            dpo[o["line"]][o["role"]] = o
    viol, margins, rej_cls, rej_eval = [], [], Counter(), Counter()
    for line, d in sorted(dpo.items()):
        c, r = d["chosen"], d["rejected"]
        margins.append(round(c["reward"] - r["reward"], 4))
        rej_cls[(r["expect"], r["reward_class"])] += 1
        rej_eval[(r["expect"], r["eval_class"], bool(r["eval_success_adjudicated"]))] += 1
        if not c["reward"] > r["reward"]:
            viol.append({"line": line, "scenario_id": c["scenario_id"], "level": c["level"], "chosen": c["reward"],
                         "rejected": r["reward"], "chosen_class": c["reward_class"], "rejected_class": r["reward_class"]})
    summ["dpo"] = {"n_pairs": len(dpo), "violations_reward_chosen_le_rejected": viol,
                   "margin_dist": dist(margins), "rejected_reward_class": {f"{a}|{b}": n for (a, b), n in rej_cls.most_common()},
                   "rejected_eval_class_and_eval_success_adjudicated": {f"{a}|{b}|success={s}": n
                                                                       for (a, b, s), n in rej_eval.most_common()},
                   "rejected_that_eval_counts_as_success": [
                       {k: d["rejected"][k] for k in ("line", "scenario_id", "level", "prompt_setting", "expect", "reward",
                                                      "reward_class", "eval_class", "accept_adjudicated", "text")}
                       for line, d in sorted(dpo.items()) if d["rejected"]["eval_success_adjudicated"]],
                   "rejected_that_are_lookups_or_nondecisions": [
                       {k: d["rejected"][k] for k in ("line", "scenario_id", "level", "expect", "reward", "reward_class",
                                                      "eval_class", "target_event_kind", "text")}
                       for line, d in sorted(dpo.items())
                       if d["rejected"]["reward_class"] == "lookup" or not d["rejected"]["target_is_decision"]]}
    # look-up turns
    lk = [o for o in rows_out if o["role"] == "lookup_target"]
    summ["lookup_targets"] = {"n": len(lk), "reward_values": dist(o["reward"] for o in lk),
                              "reward_class": dist(o["reward_class"] for o in lk),
                              "eval_target_event_kind": dist(o["target_event_kind"] for o in lk)}
    summ["render_parity"] = {"n": len(parity_rows), "system_equal": sum(p["system_equal"] for p in parity_rows),
                             "messages_equal": sum(p["messages_equal"] for p in parity_rows),
                             "mismatch_examples": [p for p in parity_rows if not (p["system_equal"] and p["messages_equal"])][:8],
                             "mismatch_by_file": dict(Counter(p["file"] for p in parity_rows
                                                              if not (p["system_equal"] and p["messages_equal"])))}
    dump_json(summ, f"{OUT}/oracle_summary.json")
    # console digest
    print(json.dumps({k: v for k, v in summ["by_file_role_expect"].items()}, indent=1)[:6000])
    print("positive-target disagreements:", summ["positive_target_disagreements"]["n"], "/",
          summ["positive_target_disagreements"]["n_positive_targets"], summ["positive_target_disagreements"]["by_reward_class"])
    print("reward-vs-eval class mismatches:", summ["reward_vs_eval_class_mismatch"]["n"], summ["reward_vs_eval_class_mismatch"]["pairs"])
    print("DPO violations:", len(viol), "margins:", summ["dpo"]["margin_dist"])
    print("render parity:", summ["render_parity"]["n"], summ["render_parity"]["system_equal"], summ["render_parity"]["messages_equal"],
          summ["render_parity"]["mismatch_by_file"])
    print("spec drift:", len(drift), list(drift.items())[:5], "kinds mismatch:", len(kinds_mismatch))


if __name__ == "__main__":
    main()
