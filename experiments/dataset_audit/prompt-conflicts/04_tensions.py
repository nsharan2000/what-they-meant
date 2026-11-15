"""Step 4 — count training targets in tension with the prompt's own instructions (checks a-h) and classify every decision
target by what the rendered prompt literally prescribes (point 4).

Inputs: results/target_table.jsonl (03), annotations.py. Units counted:
  SFT   = sft_train decision-turn targets (497 rows; look-up turns are only used for the format check)
  DPO   = dpo_train chosen (496) — the same targets as SFT on the same renderings (rejected sides checked separately)
  GRPO  = grpo_train hidden labels (496) — the reward (wtm/reward.py) reads expected decision + gold commits
Literal-prescription rubric (per decision row; reasons listed in the output):
  prescribed    an instruction in THIS row's prompt names the situation and prescribes the target's action
  supported     an instruction licenses the target without naming it as the required action (e.g. W-new-payee)
  unspecified   no instruction in the prompt bears on the choice (all L0r P0 rows without data-level authorisations)
  competing     the prompt literally prescribes a different but adjacent action (an informed approval card stating the
                clashing field, under the env's card-content rule) — the target needs intent inference to prefer asking
  contradicted  an instruction in the prompt prescribes a different action (BZ ask-only-if-needed on an executable request,
                bizops 'stop and tell', 'just show the approval card' on a direct-commit target, ...)
  A row can be both prescribed (P2 hint) and contradicted (env policy): category 'hint_vs_policy'.
Outputs: results/tensions.json, results/literal_prescription.jsonl
    python3 -B experiments/dataset_audit/prompt-conflicts/04_tensions.py
"""
from __future__ import annotations

import collections
import json
import re

from pc_common import RES, dump, dump_jsonl, rel

UNITS = {"SFT": ("sft_train", "target"), "DPO": ("dpo_train", "chosen"), "GRPO": ("grpo_train", "hidden_label")}
METHOD_STRICT = re.compile(r"PayPal|Venmo|Zelle|Cash App|\bUPI\b|Visa|Mastercard|Amex|\*{4}\d|••\d|ending \d|Checking|Savings|"
                           r"\bACH\b|\bwire\b|Pay in 3|\bdebit\b|\bbank\b|payment app|QR code|Friends & Family|Goods & Services|"
                           r"on file", re.I)


def load():
    return [json.loads(l) for l in open(f"{RES}/target_table.jsonl")]


def unit_rows(rows, unit):
    f, role = UNITS[unit]
    return [r for r in rows if r["file"] == f and r["role"] == role and r["turn"] == "decision"]


def ex(r, extra=None):
    d = {"file": f"dataset/curated_v121/{r['file']}.jsonl", "line": r["i"], "scenario_id": r["scenario_id"],
         "level": r["level"], "prompt_setting": r["prompt_setting"]}
    if r.get("target") is not None:
        d["target"] = r["target"][:220]
    if extra:
        d.update(extra)
    return d


def breakdown(rs):
    return {k: dict(sorted(collections.Counter(r[k] for r in rs).items())) for k in ("t", "level", "prompt_setting", "env")}


def literal(r) -> tuple[str, list[str]]:
    """What the row's own prompt literally prescribes for the decision, compared with the target's action family."""
    lv1 = r["level"] != "L0r"
    p2 = r["prompt_setting"] == "P2"
    env = r["env"]
    pro, sup, comp, con = [], [], [], []
    if r["t"] == "A":
        if p2:
            pro.append("H-ask-one")
        if lv1:
            exe, clash = r.get("a_exec"), r.get("a_clash")
            if exe is False:
                pro.append({"wallet": "W-keep-light", "bills": "BL-no-guess", "shopping": "SH-ask",
                            "bizops": "BZ-ask-only-if-needed"}[env])
            if env == "wallet" and clash == "recipient":
                sup.append("W-new-payee")
            if env == "wallet" and r["fraud_tag"]:
                sup.append("W-warning-signs(tell+let decide)")
            if env == "bills" and clash == "payment_method":
                pro.append("BL-no-guess(card: look it up, or ask)")
            if env == "bizops" and exe:
                con.append("BZ-ask-only-if-needed")
            if env == "bizops" and r["fraud_tag"]:
                con.append("BZ-fraud/BZ-escalate(stop+tell+report path)")
            if env == "bizops" and r["scenario_id"] == "S14A":
                con.append("BZ-SA(SA-2 covers the retry: carry it out without a card)")
            if r.get("a_card_states_clash") and exe:
                comp.append({"wallet": "W-card-content", "bills": "BL-card-content", "shopping": "SH-card-content",
                             "bizops": "BZ-card-content"}[env])
        if pro and con:
            cat = "hint_vs_policy" if pro == ["H-ask-one"] else "prescribed_and_contradicted"
        elif con:
            cat = "contradicted"
        elif pro:
            cat = "prescribed"
        elif sup:
            cat = "supported" if not comp else "supported_with_competing_option"
        elif comp:
            cat = "competing"
        else:
            cat = "unspecified"
    else:
        sa_vis = bool(r.get("b_sa_visible"))
        if r["expected"] == "act":
            if p2:
                con.append("H-just-card")
            if r.get("b_act_uncovered") and lv1:
                con.append("BZ-card-required + SI 'Every vendor payment is approved on an approval card'")
            if sa_vis and lv1:
                (pro if env == "bizops" else sup).append({"shopping": "SH-oneoff/SH-SA (permissive 'can')",
                                                          "bizops": "BZ-SA (imperative)"}.get(env, "SA"))
            elif sa_vis:
                sup.append("data: request names item + price ceiling; member card 'Policy: spend only if user authorised'")
            if con and (pro or sup):
                cat = "contradicted_by_hint_but_permitted_by_policy" if con == ["H-just-card"] else "prescribed_and_contradicted"
            elif con:
                cat = "contradicted"
            elif pro:
                cat = "prescribed"
            elif sup:
                cat = "supported"
            else:
                cat = "unspecified"
        else:
            if p2:
                pro.append("H-just-card")
            if lv1:
                pro.append({"wallet": "W-card-required", "bills": "BL-card-required", "shopping": "SH-card",
                            "bizops": "BZ-card-required"}[env])
            if sa_vis:
                comp.append(("authorisation visible + " + {"shopping": "SH-SA", "bills": "BL-SA"}.get(env, "SA")) if lv1 else
                            "member card: 'no approval needed below ...' / 'auto-approved under ...'")
            if pro and comp:
                cat = "prescribed_with_competing_option"
            elif pro:
                cat = "prescribed"
            elif comp:
                cat = "competing"
            else:
                cat = "unspecified"
    return cat, [f"prescribes:{x}" for x in pro] + [f"supports:{x}" for x in sup] + [f"competing:{x}" for x in comp] + \
        [f"contradicts:{x}" for x in con]


def main():
    rows = load()
    res = {}
    # ---------- (a) ask-limiting rules vs A asks on executable literal requests ----------
    a = {}
    for u in UNITS:
        A = [r for r in unit_rows(rows, u) if r["t"] == "A" and r["level"] != "L0r"]
        exe = [r for r in A if r.get("a_exec")]
        bz = [r for r in exe if r["env"] == "bizops"]
        wl = [r for r in exe if r["env"] == "wallet"]
        wl_unlicensed = [r for r in wl if r.get("a_clash") != "recipient" and not r["fraud_tag"]]
        a[u] = {"A_rows_L1plus": len(A), "A_rows_L1plus_literal_executable": len(exe),
                "bizops_BZ_ask_only_if_needed_vs_executable": len(bz), "bizops_breakdown": breakdown(bz),
                "bizops_of_which_P2_hint_also_present": sum(r["prompt_setting"] == "P2" for r in bz),
                "wallet_W_keep_light_vs_executable": len(wl), "wallet_executable_without_licensing_rule": len(wl_unlicensed),
                "wallet_unlicensed_scenarios": sorted({r["scenario_id"] for r in wl_unlicensed}),
                "examples": [ex(r) for r in bz[:3]]}
    a["A_scenarios_literal_executable"] = None
    res["a_ask_limiting_vs_executable_A"] = a
    # ---------- (b) approval-card rule vs B targets; card content ----------
    b = {}
    for u in ("SFT", "DPO"):
        B = [r for r in unit_rows(rows, u) if r["t"] == "B"]
        cards = [r for r in B if r["decision"].startswith("approve")]
        commits = [r for r in B if r["decision"].startswith("commit")]
        meth = {e: {"cards": 0, "state_method_or_rail": 0} for e in ("wallet", "bills", "shopping", "bizops")}
        omit = []
        for r in cards:
            m = bool(METHOD_STRICT.search(r.get("card_summary", "")))
            meth[r["env"]]["cards"] += 1
            meth[r["env"]]["state_method_or_rail"] += m
            if not m:
                omit.append(r)
        # only rows whose prompt carries a card-content rule: L1+ (W/BL/SH/BZ-card-content) and wallet L0r (TD-wallet-card)
        ruled = [r for r in cards if r["level"] != "L0r" or r["env"] == "wallet"]
        ruled_omit = [r for r in ruled if not METHOD_STRICT.search(r.get("card_summary", ""))]
        b_rule = {e: {"cards_with_rule_in_prompt": sum(r["env"] == e for r in ruled),
                      "omit_method_or_rail": sum(r["env"] == e for r in ruled_omit)} for e in ("wallet", "bills", "shopping", "bizops")}
        b[u] = {"B_rows": len(B), "card_targets": len(cards), "direct_commit_targets": len(commits),
                "cards_where_card_content_rule_present": len(ruled), "of_which_omit_method_or_rail": len(ruled_omit),
                "by_env_rule_present": b_rule,
                "direct_commits_at_L1plus_where_card_rule_present": sum(r["level"] != "L0r" for r in commits),
                "card_content_method_or_rail_field": meth,
                "cards_omitting_method_or_rail": len(omit),
                "examples_omitting": [ex(r) for r in sorted(omit, key=lambda r: (r["env"], r["i"]))[::max(1, len(omit) // 3)][:3]]}
    res["b_card_rule_vs_B"] = b
    # ---------- (c) standing authorisations vs B targets; P2 hint vs direct commits ----------
    c = {}
    for u in UNITS:
        B = [r for r in unit_rows(rows, u) if r["t"] == "B"]
        act = [r for r in B if r["expected"] == "act"]
        act_p2 = [r for r in act if r["prompt_setting"] == "P2"]
        act_no_sa = [r for r in act if not r.get("b_sa_visible")]
        appr_sa = [r for r in B if r["expected"] == "approve" and r.get("b_sa_visible")]
        c[u] = {"act_rows": len(act), "act_rows_by_scenario": dict(collections.Counter(r["scenario_id"] for r in act)),
                "act_rows_P2_hint_says_just_show_card": len(act_p2), "act_P2_breakdown": breakdown(act_p2),
                "act_P2_examples": [ex(r) for r in act_p2[:3]],
                "act_rows_without_visible_authorisation": len(act_no_sa),
                "act_without_authorisation_detail": sorted({(r["scenario_id"], r["level"]) for r in act_no_sa}),
                "act_without_authorisation_examples": [ex(r) for r in act_no_sa[:3]],
                "approve_rows_where_authorisation_visible": len(appr_sa),
                "approve_with_authorisation_detail": sorted({(r["scenario_id"], r["level"], r["prompt_setting"]) for r in appr_sa}),
                "approve_with_authorisation_examples": [ex(r) for r in appr_sa[:3]]}
    res["c_standing_authorisation_and_hint_vs_B"] = c
    # GRPO reward treatment of these rows (hidden labels; reward tables in wtm/reward.py)
    g = unit_rows(rows, "GRPO")
    res["c_reward_view"] = {
        "grpo_approve_rows_with_visible_authorisation_where_commit_gold_scores_-0.5":
            sum(1 for r in g if r["t"] == "B" and r["expected"] == "approve" and r.get("b_sa_visible")),
        "grpo_act_rows_P2_where_hint_compliant_card_scores_+0.5_vs_commit_+1.0":
            sum(1 for r in g if r["t"] == "B" and r["expected"] == "act" and r["prompt_setting"] == "P2"),
        "grpo_act_rows_uncovered_V152B": sum(1 for r in g if r.get("b_act_uncovered")),
        "note": "wtm/reward.py ignores expected.accept (label adjudications are applied only by wtm/judge.py and "
                "experiments/replication/analysis.py), so on adjudicated 'accept approve+act' rows a direct commit is a strict "
                "success in the analysis but scores -0.5 in GRPO"}
    # ---------- (d) fraud rules vs A asks; reward tell 0.5 vs ask 1.0 ----------
    d = {}
    for u in UNITS:
        A = [r for r in unit_rows(rows, u) if r["t"] == "A" and r["fraud_tag"]]
        lv1 = [r for r in A if r["level"] != "L0r"]
        d[u] = {"fraud_tagged_A_rows": len(A), "at_L1plus_with_env_fraud_rule": len(lv1), "breakdown_L1plus": breakdown(lv1),
                "bizops_stop_and_tell_rule": sum(r["env"] == "bizops" for r in lv1),
                "wallet_tell_and_let_decide_rule": sum(r["env"] == "wallet" for r in lv1)}
        if u != "GRPO":
            d[u]["targets_that_are_questions"] = sum(r.get("msg", {}).get("n_qmarks", 0) > 0 for r in A)
            d[u]["targets_with_report_path"] = sum(r.get("msg", {}).get("report_path", False) for r in A)
            d[u]["examples_bizops"] = [ex(r) for r in lv1 if r["env"] == "bizops"][:3]
    d["reward_values"] = {"ask_informed": 1.0, "tell_informed": 0.5, "source": "wtm/reward.py A_TABLE"}
    res["d_fraud_rules_vs_A"] = d
    # ---------- (e) motives / accusations ----------
    e = {}
    for u in ("SFT", "DPO"):
        A = [r for r in unit_rows(rows, u) if r["t"] == "A"]
        scam = [r for r in A if r["msg"]["scam_words"]]
        e[u] = {"A_targets": len(A), "with_scam_or_fraud_words": len(scam),
                "bizops_A_targets": sum(r["env"] == "bizops" for r in A), "examples": [ex(r) for r in scam[:3]]}
    res["e_motives_vs_targets"] = e
    # ---------- (f) look-up rules ----------
    f = {}
    for u in UNITS:
        R = unit_rows(rows, u)
        zero = [r for r in R if r["n_lookups_in_prompt"] == 0]
        zero_b_l1 = [r for r in zero if r["t"] == "B" and r["level"] != "L0r"]
        f[u] = {"decision_rows": len(R), "decision_rows_with_zero_lookups_in_prompt": len(zero),
                "zero_lookup_scenarios": sorted({r["scenario_id"] for r in zero}), "zero_breakdown": breakdown(zero),
                "B_cards_prepared_without_lookup_at_L1plus(W-lookup-first)": len(zero_b_l1),
                "zero_lookup_P2(H-check)": sum(r["prompt_setting"] == "P2" for r in zero),
                "examples": [ex(r) for r in zero[:3]]}
    res["f_lookup_rules"] = f
    # ---------- (g) protocol format ----------
    sft = [r for r in rows if r["file"] in ("sft_train", "sft_val") and r["role"] == "target"]
    dpo = [r for r in rows if r["file"] == "dpo_train" and r["role"] == "chosen"]
    allt = sft + dpo
    A_msgs = [r for r in allt if r["kind"] == "message"]
    res["g_protocol_format"] = {
        "targets_checked": len(allt),
        "non_strict": sum(not r["strict_ok"] for r in allt),
        "with_flags(mixed_text/multi_call/...)": sum(bool(r["flags"]) for r in allt),
        "multi_line_targets": sum(r["target_newlines"] > 0 for r in allt),
        "tool_call_not_listed_in_row_prompt": sum(r["tool_listed_in_prompt"] is False for r in allt),
        "messages": len(A_msgs),
        "messages_with_2plus_questions": sum(r["msg"]["n_qmarks"] >= 2 for r in A_msgs),
        "messages_2plus_questions_examples": [ex(r) for r in A_msgs if r["msg"]["n_qmarks"] >= 2][:3],
        "messages_with_list_or_heading": sum(r["msg"]["has_list"] for r in A_msgs),
        "messages_naming_internal_tool_names(with underscore)": sum(any("_" in n for n in r["msg"]["tool_names_mentioned"])
                                                                    for r in A_msgs),
        "message_words": {"max": max(r["msg"]["words"] for r in A_msgs), "mean": round(sum(r["msg"]["words"] for r in A_msgs)
                                                                                     / len(A_msgs), 1)},
        "messages_over_40_words": sum(r["msg"]["words"] > 40 for r in A_msgs),
        "messages_over_40_words_scenarios": sorted({r["scenario_id"] for r in A_msgs if r["msg"]["words"] > 40}),
    }
    # DPO rejected: does any rejected side do what the row's prompt literally prescribes?
    rej = [r for r in rows if r["file"] == "dpo_train" and r["role"] == "rejected"]
    res["g_dpo_rejected"] = {
        "rejected": len(rej), "by_decision": dict(collections.Counter(r["decision"] for r in rej).most_common()),
        "rejected_commit_gold_on_B_approve_rows": [ex(r, {"sa_visible": r.get("b_sa_visible", False)}) for r in rej
                                                    if r["t"] == "B" and r["decision"] == "commit_gold"],
        "rejected_tell_informed_on_A": sum(r["t"] == "A" and r["decision"] == "tell_informed" for r in rej),
        "rejected_approve_informed_on_A": sum(r["t"] == "A" and r["decision"] == "approve_informed" for r in rej),
        "rejected_approval_card_on_B_act_rows": sum(r["t"] == "B" and r["expected"] == "act" and r["decision"].startswith("approve")
                                                     for r in rej)}
    # ---------- (h) level dependence ----------
    sftd = unit_rows(rows, "SFT")
    lvl = {}
    for sid in sorted({r["scenario_id"] for r in sftd if r.get("b_sa_covered")}):
        lvl[sid] = {r["level"]: bool(r.get("b_sa_visible")) for r in sftd if r["scenario_id"] == sid}
    res["h_level_dependence"] = {
        "env_policy_instructions_present": {"L0r": 0, "L1/L2/L3": "all catalogued env rules (see instructions.json)"},
        "authorisation_visibility_by_level(SFT decision rows)": lvl,
        "authorisation_visible_at_L1plus_but_not_L0r": sorted(k for k, v in lvl.items()
                                                              if not v.get("L0r", True) and any(v.get(x) for x in ("L1", "L2", "L3"))),
    }
    # L0r member cards (the seed's own text) that name tools absent from the row's tool docs ("Tools: get_billing (read-only), renew.")
    from pc_common import load_envs, load_rows, split_system, tools_in_docs
    envs = load_envs()
    bad_tools = []
    for f in ("sft_train", "dpo_train"):
        for i, r in enumerate(load_rows(f)):
            if r["level"] != "L0r" or r.get("turn", "decision") != "decision":
                continue
            sec = split_system(r["prompt"][0]["content"], envs[r["env_id"]], r["level"])
            m = re.search(r"Tools:\s*([^\n]*)", sec["member"])
            if not m:
                continue
            names = [re.sub(r"\s*\(.*?\)", "", x).strip(" .") for x in m.group(1).split(",")]
            miss = [n for n in names if n and n not in set(tools_in_docs(sec["tools"]))]
            if miss:
                bad_tools.append({"file": f"dataset/curated_v121/{f}.jsonl", "line": i, "scenario_id": r["scenario_id"],
                                  "level": r["level"], "prompt_setting": r["prompt_setting"], "card_names": miss})
    res["h_level_dependence"]["L0r_member_card_names_tools_not_in_tool_docs"] = {
        "rows": len(bad_tools), "by_file": dict(collections.Counter(b["file"].split("/")[-1] for b in bad_tools)),
        "names": dict(collections.Counter(n for b in bad_tools for n in b["card_names"])), "examples": bad_tools[:3]}
    # ---------- point 4: literal prescription per decision row ----------
    lit = []
    for u in UNITS:
        for r in unit_rows(rows, u):
            cat, why = literal(r)
            lit.append({"unit": u, "file": r["file"], "i": r["i"], "scenario_id": r["scenario_id"], "t": r["t"], "env": r["env"],
                        "level": r["level"], "prompt_setting": r["prompt_setting"], "expected": r["expected"],
                        "category": cat, "reasons": why})
    dump_jsonl("literal_prescription.jsonl", lit)
    summ = {}
    for u in UNITS:
        L = [x for x in lit if x["unit"] == u]
        summ[u] = {"n": len(L), "overall": dict(collections.Counter(x["category"] for x in L).most_common())}
        for key in ("t", "level", "prompt_setting", "env"):
            summ[u][f"by_{key}"] = {k: dict(collections.Counter(x["category"] for x in L if x[key] == k).most_common())
                                    for k in sorted({x[key] for x in L})}
        summ[u]["by_t_level_setting"] = {f"{t}|{lv}|{ps}": dict(collections.Counter(
            x["category"] for x in L if (x["t"], x["level"], x["prompt_setting"]) == (t, lv, ps)).most_common())
            for t in ("A", "B") for lv in ("L0r", "L1", "L2", "L3") for ps in ("P0", "P2")
            if any((x["t"], x["level"], x["prompt_setting"]) == (t, lv, ps) for x in L)}
        summ[u]["examples"] = {cat: [{k: x[k] for k in ("file", "i", "scenario_id", "level", "prompt_setting", "reasons")}
                                     for x in L if x["category"] == cat][:3] for cat in summ[u]["overall"]}
    res["point4_literal_prescription"] = summ
    p = dump("tensions.json", res)
    print(json.dumps({u: summ[u]["overall"] for u in summ}, indent=1))
    print("wrote", rel(p))


if __name__ == "__main__":
    main()
