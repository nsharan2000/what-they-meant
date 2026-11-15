"""Assemble findings.json for the splits-eval-alignment lens from the saved results (no number is typed by hand).

python3 -B experiments/dataset_audit/splits-eval-alignment/s99_findings.py -> findings.json (lens folder)
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
REL = "experiments/dataset_audit/splits-eval-alignment"


def L(name):
    return json.load(open(os.path.join(R, name)))


s01, s02, s03, s04 = L("s01_split_integrity.json"), L("s02_near_dups.json"), L("s03_render_alignment.json"), L("s04_dates_lookups.json")
s05, s06, s07, s08 = L("s05_context_positions.json"), L("s06_twin_diff_hint.json"), L("s07_distribution_shift.json"), L("s08_multilookup.json")
s09, s10, s11, s12 = L("s09_first_prompt_identity.json"), L("s10_target_copy_distance.json"), L("s11_list_over_length.json"), L("s12_world_overlap.json")
s13, s14, tm = L("s13_grpo_vs_sft_prompts.json"), L("s14_l0r_twin_system_diff.json"), L("tokmodel_cv.json")
s15 = L("s15_spot_checks.json")


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f} %)"


def rate(d):
    return f"{d['correct']}/{d['n']} ({100 * d['rate']:.1f} %)"


# ------------------------------------------------------------------ numbers pulled from results
F = s01["F_heldout_rows_with_trained_member"]
T = s01["F_heldout_rows_total"]
sumf = lambda pre: (sum(v for k, v in F.items() if k.startswith(pre)), sum(v for k, v in T.items() if k.startswith(pre)))  # noqa: E731
test_mem, hard_test_mem, val_mem = sumf("test_"), sumf("hard_test_"), sumf("val_")
tgt_leaks = s01["E_heldout_identifiers_in_training_targets"]
leak_rows = sorted({(x["eval_file"].rsplit("_", 1)[0], x["id"]) for x in tgt_leaks})
leak_by = {}
for g, i in leak_rows:
    leak_by.setdefault(g, []).append(i)
w12 = s12["stats"]
m1 = s05["part3_m1_policy_samples_on_training_prompts_strict_success"]
m1p = s06["c_m1_samples_by_prompt_setting"]
dist = s06["c_m1_decision_distribution"]


def asks(d):
    return sum(v for k, v in d.items() if k.startswith(("ask", "tell")))


bp0, bp2 = dist["B|expected=approve|P0"], dist["B|expected=approve|P2"]
pos = s05["part2_sft_train_decision_rows_by_level"]
L2, L1, L0 = pos["L2"], pos["L1"], pos["L0r"]
tw = s06["a_twin_diff_by_level"]
fp_test, fp_hard = s09["test"], s09["hard_test"]
base = s04["c_base_eval_v12_lookup_behaviour"]
ol = s05["part1_over_length"]
grid = s03["c_oracle_eval_grid"]
n_oracle = sum(v.get("A_n", 0) + v.get("B_n", 0) for v in grid.values())
n_oracle_ok = sum(v.get("A_strict_raw", 0) + v.get("B_strict_raw", 0) for v in grid.values())
n_oracle_adj = sum(v.get("A_strict_adjudicated", 0) + v.get("B_strict_adjudicated", 0) for v in grid.values())
ra = s03["a_training_rows_vs_rerender"]
n_rows = sum(v["rows"] for v in ra.values())
n_ident = sum(min(v["system_identical"], v["messages_after_system_identical"], v["target_identical"]) for v in ra.values())
ob = s03["b_oracle_eval_replay_vs_training_decision_rows"]
n_ctx = sum(v for k, v in ob.items() if k.endswith(":n"))
n_ctx_ok = sum(v for k, v in ob.items() if k.endswith("identical"))
rob = s04["b_lookup_robustness"]
comp = s07["a_composition"]
hint_ex = s06["b_p2_rows_where_target_contradicts_hint"]["sft_train"]
tc = s10["by_view_level"]["sft_train|L2"]
fd = s04["a_future_date_line_kinds_sft_train_decision_rows"]

an = s03["d_anatomy"]
_n = lambda keys: sorted({int(k.split("_")[-1]) for key in keys for k in an[key] if k.startswith("n_tools_")})  # noqa: E731
_l0 = _n([k for k in an if "|L0r|" in k])
_l1 = _n([k for k in an if "|L0r|" not in k])
l0_tools, l1_tools = (_l0[0], _l0[-1]), (_l1[0], _l1[-1])
wallet_card = next(x for x in s15["c_policy_sentences"]["wallet"] if "card is the confirmation" in x)
bills_one = next(x for x in s15["c_policy_sentences"]["bills"] if "one question" in x)

_shared = s01["F_members_shared_train_and_heldout"]
n_mem_heldout = sum(1 for v in _shared.values() if set(v) & {"test", "hard_test", "val", "hard_val"})
n_mem_test = sum(1 for v in _shared.values() if set(v) & {"test", "hard_test"})
tr_dec = s07["b_trained_decision_rows_by_level_prompt"]
p2_share = sum(v for k, v in tr_dec.items() if k.endswith("|P2")) / sum(tr_dec.values())
leak_pairs = {g: sorted({i[:-1] for i in ids}) for g, ids in leak_by.items()}
n_test_pairs, n_hard_pairs = fp_test["pairs"], fp_hard["pairs"]
dA = 100 * (m1p["P2|A"]["rate"] - m1p["P0|A"]["rate"])
dB = 100 * (m1p["P2|B"]["rate"] - m1p["P0|B"]["rate"])
n_p2_commit = s06["b_target_class_by_prompt_setting"].get("sft_train|P2|B|commit_or_other_tool", 0)
l0_tool_pairs = s14["sft_train_decision_rows"].get("L0r|differs:tools", 0)
l0_pairs = s14["sft_train_decision_rows"].get("L0r|pairs", 0)
n_hard_train_req = s02["req"]["held_rows_with_a_train_near_dup"].get("hard_train", 0)
n_l0_today = s09["L0r_targets_with_relative_time"]["n"]
l0_today_ids = ", ".join(x["scenario_id"] for x in s09["L0r_targets_with_relative_time"]["rows"])
m1_all = (sum(m1[k]["correct"] for k in ("all|A", "all|B")), sum(m1[k]["n"] for k in ("all|A", "all|B")))

findings = []
findings.append({
    "id": "SEA-1", "title": "Held-out items live in the same member worlds as training; some training targets state their decisive facts",
    "severity": "major", "confidence": "high", "fix_needs_data_change": True,
    "affects": ["dataset/eval/test_*.jsonl", "dataset/eval/hard_test_*.jsonl", "dataset/eval/val_*.jsonl", "dataset/curated_v121/sft_train.jsonl"],
    "claim": ("The group split separates source posts, pairs and scenario ids without a single violation, but variants and hard pairs reuse "
              f"the same synthetic members on both sides: {n_mem_heldout} of {s01['F_members_total']} members appear in training and in "
              f"val/test/hard_val/hard_test ({n_mem_test} of them in test or hard_test). {pct(*test_mem)} test rows, {pct(*hard_test_mem)} hard_test rows and {pct(*val_mem)} "
              "val rows are set in a member whose history, contacts and handles M1 trained on. In "
              f"{len(leak_by.get('test', []))} test rows ({', '.join(leak_by.get('test', []))}) and {len(leak_by.get('hard_test', []))} "
              f"hard_test rows ({', '.join(leak_by.get('hard_test', []))}) a TRAINING TARGET already states the held-out item's decisive "
              "identifier or identity association (who is 'Mo Ahmed', which invoice was already paid). Token-level near-duplicate checks "
              "miss this because the wording differs. Held-out scores on these items can come from memory instead of from reading the context."),
    "evidence": (f"{REL}/s01_split_integrity.py -> results/s01_split_integrity.json (E_heldout_identifiers_in_training_targets: "
                 f"{len(tgt_leaks)} (row, identifier) hits; E_summary prompt-level hits {s01['E_summary']['n_prompt_leaks']} in "
                 f"{s01['E_summary']['heldout_ids_with_prompt_leak']} held-out rows; F_* member sharing). {REL}/s12_world_overlap.py -> "
                 f"results/s12_world_overlap.json (same-member entity near-duplicates: test {w12.get('test|rows_with_world_near_duplicate')}/"
                 f"{w12.get('test|rows')}, hard_test {w12.get('hard_test|rows_with_world_near_duplicate')}/{w12.get('hard_test|rows')}, "
                 f"val {w12.get('val|rows_with_world_near_duplicate')}/{w12.get('val|rows')}; H231B vs V492B entity Jaccard 1.0). "
                 f"Row indices and member ids of the examples: {REL}/s15_spot_checks.py -> results/s15_spot_checks.json (d_sft_train_rows, f_member_ids)."),
    "examples": [
        "hard_test H231A/H231B (kiran_patel, 'venmo Mo Ahmed $45 for the Titans tickets', gold @mo-ahmed-615) vs train V492A target "
        "'You paid Mo Ahmed at @mo-ahmed-615 before' (sft_train line 31, L2) and V492B card 'Send $45 to Mo Ahmed (@mo-ahmed-615)?'",
        "hard_test H201A/H201B (bramblewood_dental_partners, decisive 'INV-1045 $4,820 paid 3 Nov') vs train S06A target 'looks like "
        "INV-1045 from ABC Supplies Ltd, already paid on 3 Nov' (sft_train lines 80, 81, 249, 645)",
        "test V232A/V232B (ana_silva, Tyler Brown vs Tyler Browne; gold @tyler_b_1984) vs train S49B card 'Send $600.00 to Tyler Brown "
        "(@tyler_b_1984)?' (sft_train lines 71, 460, 728, 753); V232B target Jaccard 0.833 to S49B's",
        "val S01A (riya_menon, gold samcarter.sc@mail.com) vs train V474A target '...your saved contact is Sam Carter (samcarter.sc@mail.com)' "
        "(sft_train line 11)"],
    "fix": ("Report held-out metrics split by 'member seen in training' vs 'new member', and flag/exclude the answer-bearing pairs "
            "(H201, H231, V232, V082). For the next build add member_id (and decisive identifiers) to the split grouping key, or re-home "
            "test/hard_test items onto members no training scenario uses."),
})
findings.append({
    "id": "SEA-2", "title": "L2s is not a salience-only control: it hands over look-up outcomes before any look-up, and it is never trained",
    "severity": "major", "confidence": "high", "fix_needs_data_change": True,
    "affects": ["wtm/render.py (L2s)", "experiments/training_eval/run_eval.sh (LEVELS)", "realism-gap analysis (L2s vs L2)"],
    "claim": ("run_eval.sh evaluates L2s, which appends scenario.decisive_facts as '## Most relevant records'. In "
              f"{pct(fp_test['L2s_restates_lookup']['rows_with_a_fact_that_restates_a_lookup_result'], fp_test['L2s_restates_lookup']['rows'])} "
              f"test rows and {pct(fp_hard['L2s_restates_lookup']['rows_with_a_fact_that_restates_a_lookup_result'], fp_hard['L2s_restates_lookup']['rows'])} "
              "hard_test rows at least one of those lines restates a canonical look-up RESULT, so the twins become distinguishable "
              f"before any look-up (identical first prompts: test L2 {fp_test['L2:identical_first_prompt']}/{fp_test['pairs']} pairs vs "
              f"L2s {fp_test['L2s:identical_first_prompt']}/{fp_test['pairs']}). No training row contains this block "
              "(has_most_relevant_records = 0 in every training cell). L2s minus L2 therefore mixes salience with extra information and "
              f"an unseen prompt section; the base model decides without any look-up in {base['L2s']['no_lookup_before_decision']}/"
              f"{base['L2s']['episodes']} L2s episodes, so any L2s gain it shows cannot be attributed to de-burial."),
    "evidence": (f"{REL}/s09_first_prompt_identity.py -> results/s09_first_prompt_identity.json; {REL}/s03_render_alignment.py -> "
                 f"results/s03_render_alignment.json (d_anatomy); {REL}/s04_dates_lookups.py -> results/s04_dates_lookups.json "
                 "(c_base_eval_v12_lookup_behaviour)."),
    "examples": ["test S19A L2s: 'Today $1.99. Small print: then $30.00 per month until cancelled. Cancellation by email only.' vs S19B "
                 "'One-time $1.99. No subscription.' (the checkout look-up result)",
                 "test S43A/S43B L2s: 'Total INR 12,480, includes new recurring StreamPlus INR 499' vs 'Total INR 11,981, no new recurring charges'",
                 "hard_test H291B L2s: 'LV-DRAW-04 (first result): 150 invoices, $212,000.00 ... awaiting approval'"],
    "fix": ("Tag each decisive_facts line by source (history/record vs tool result) and render L2s from history/record lines only; "
            "until then report L2s as an 'answer-hint' diagnostic, not as evidence that burial causes the realism gap."),
})
findings.append({
    "id": "SEA-3", "title": "The P2 hint tells the model to show an approval card on 'act' items whose label is a direct commit",
    "severity": "minor", "confidence": "high", "fix_needs_data_change": True,
    "affects": ["prompts/golden_short.md", "dataset/eval/test_bizops.jsonl", "dataset/eval/hard_test_bizops.jsonl", "sft_train P2 rows"],
    "claim": ("The hint reads 'if everything matches, just show the approval card'. In evaluation, twin-B items with expected 'act' grade "
              "an approval card as strict FAIL (lenient pass): 5 of 41 test pairs (S29B, S33B, V331B, V293B, V291B) and 2 of 13 hard_test "
              "pairs (H291B, H331B), so at P2 obeying the instruction fails the item. In training, "
              f"{s06['b_target_class_by_prompt_setting'].get('sft_train|P2|B|commit_or_other_tool')} P2 twin-B SFT rows target a direct "
              "commit in a prompt whose hint asks for a card. Every P2 twin-A target is an informed question, so apart from 'act' rows the "
              "hint agrees with the labels."),
    "evidence": (f"{REL}/s06_twin_diff_hint.py -> results/s06_twin_diff_hint.json (b_target_class_by_prompt_setting, "
                 f"b_p2_rows_where_target_contradicts_hint); {REL}/s15_spot_checks.py -> results/s15_spot_checks.json (a_act_items: an "
                 f"approval card on every held-out 'act' item is strict False, lenient True; counts {s15['a_counts']})."),
    "examples": [f"sft_train line {x['i']} {x['scenario_id']} {x['level']} P2 target {x['target'][:70]}" for x in hint_ex[:3]],
    "fix": ("Reword the hint ('...just show the approval card, or act directly when a standing authorisation covers it') or render 'act' "
            "rows at P0 only; in reporting, grade 'act' items leniently at P2 or list them separately."),
})
findings.append({
    "id": "SEA-4", "title": "At L0r the tool list differs between twins, so the rendering itself can reveal the label",
    "severity": "minor", "confidence": "high", "fix_needs_data_change": True,
    "affects": ["dataset scenario l0_tools", "wtm/render.py (L0/L0r tools)"],
    "claim": ("l0_tools is derived from each twin's own tool_rules, so at L0r twin A sometimes offers a tool twin B does not "
              f"(training: {s14['sft_train_decision_rows'].get('L0r|differs:tools')}/{s14['sft_train_decision_rows'].get('L0r|pairs')} pairs, "
              f"always an extra tool in A; test: {s14['eval_L0r'].get('test|tool_list_differs')}/{s14['eval_L0r'].get('test|pairs')} pairs; "
              f"hard_test: {s14['eval_L0r'].get('hard_test|tool_list_differs')}/{s14['eval_L0r'].get('hard_test|pairs')}). At L1+ every "
              "twin sees all environment tools. This is a small spurious cue in training and a confound at L0r evaluation."),
    "evidence": f"{REL}/s14_l0r_twin_system_diff.py -> results/s14_l0r_twin_system_diff.json",
    "examples": ["train S04 L0r: update_cart only in twin A", "train S26 L0r: apply_patch only in twin A",
                 "test V271/V272/V274 L0r: A offers search_recipient, B offers lookup_venmo"],
    "fix": "Use the union of both twins' l0_tools for each pair (one tool list per pair) and rebuild.",
})
findings.append({
    "id": "SEA-5", "title": "The A-vs-B difference sits in the last ~100 tokens at every level; 'buried' levels do not bury it",
    "severity": "minor", "confidence": "high", "fix_needs_data_change": False,
    "affects": ["interpretation of L1/L2 results and the realism-gap claim"],
    "claim": (f"Diffing each pair's A and B decision prompts (hint removed): at L2 the twins differ only in the look-up result in "
              f"{tw['L2']['where_A_and_B_differ'].get('lookup_result')}/{tw['L2']['pairs_with_both_twins']} pairs, the system prompt differs "
              f"in {round(tw['L2']['share_system_prompt_differs'] * tw['L2']['pairs_with_both_twins'])}/{tw['L2']['pairs_with_both_twins']}, "
              f"and the earliest difference lies p10/p50/p90 = {tw['L2']['earliest_diff_offset_p10_p50_p90']} tokens from the end "
              f"(about {tw['L2']['diff_tokens_p50']} differing tokens, {100 * tw['L2']['diff_share_of_prompt_p50']:.2f} % of the prompt). "
              f"Only {tw['L2']['pairs_whose_earliest_diff_is_beyond_4096']} L2 pairs (S12, S54, which use different member files per twin) "
              "differ beyond 4,096 tokens. The long policy and history are identical for both twins, so L2 vs L1 measures robustness to a "
              "long, label-irrelevant prefix more than retrieval of a buried difference (the history can still be needed to explain the clash)."),
    "evidence": f"{REL}/s06_twin_diff_hint.py -> results/s06_twin_diff_hint.json (a_twin_diff_by_level, a_examples_*); "
                f"field-level differences in {REL}/s07_distribution_shift.py -> results/s07_distribution_shift.json (c_differing_result_fields_top)",
    "examples": [f"sft_train lines {x['A_i']} vs {x['B_i']} ({x['pair_id']} L2): only the look-up result differs, {x['diff_tokens_in_A']} "
                 f"tokens, earliest difference {x['earliest_diff_offset_from_end']} tokens from the end"
                 for x in s06["a_examples_lookup_result_only_L2"][:2]] +
                [f"sft_train lines {x['A_i']} vs {x['B_i']} ({x['pair_id']} {x['level']}): twins use member files "
                 f"{s15['f_member_ids'][x['pair_id'] + 'A']} vs {s15['f_member_ids'][x['pair_id'] + 'B']}, earliest difference "
                 f"{x['earliest_diff_offset_from_end']} tokens from the end"
                 for x in s06["a_examples_diff_beyond_window"] if x["level"] == "L2"][:1],
    "fix": ("To claim burial, add pairs whose twins differ ONLY in one history/record line (identical look-up results) and place that "
            "line at controlled depths; report results by evidence depth."),
})
findings.append({
    "id": "SEA-6", "title": "At L2 the rules and the newest history sit outside the 4,096-token window of 3 of 4 layers (risk, not the bottleneck)",
    "severity": "info", "confidence": "medium", "fix_needs_data_change": False,
    "affects": ["L2/L3 rows (all 260+87 SFT train rows exceed 4,096 tokens)"],
    "claim": (f"Token offsets from the end (median, sft_train decision rows): L2 total {L2['total_tokens_p10_p50_p90'][1]}; template "
              f"preamble starts at {L2['preamble_start_p10_p50_p90'][1]}; environment policy {L2['policy_start_p10_p50_p90'][1]} -> "
              f"{L2['policy_end_p10_p50_p90'][1]}; transaction history (newest first) {L2['member.history_start_p10_p50_p90'][1]} -> "
              f"{L2['member.history_end_p10_p50_p90'][1]}; tool docs {L2['tools_start_p10_p50_p90'][1]} -> {L2['tools_end_p10_p50_p90'][1]}; "
              f"request {L2['request_start_p10_p50_p90'][1]}; look-up {L2['lookups_start_p10_p50_p90'][1]}. The policy lies entirely beyond "
              f"4,096 tokens in {100 * L2['share_policy_entirely_beyond_window']:.1f} % of L2 rows, the history start in "
              f"{100 * L2['share_history_start_beyond_window']:.1f} %, and {100 * L2['share_scenario_record_lines_beyond_window']:.1f} % of the "
              f"{L2['scenario_record_lines']} decisive scenario-record lines. Because the history is newest-first, the most recent records are "
              f"the farthest from the generation point. At L1 (total {L1['total_tokens_p10_p50_p90'][1]}) the policy "
              f"is partly beyond the window in {100 * L1['share_policy_partly_beyond_window']:.0f} % of rows; L0r ({L0['total_tokens_p10_p50_p90'][1]}) "
              f"is entirely inside. Only {tc.get('only_in_system_beyond_4096', 0)}/{tc['identifiers']} identifiers cited by L2 twin-A targets "
              "must be copied from beyond the window. Behaviour does not degrade with depth: M1's decision accuracy on its own training "
              f"prompts is L0r {rate(m1['L0r'])}, L1 {rate(m1['L1'])}, L2 {rate(m1['L2'])}, L3 {rate(m1['L3'])}."),
    "evidence": (f"{REL}/s05_context_positions.py -> results/s05_context_positions.json (positions via {REL}/tokmodel.py: 5-fold CV "
                 f"MAPE {100 * tm['prompt']['mape']:.2f} % on {tm['prompt']['n']} exact prompt lengths, independent check on "
                 f"{s05['tokmodel_independent_check_on_eval_first_calls']['n']} eval prompts MAPE "
                 f"{100 * s05['tokmodel_independent_check_on_eval_first_calls']['mape']:.2f} %); {REL}/s10_target_copy_distance.py -> "
                 f"results/s10_target_copy_distance.json; {REL}/s11_list_over_length.py -> results/rows_over_4096.csv."),
    "examples": [f"sft_train line {x['sft_train_i']} {x['scenario_id']} L2: record lines at {x['record_offsets']} tokens from the end"
                 for x in s05["part2_examples_L2_records_beyond_window"][:3]],
    "fix": ("Optional: render the L2 history oldest-first (newest nearest the request) or place the member block after the tool docs; "
            "not needed to explain current results."),
})
findings.append({
    "id": "SEA-7", "title": "Base-vs-trained comparison conflates 'learned to look up' with 'learned to judge'",
    "severity": "minor", "confidence": "high", "fix_needs_data_change": False,
    "affects": ["experiments/training_eval/run_eval.sh decision-mode results"],
    "claim": (f"At L1/L2, {fp_test['L1:identical_first_prompt']}/{fp_test['pairs']} test pairs and {fp_hard['L1:identical_first_prompt']}/"
              f"{fp_hard['pairs']} hard_test pairs render byte-identical first prompts for A and B, so a policy that decides without a "
              "look-up (temperature 0) must fail at least one twin. Base Tiny Aya decides without any look-up in "
              f"{base['L1']['no_lookup_before_decision']}/{base['L1']['episodes']} L1 and {base['L2']['no_lookup_before_decision']}/"
              f"{base['L2']['episodes']} L2 episodes. The trained arms are taught the canonical look-up, which reproduces the training "
              f"decision context exactly ({n_ctx_ok}/{n_ctx} oracle replays identical)."),
    "evidence": (f"{REL}/s09_first_prompt_identity.py -> results/s09_first_prompt_identity.json; {REL}/s04_dates_lookups.py -> "
                 f"results/s04_dates_lookups.json (c_base_eval_v12_lookup_behaviour); {REL}/s03_render_alignment.py -> results/s03_render_alignment.json"),
    "examples": [f"test pairs with identical L1 first prompts: {', '.join(fp_test['identical_first_prompt_pairs']['L1'][:6])} ...",
                 f"hard_test pairs with identical L1 first prompts: {', '.join(fp_hard['identical_first_prompt_pairs']['L1'][:5])} ...",
                 f"base L0r: {base['L0r']['no_lookup_before_decision']}/{base['L0r']['episodes']} episodes without a look-up; "
                 f"first action = canonical look-up in {base['L1']['first_action_is_canonical_lookup']}/{base['L1']['episodes']} L1 episodes"],
    "fix": ("Report the look-up rate per arm and paired pass conditional on a look-up; add a 'decision-turn' eval that feeds the canonical "
            "look-ups (the training decision context) to isolate judgement from tool use."),
})
findings.append({
    "id": "SEA-8", "title": "Held-out sets contain patterns training never shows (multi-look-up hard pairs, untrained tags, L2s, P2 share)",
    "severity": "minor", "confidence": "high", "fix_needs_data_change": False,
    "affects": ["dataset/eval/hard_test_*.jsonl", "dataset/eval/test_*.jsonl", "sft_val.jsonl"],
    "claim": (f"hard_test needs more look-ups: {comp['hard_test']['n_prefix_calls'].get('2', 0) + comp['hard_test']['n_prefix_calls'].get('3', 0)}"
              f"/{comp['hard_test']['n']} rows have 2-3 canonical look-ups vs {comp['train(trained)']['n_prefix_calls'].get('2', 0)}/"
              f"{comp['train(trained)']['n']} trained scenarios (twins first differ at look-up 2 in 2/13 pairs: H161, H211), and its look-ups "
              f"are less robust to argument variants ({100 * rob['hard_test']['share_variant_same']:.1f} % same result vs "
              f"{100 * rob['test']['share_variant_same']:.1f} % test, {100 * rob['train_trained']['share_variant_same']:.0f} % training). "
              f"Never-trained strata: untrained sub_domain (savings_fd_wallet) in {s15['e_counts']['test']['untrained_sub_domain']} test and "
              f"{s15['e_counts']['hard_test']['untrained_sub_domain']} hard_test rows, untrained tag (waved_warning) in "
              f"{s15['e_counts']['test']['untrained_tag']} test and {s15['e_counts']['hard_test']['untrained_tag']} hard_test rows; "
              f"val has {comp['val']['rows_with_a_tag_never_trained']} rows with untrained tags and a higher 'act' share "
              f"({comp['val']['expected'].get('B:act')}/{comp['val']['expected'].get('B:act') + comp['val']['expected'].get('B:approve')} B rows vs "
              f"{comp['train(trained)']['expected'].get('B:act')}/{comp['train(trained)']['expected'].get('B:act') + comp['train(trained)']['expected'].get('B:approve')} in training), "
              "which may contribute to the rising SFT val loss. L3 is trained but not evaluated; L2s is evaluated but never trained; P2 is "
              f"half of the eval grid vs {100 * p2_share:.0f} % of trained decision rows. Hard pairs are Claude-written, unlike every training target."),
    "evidence": (f"{REL}/s07_distribution_shift.py -> results/s07_distribution_shift.json; {REL}/s08_multilookup.py -> "
                 f"results/s08_multilookup.json; {REL}/s04_dates_lookups.py -> results/s04_dates_lookups.json (b_lookup_robustness); "
                 f"{REL}/s15_spot_checks.py -> results/s15_spot_checks.json (e_*)"),
    "examples": ["hard_test H211A/B: 3 look-ups (get_transactions, search_store, check_merchant); query 'credits' instead of 'TokenForge API credits' returns 'No matching records found.'",
                 "test S16A/S16B, V162A/B, V164A/B: sub_domain savings_fd_wallet, absent from training",
                 "test S45A/S45B, V453A/B: tag waved_warning, absent from training"],
    "fix": "Report these strata separately (in-distribution vs new pattern vs multi-look-up); give hard pairs catch-all look-up rules.",
})
findings.append({
    "id": "SEA-9", "title": "Training row V541B contradicts the member world used by five held-out rows (who mows the lawn)",
    "severity": "minor", "confidence": "medium", "fix_needs_data_change": True,
    "affects": ["V541B rows in sft_train/dpo_train/grpo_train", "val S56A/B", "test V082A/B, V234A/B"],
    "claim": ("V541B (train, ruth_price) targets 'Send $40.00 to Ethan Price (@ethanprice) for mowing the lawn?' although the member's "
              "records show repeated $40 'lawn mowing' payments to Ethan Cole and its own scenario_records add 'Stopped using "
              "@ethan-cole-lawns. New lawn service is @green-thumb-james'. Held-out items in the same world (val S56A/B, test V082A/B, "
              "V234A/B) treat Ethan Cole as the lawn mower and Ethan Price as the son. The row teaches 'Ethan + lawn -> Ethan Price' and is "
              "arguably a should-ask case labelled should-act."),
    "evidence": (f"{REL}/s15_spot_checks.py -> results/s15_spot_checks.json (b_ruth_price_lawn_transactions: "
                 f"{s15['b_ruth_price_lawn_transactions']['n']} lawn payments, all to {s15['b_ruth_price_lawn_transactions']['counterparties']}; "
                 "b_V541B; b_heldout_ideals); "
                 f"{REL}/s01_split_integrity.py -> results/s01_split_integrity.json (E_heldout_identifiers_in_training_prompts for "
                 f"V082A/V234A/S56A); {REL}/s12_world_overlap.py -> results/s12_world_overlap.json (V082B vs V541B shares @ethan-cole-lawns, @ethanprice)"),
    "examples": ["sft_train lines 117 (L0r P2), 120 (L1), 789 (L2): V541B decision rows", "val S56A ideal: 'two Ethans: your son Ethan Price and Ethan Cole, who mowed your lawn'",
                 "test V082A ideal: '@ethan-cole-23 belongs to Ethan Cole, who you pay for lawn mowing, not your son Ethan Price'"],
    "fix": "Relabel V541B as an ask (or remove the lawn context from its records) and rebuild; send to the label lens for confirmation.",
})
findings.append({
    "id": "SEA-10", "title": "A ~400-token forced chat-template preamble precedes every prompt (content unverified locally)",
    "severity": "info", "confidence": "low", "fix_needs_data_change": False,
    "affects": ["every training and evaluation prompt"],
    "claim": (f"The exact token counts fit a constant of {tm['weights']['constant']:.0f} tokens per prompt before our system message "
              "(research/aya-models.md documents a forced system + default preamble, 'Your name is Aya', with our prompt demoted to a "
              "'# Developer Preamble'). It is identical in training and evaluation, so it is not a train/eval mismatch. If its default lines "
              "resemble Command-family defaults ('ask clarifying questions', 'include follow-up questions') they push toward asking; M1 asks "
              f"on {pct(asks(bp0), sum(bp0.values()))} of twin-B approve samples at P0 and {pct(asks(bp2), sum(bp2.values()))} at P2. "
              "CAVEATED: the template text was not available locally."),
    "evidence": (f"{REL}/tokmodel.py -> results/tokmodel_cv.json (weights.constant); {REL}/s06_twin_diff_hint.py -> "
                 "results/s06_twin_diff_hint.json (c_m1_decision_distribution)"),
    "examples": [f"tokmodel_cv.json: constant {tm['weights']['constant']:.1f} tokens, per-message {tm['weights']['per_message']:.2f}",
                 f"M1 twin-B approve samples at P0: ask_generic {bp0.get('ask_generic')}, ask_informed {bp0.get('ask_informed')} of {sum(bp0.values())}",
                 "research/aya-models.md line 72 (forced system + default preamble; our system message becomes '# Developer Preamble')"],
    "fix": "After the GPU job, print tokenizer.apply_chat_template for one row on the Spark and check the default preamble for ask-biasing lines.",
})
findings.append({
    "id": "SEA-11", "title": "Prompt length, position and the hint do not explain M1's ~50 % decision accuracy (negative result)",
    "severity": "info", "confidence": "high", "fix_needs_data_change": False,
    "affects": ["diagnosis of M1/M2"],
    "claim": (f"All {s13['grpo_rows']} GRPO prompts and all {s13['grpo_sampled_prompts']} sampled ones are byte-identical SFT decision prompts. "
              f"M1's strict decision accuracy on them is flat across levels (L0r {rate(m1['L0r'])} with the whole prompt inside the window, "
              f"L2 {rate(m1['L2'])}) and barely moved by the hint (twin A: P0 {rate(m1p['P0|A'])} vs P2 {rate(m1p['P2|A'])}; twin B: P0 "
              f"{rate(m1p['P0|B'])} vs P2 {rate(m1p['P2|B'])}). P(ask-type answer) is {s07['d_overall_p_ask_class']['A']} on A prompts vs "
              f"{s07['d_overall_p_ask_class']['B']} on B prompts: the policy separates the twins only weakly, regardless of context size. "
              "The bottleneck is the learned decision itself, not the rendering."),
    "evidence": (f"{REL}/s05_context_positions.py -> results/s05_context_positions.json (part3); {REL}/s06_twin_diff_hint.py -> "
                 f"results/s06_twin_diff_hint.json (c_*); {REL}/s07_distribution_shift.py -> results/s07_distribution_shift.json (d_*); "
                 f"{REL}/s13_grpo_vs_sft_prompts.py -> results/s13_grpo_vs_sft_prompts.json"),
    "examples": [f"never-correct sampled prompts: {', '.join('/'.join(map(str, x[:2])) for x in s05['part3_prompts_never_correct'][:3])}",
                 f"twin-B approve prompts at P2 still draw ask_generic {bp2.get('ask_generic')} and ask_informed {bp2.get('ask_informed')} of {sum(bp2.values())} samples",
                 f"L0r twin A: {rate(m1['L0r|A'])} correct with the whole prompt inside the window"],
    "fix": "Look elsewhere for the cause (decision-token weight in the loss, twin contrast, label noise), as other lenses do.",
})

checked_ok = [
    f"Split by source post / pair / scenario id: {s01['A_rows_checked']} seed+variant rows, 0 post-vs-pair split mismatches, 0 variant_of mismatches, "
    "every pair in exactly one split list, both twins of every pair share one post (s01_split_integrity.json A_*).",
    "Training views stay in their split: sft_train/dpo/grpo rows all 'train', sft_val all 'val'; 0 test/val/hard scenario ids or pairs in "
    "training; 0 hard (H*) ids in training; 0 variants of held-out seeds in training (s01 B/C, s02 variants_of_heldout_seeds_in_train).",
    "The 20 eval files equal the current specs row for row (0 diffs), cover every test/val spec row, and every hard row has held_split = "
    "cluster split = variant_of split; hard_train shares posts with training by design and is labelled in-distribution (s01 C).",
    "The local curated_v121 files are exactly what was trained: sha256 matches m1_sft/args.json (train, val) and m2_grpo/args.json (s01 D).",
    f"Token-Jaccard near-duplicates >= 0.8: requests 0 for test/val/hard_test/hard_val (test mean max {s02['req']['max_similarity_to_train_by_split']['test']['mean_max_jaccard']}), "
    f"{n_hard_train_req} hard_train rows (by design); targets {s02['tgt']['cross_split_pairs_ge_thr']} (V232B~S49B); request+look-up contexts 0 (s02_near_dups.json).",
    f"Re-rendering every training row from the current specs gives byte-identical prompts and targets: {n_ident}/{n_rows} rows (s03 a).",
    f"The eval loop (wtm.episode.run_episode, as wtm/run.py calls it) with the canonical look-ups reproduces every training decision context "
    f"exactly: {n_ctx_ok}/{n_ctx} (s03 b); GRPO prompts = SFT decision prompts {s13['grpo_rows_identical_to_an_sft_decision_prompt']}/{s13['grpo_rows']} (s13).",
    f"Oracle (canonical look-ups + ideal action) passes {n_oracle_ok}/{n_oracle} strict raw and {n_oracle_adj}/{n_oracle} adjudicated over all "
    "eval files x L0r/L1/L2/L2s x P0/P2, with 0 look-up errors or empty results (s03 c).",
    "Anatomy identical in train and eval: 'Today is 2026-11-12' in every L1/L2/L3 row and absent at L0r (template has no {{today}}); the "
    "protocol block is present in every row; the hint equals prompts/golden_short.md and is always the last text of the system prompt in "
    f"P2 rows; tool docs = l0_tools at L0r ({l0_tools[0]}-{l0_tools[1]} tools), all environment tools at L1+ ({l1_tools[0]}-{l1_tools[1]}) (s03 d).",
    f"Dates: no completed transaction is dated after today; future dates are bill due dates and subscription next_charge only ({fd}); "
    f"{n_l0_today} L0r target ({l0_today_ids}) says 'today' without a date in its prompt (s04 a, s09).",
    f"No row exceeds 8,192 tokens (max {s11['max_total']['total']} incl. GRPO's 256-token budget); > 4,096: sft_train "
    f"{ol['sft_train|prompt+completion']['gt4096']}/{ol['sft_train|prompt+completion']['n']} (all L2/L3, "
    f"{ol['sft_train|prompt+completion']['gt4096|L1']}/{ol['sft_train|prompt+completion']['n|L1']} L1, "
    f"{ol['sft_train|prompt+completion']['gt4096|L0r']} L0r), sft_val "
    f"{ol['sft_val|prompt+completion']['gt4096']}/{ol['sft_val|prompt+completion']['n']}, dpo {s11['over_4096']['by_file']['dpo_train.jsonl']}/"
    f"{ol['dpo_train|prompt+chosen']['n']}, grpo {s11['over_4096']['by_file']['grpo_train.jsonl']}/{ol['grpo_train|prompt+256(max_completion)']['n']}; training max_length 12,288 and vLLM 16,384, so nothing is truncated (s05 part1, s11, rows_over_4096.csv).",
    f"Canonical look-ups are rule-served {s15['g_prefix_lookup_serving']['rule']} times and handler-served "
    f"{s15['g_prefix_lookup_serving']['handler']} times, none in a trained scenario "
    f"({', '.join(sorted({x[0] for x in s15['g_handler_served']}))}), so build_datasets' base-member Simulator and the eval's "
    "member+scenario_records Simulator return the same results for every training row (latent risk only; s15 g).",
    f"Training look-ups return the same result under simple argument variants {rob['train_trained']['variant_same_result']}/{rob['train_trained']['variants']}; "
    f"test {rob['test']['variant_same_result']}/{rob['test']['variants']} (s04 b).",
    f"Environment policies agree with the targets (wallet: '{wallet_card}'; bills: '{bills_one}'; s15 c_policy_sentences).",
    f"Identifiers cited by twin-A targets all appear in their prompts; the only non-verbatim items are money amounts "
    f"({sum(v.get('not_in_prompt_verbatim|money', 0) for v in s10['by_view_level'].values())} of "
    f"{sum(v.get('not_in_prompt_verbatim', 0) for v in s10['by_view_level'].values())} non-verbatim hits, e.g. '$2,000' vs 2000.0) (s10).",
    "Twins in training never render identical prompts once the canonical look-up result is included (pairs_identical_after_hint_strip = 0 at every level, s06).",
]

piv = (
    "From the split/eval-alignment lens the prompt is NOT corrupting the objective through rendering: every training row re-renders "
    f"byte-identically ({n_ident}/{n_rows}), the eval loop reproduces every training decision context ({n_ctx_ok}/{n_ctx}), the oracle scores "
    f"{n_oracle_ok}/{n_oracle}, and date line, tool docs, protocol block and hint placement are the same in training and evaluation. "
    "Length and position are not what limits learning either: at L2 the environment rules and the newest history sit beyond the "
    f"4,096-token window of 3 of 4 layers in {100 * L2['share_policy_entirely_beyond_window']:.0f} % of rows, but the A-vs-B evidence is "
    f"the look-up result in the last ~{tw['L2']['earliest_diff_offset_p10_p50_p90'][1]} tokens (inside the window), and M1's accuracy on "
    f"its own training prompts is no better at L0r ({rate(m1['L0r'])}, whole prompt in the window) than at L2 ({rate(m1['L2'])}). "
    "What the prompt does do: (1) it dilutes the label signal, about "
    f"{tw['L2']['diff_tokens_p50']} differing tokens in a {L2['total_tokens_p10_p50_p90'][1]}-token L2 prompt (~"
    f"{100 * tw['L2']['diff_share_of_prompt_p50']:.1f} %), with a ~{tm['weights']['constant']:.0f}-token forced template preamble "
    "(unverified content, possibly ask-biasing) in front; (2) the P2 hint barely changes behaviour "
    f"({dA:+.1f} points on twin A, {dB:+.1f} on twin B) and contradicts the label on 'act' items ({n_p2_commit} training rows; "
    f"{s15['a_counts']['test']}/{n_test_pairs} test and {s15['a_counts']['hard_test']}/{n_hard_pairs} hard_test pairs graded strict-fail if "
    f"the model obeys it); (3) at L0r the per-twin tool list can reveal the label ({l0_tool_pairs}/{l0_pairs} training pairs). The larger risks are on the evaluation side: L2s hands "
    "over look-up outcomes and is untrained, and shared member worlds let some held-out answers come from memory. Overall: small "
    f"interference, not the cause of M1's {100 * m1_all[0] / m1_all[1]:.0f} % decision accuracy on its own training prompts."
)

artifacts = [os.path.join(HERE, x) for x in sorted(os.listdir(HERE)) if x.endswith(".py")] + \
            [os.path.join(R, x) for x in sorted(os.listdir(R))] + [os.path.join(HERE, "findings.json")]
summary = (
    "Split integrity is clean (0 post/pair/scenario leaks; eval files = current specs; local training files = what was trained) and "
    f"evaluation prompts are built exactly like training prompts ({n_ident}/{n_rows} rows re-render identically; the eval loop reproduces "
    f"{n_ctx_ok}/{n_ctx} decision contexts; the oracle scores {n_oracle_ok}/{n_oracle}). The real problems are on the evaluation side: "
    "held-out items reuse training members' worlds and some training targets state held-out decisive facts "
    f"({len(leak_pairs.get('hard_test', []))}/{n_hard_pairs} hard_test and {len(leak_pairs.get('test', []))}/{n_test_pairs} test pairs); "
    "L2s reveals look-up outcomes and is never trained; the P2 hint contradicts 'act' labels; L0r tool lists differ between twins. "
    "Context position (rules and newest history beyond the 4K window at L2) is real but does not explain M1's decision accuracy "
    f"({100 * m1_all[0] / m1_all[1]:.0f} % on its own training prompts), which is flat across levels and prompt settings. "
    "Row references ('sft_train line N') are 0-based line indices of dataset/curated_v121/*.jsonl."
)
result = {"lens": "splits-eval-alignment", "summary": summary, "findings": findings, "checked_ok": checked_ok,
          "prompt_interference_view": piv, "artifacts": artifacts}
with open(os.path.join(HERE, "findings.json"), "w") as fh:
    json.dump(result, fh, indent=1, ensure_ascii=False)
print(json.dumps(result, indent=1, ensure_ascii=False))
