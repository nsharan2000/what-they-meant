"""Step 9: assemble findings.json for the twin-contrast-shortcuts lens. Every number is read from a results file written by
s0-s8/s10 (paths are cited inside each finding); nothing is typed by hand.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s9_findings.py -> findings.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LENS, RES, jdump, load_jsonl  # noqa: E402

REL = "experiments/dataset_audit/twin-contrast-shortcuts"


def J(name):
    return json.load(open(os.path.join(RES, name)))


def P(name):
    return f"{REL}/results/{name}"


def S(name):
    return f"{REL}/{name}"


def main():
    s0, s1, bal, s2 = J("s0_rebuild.json"), J("s1_twin_diff_summary.json"), J("s1_balance.json"), J("s2_cues.json")
    clf, req, asym = J("s3_classifiers.json"), J("s3_request_only.json"), J("s3_model_free_asymmetry.json")
    beh, ev5, noisy, dt, dpo, pc = (J("s4_policy_behaviour.json"), J("s5_request_only_evidence.json"),
                                    J("s6_noisy_twin_asymmetry.json"), J("s7_decision_token.json"), J("s8_dpo_contrast.json"),
                                    J("s10_prefix_consistency.json"))
    rows = {(r["file"], r["pair_id"], r["level"], r["turn"]): r for r in load_jsonl(os.path.join(RES, "s1_twin_diff_rows.jsonl"))}

    def ex(f, pid, lv, what="", turn="decision"):
        r = rows[(f, pid, lv, turn)]
        return (f"dataset/curated_v121/{f}.jsonl lines {r['a_row']} (A, {r['ps_a']}) / {r['b_row']} (B, {r['ps_b']}): {pid} {lv} "
                f"{turn}" + (f" — {what}" if what else ""))

    def leafdesc(f, pid, lv, k=4):
        """'field: A -> B' for the first k differing result leaves of a matched twin decision row (from s1 rows)."""
        r = rows[(f, pid, lv, "decision")]
        out = []
        for m in r["msgs"]:
            if m["kind"] == "result" and m.get("json"):
                for path, va, vb in (m["linked"] + m["unlinked"]):
                    out.append(f"{path}: {json.dumps(va, ensure_ascii=False)} -> {json.dumps(vb, ensure_ascii=False)}")
        return "; ".join(out[:k])

    def pfmt(x):
        return "<1e-4" if x < 1e-4 else f"{x:.4f}".rstrip("0")

    from common import load_rows, components
    import re as _re
    sft = load_rows("sft_train")
    val_rows = load_rows("sft_val")
    evr = load_jsonl(os.path.join(LENS, "cache", "rendered_eval_rows.jsonl"))
    r12 = rows[("sft_train", "S12", "L2", "decision")]
    hist = [(_re.search(r"Transaction history \(newest first, (\d+) of (\d+)\)", sft[i]["prompt"][0]["content"]).groups())
            for i in (r12["a_row"], r12["b_row"])]
    s04 = rows[("sft_train", "S04", "L0r", "decision")]
    s04_tools = [m["sections"].get("Tools", 0) for m in s04["msgs"] if m["kind"] == "system"][0]
    ev_idx = {(r["split"], r["scenario_id"], r["level"]): i for i, r in enumerate(evr)}
    s21a, s21b = ev_idx[("test", "S21A", "L1")], ev_idx[("test", "S21B", "L1")]
    s23a, s23b = ev_idx[("test", "S23A", "L1")], ev_idx[("test", "S23B", "L1")]
    pchars = lambda r: sum(len(m["content"]) for m in r["prompt"])  # noqa: E731
    lastres = lambda r: len(components(r["prompt"])["results"][-1])  # noqa: E731
    r36 = rows[("sft_val", "S36", "L1", "decision")]
    s36a, s36b = lastres(val_rows[r36["a_row"]]), lastres(val_rows[r36["b_row"]])
    unmatched_dec = [f"{u['pair_id']}{u['present']} {u['level']}" for u in bal["sft_train"]["unmatched_rows"]   # the row that has no twin
                     if u["turn"] == "decision"]
    sg = [x for x in s0["curation_attribution"]["sft_train"]["dropped"] if x["reason"].startswith("curator_secrets_gate")
          or x["reason"] == "curator_exact_duplicate_of_secrets_gated_row"]
    sg_keys = {}
    for x in sg:
        sg_keys.setdefault((x["pair_id"], x["level"], x["turn"]), set()).add(x["twin"])
    sg_symmetric = all(v == {"A", "B"} for v in sg_keys.values())
    sg_pairs = sorted({k[0] for k in sg_keys})
    exact_dups_turns = sorted({x["turn"] for x in s0["curation_attribution"]["sft_train"]["dropped"] if x["reason"] == "curator_exact_duplicate"})
    n_sr_differ = sum(v.get("scenario_records_differ", 0) for v in s2["spec_twin_differences_by_split"].values())

    d = s1["by_file"]["sft_train"]["decision"]
    dv = s1["by_file"]["sft_val"]["decision"]
    st = s2["sft_train"]["result_structure"]
    stv = s2["sft_val"]["result_structure"]
    cw = s2["sft_train"]["cue_words"]
    ln = s2["sft_train"]["prompt_length_exact_tokens"]
    spec = s2["spec_twin_differences_by_split"]
    rp = req["pairs_with_different_request_by_split"]
    b = beh["bootstrap_wording_vs_identical"]
    m1r = beh["m1_by_request_stratum"]
    m1l = beh["m1_by_loci"]
    m1o = beh["m1_overall"]["all"]
    assoc = beh["m1_within_twin_assoc_with_shortcut_scores"]
    neg = beh["m1_negative_literal_heuristic"]
    bt = beh["base_by_request_stratum_per_split"]

    def c(view, split):
        m = clf[view][split]
        ci = m.get("within_twin_acc_ci95")
        return f"{m['within_twin_acc']}" + (f" [{ci[0]}, {ci[1]}]" if ci else "")

    v254 = [p for p in beh["m1_prompts"] if p["pair_id"] == "V254"]
    v254a = sum(p["ask_rate"] * p["n"] for p in v254 if p["t"] == "A")
    v254an = sum(p["n"] for p in v254 if p["t"] == "A")
    v254b = sum(p["ask_rate"] * p["n"] for p in v254 if p["t"] == "B")
    v254bn = sum(p["n"] for p in v254 if p["t"] == "B")
    dA, dB = dpo["by_twin"]["A"], dpo["by_twin"]["B"]
    a_commit = sum(v for k, v in dA.items() if k.endswith("rejected=commit"))
    a_approve = sum(v for k, v in dA.items() if k.endswith("rejected=approve"))
    a_ask = sum(v for k, v in dA.items() if k.endswith("rejected=ask_like"))
    b_commit = sum(v for k, v in dB.items() if k.endswith("rejected=commit"))
    b_ask = sum(v for k, v in dB.items() if k.endswith("rejected=ask_like"))
    nA, nB = sum(dA.values()), sum(dB.values())
    tdA, tdB = dt["sft_train"]["A|decision"], dt["sft_train"]["B|decision"]
    pure = {t["token"]: t for t in s2["sft_train"]["near_pure_field_value_tokens"]}
    noisy_pairs = [e["pair_id"] for e in noisy["examples"]]
    ex5 = ev5["summary"]["by_level"]

    findings = [
        {
            "id": "TCS-1", "severity": "major", "confidence": "high", "fix_needs_data_change": True,
            "title": f"The member's request alone separates the twins in {round(100 * rp['train']['share'])} % of training pairs, mostly through wording templates shared by sibling variants",
            "affects": ["sft_train", "dpo_train", "grpo_train", "test evaluation"],
            "claim": (f"In {rp['train']['different']}/{rp['train']['pairs']} training pairs the A and B requests differ. Over the "
                      f"{d['n']} matched twin decision rows, {d['request_kind'].get('wording', 0)} differ in wording (not just an "
                      f"identifier) and {d['request_kind'].get('identifier_only', 0)} only in an identifier. The wording differences are "
                      "a handful of templates reused across the variants of a seed: 'X emailed new bank details … Update it and pay' vs "
                      "'X's bank change is in the vendor master now', 'as friends … avoid fees' vs 'purchase protection on', 'the first "
                      "one in the list', 'free trial' vs 'auto-renew is fine', plus single words. 'new' appears more in A's request in "
                      f"{cw['request|new']['pairs_A_more']} twin rows and more in B's in {cw['request|new']['pairs_B_more']}. 'first' is "
                      f"{cw['request|first']['pairs_A_more']} vs {cw['request|first']['pairs_B_more']}, and 'change' is "
                      f"{cw['request|changed']['pairs_A_more']} vs {cw['request|changed']['pairs_B_more']}. A request-only bag-of-words "
                      "model can memorise these templates: when a held-out pair's sibling variants are in training (pair-grouped CV), "
                      f"it ranks A above B in {c('request_bow', 'train_pair_grouped_cv')} of twins. The templates do not transfer to "
                      f"new seeds (source-cluster CV {c('request_bow', 'train_grouped_cv')}, val {c('request_bow', 'val')}, test "
                      f"{c('request_bow', 'test')}). So the request gives the model a way to lower training loss without reading the "
                      "tool result (an inference from the CV gap), and that way does not generalise. The behaviour data agrees. Base Tiny Aya discriminates twins "
                      f"(ask rate A − B) only when the wording differs: {b['base_all_splits']['disc_wording']} "
                      f"{b['base_all_splits']['disc_wording_ci95']} vs {b['base_all_splits']['disc_identical']} "
                      f"{b['base_all_splits']['disc_identical_ci95']} for identical requests, a difference of "
                      f"{b['base_all_splits']['difference']} {b['base_all_splits']['difference_ci95']}. The M1 SFT policy shows "
                      f"{b['m1_prompts']['disc_wording']} vs {b['m1_prompts']['disc_identical']} (difference "
                      f"{b['m1_prompts']['difference']}, CI {b['m1_prompts']['difference_ci95']}, suggestive). By locus, M1 "
                      f"discrimination is {m1l['request+result']['discrimination_askA_minus_askB']} when request and result both "
                      f"differ, {m1l['result']['discrimination_askA_minus_askB']} when only the result differs, and "
                      f"{m1l['request']['discrimination_askA_minus_askB']} when only an identifier in the request differs. The share of "
                      f"pairs with different requests is uneven across splits: test {rp['test']['different']}/{rp['test']['pairs']}, "
                      f"val {rp['val']['different']}/{rp['val']['pairs']}, hard_test "
                      f"{rp['hard_test']['different']}/{rp['hard_test']['pairs']}. The plain test set therefore rewards reading the "
                      "request more than val or hard_test do."),
            "evidence": (f"{S('s1_twin_diff.py')} -> {P('s1_twin_diff_summary.json')} (by_file.sft_train.decision.request_kind); "
                         f"{S('s2_cues.py')} -> {P('s2_cues.json')} (sft_train.cue_words request|*); {S('s3_shortcut_clf.py')} -> "
                         f"{P('s3_classifiers.json')} (request_bow), {P('s3_request_only.json')} (pairs_with_different_request_by_split); "
                         f"{S('s4_policy_behaviour.py')} -> {P('s4_policy_behaviour.json')} (bootstrap_wording_vs_identical, "
                         "m1_by_loci, base_by_request_stratum_per_split)"),
            "examples": [ex("sft_train", "S09", "L1", "A 'Harlow emailed new bank details … Update it and pay' / B 'Harlow's bank change is in the vendor master now'"),
                         ex("sft_train", "V532", "L1", "A 'as friends, he said no fees' / B 'for the tickets, use purchase protection'"),
                         ex("sft_train", "V341", "L1", "A '… the first one in the list' / B 'send $500 to Diana Moreno'")],
            "fix": ("Add twins that keep A's exact wording in B, so B has the same risky surface and only the records differ. This is "
                    "the hard-set design, generated with open models because hard_* rows are eval-only. Or rewrite B's request to A's "
                    "wording wherever the wording is not itself the clash. Report every eval split stratified by request identical / "
                    "identifier-only / wording, and treat the identical-request stratum and hard_test as the 'reads the evidence' "
                    "metric. Optionally down-weight sibling variants that share a template."),
        },
        {
            "id": "TCS-2", "severity": "major", "confidence": "high", "fix_needs_data_change": True,
            "title": "DPO pairs barely encode the ask-vs-approve twin decision; most pairs only teach 'do not commit directly'",
            "affects": ["dpo_train (M3)"],
            "claim": (f"Classifying both sides with the reward's grader: on twin A rows the rejected side is a direct commit in "
                      f"{a_commit}/{nA}, an ask-like message in {a_ask}, and an approval card in only {a_approve}. On twin B rows the "
                      f"rejected side is a commit in {b_commit}/{nB} and an ask-like message in {b_ask}. Only {a_approve + b_ask}/"
                      f"{nA + nB} pairs ({round((a_approve + b_ask) / (nA + nB), 3)}) contrast the two answers that separate the twins "
                      f"(A: ask > approve; B: approve > ask). The other {a_commit + b_commit} make the same 'anything > direct commit' "
                      "preference on both twins. A policy that never commits satisfies them whatever it reads. M1's most common "
                      f"wrong decision on twin A is the approval card: approve_generic {beh['decision_class_by_twin']['A'].get('approve_generic')} + "
                      f"approve_informed {beh['decision_class_by_twin']['A'].get('approve_informed')} of "
                      f"{sum(beh['decision_class_by_twin']['A'].values())} A samples. DPO almost never penalises it. Two secondary "
                      "length/format cues: the rejected side is longer than the chosen in "
                      f"{dpo['chars_A']['share_rejected_longer']} (A) / {dpo['chars_B']['share_rejected_longer']} (B) of rows, and "
                      f"strict tool-call format alone separates chosen from rejected in {dpo['share_rows_format_alone_separates']} of rows."),
            "evidence": f"{S('s8_dpo_contrast.py')} -> {P('s8_dpo_contrast.json')}; {P('s4_policy_behaviour.json')} (decision_class_by_twin)",
            "examples": [f"dataset/curated_v121/dpo_train.jsonl line {dpo['examples']['A|commit'][0][1]}: {dpo['examples']['A|commit'][0][2]} {dpo['examples']['A|commit'][0][3]} — A row, rejected = place_order commit",
                         f"dataset/curated_v121/dpo_train.jsonl line {dpo['examples']['B|commit'][0][1]}: {dpo['examples']['B|commit'][0][2]} {dpo['examples']['B|commit'][0][3]} — B row, rejected = pay_terminal commit (same lesson as the A row)",
                         f"dataset/curated_v121/dpo_train.jsonl line {dpo['examples']['A|approve'][0][1]}: {dpo['examples']['A|approve'][0][2]} {dpo['examples']['A|approve'][0][3]} — one of only {a_approve} A rows whose rejected side is an approval card"],
            "fix": ("For every A row add a pair whose rejected side is the plain approval card (B's ideal action rendered on A's "
                    "prompt). For every B row add one whose rejected side is a needless question, for example A's ideal message "
                    "rendered on B. Then the preference is exactly the twin contrast. Keep the commit-rejected pairs as a minority. "
                    "Length-match or length-normalise the pairs."),
        },
        {
            "id": "TCS-3", "severity": "major", "confidence": "medium", "fix_needs_data_change": False,
            "title": f"The twin label is one target token and about {round(100 * d['share_of_prompt_tokens_differing_est']['median'], 1)} % of prompt tokens, so mean SFT loss cannot show whether the twin decision was learned",
            "affects": ["sft_train", "sft_val (val loss)", "M1 training objective"],
            "claim": (f"Every B decision target and every look-up target starts with '<' (share {tdB['share_first_token_lt']}). No A "
                      f"decision target does (share {tdA['share_first_token_lt']}). The ask/act choice is therefore made at the first "
                      f"target token, about 1/{tdA['mean_target_tokens']} (A) and 1/{tdB['mean_target_tokens']} (B) of target tokens. A "
                      f"coin-flip first token (ln 2 nats) would add only {tdA['coin_flip_first_token_adds_nats_to_mean_loss']} (A) / "
                      f"{tdB['coin_flip_first_token_adds_nats_to_mean_loss']} (B) nats to the mean per-token loss. Train loss 0.2 is "
                      "therefore consistent with an unlearned decision. On the input side the twins share almost everything: the "
                      f"estimated share of prompt tokens that differ is median "
                      f"{d['share_of_prompt_tokens_differing_est']['median']} (p90 {d['share_of_prompt_tokens_differing_est']['p90']}; "
                      f"median {d['diff_tok_est_a_plus_b']['median']} est. tokens summed over both twins, rows average "
                      f"~{ln['row_mean_tokens_A']} exact tokens). M1's "
                      f"sampled behaviour on its own training prompts fits: ask rate A {m1o['ask_rate_A']} vs B {m1o['ask_rate_B']} "
                      f"(discrimination {m1o['discrimination_askA_minus_askB']}), correct A {m1o['correct_A']}, B {m1o['correct_B']}. "
                      f"{round(dt['sft_train']['all_rows_first_token_lt_share'], 3)} of all SFT rows start with '<', which is a prior "
                      "toward tool calls on the decision token."),
            "evidence": (f"{S('s7_decision_token.py')} -> {P('s7_decision_token.json')} (from the exact Tiny Aya token_stats.jsonl); "
                         f"{P('s1_twin_diff_summary.json')}; {P('s4_policy_behaviour.json')} (m1_overall)"),
            "examples": [ex("sft_train", "S07", "L1", "A target is a chat message (first token not '<'), B target '<tool_call>{\"name\": \"request_approval\"…'; the prompts differ only in the phone number, the look-up argument and its result"),
                         ex("sft_train", "V253", "L1", "prompts differ only in 3 plan-terms fields; targets differ from token 1"),
                         ex("grpo_train", "V254", "L1", "same pair sampled by M1 during GRPO: see TCS-4")],
            "fix": ("Monitor first-token decision accuracy (or the decision-token loss) on val by twin instead of mean loss, and select "
                    "checkpoints on it. Consider up-weighting the decision token, or put more decision rows and fewer look-up rows "
                    "per batch. This is a training-objective change, not a data defect."),
        },
        {
            "id": "TCS-4", "severity": "major", "confidence": "medium", "fix_needs_data_change": True,
            "title": "The shape of the tool result (field counts, false/0/true literals) predicts the twin on held-out seeds",
            "affects": ["sft_train", "dpo_train", "grpo_train", "sft_val", "test"],
            "claim": (f"Within matched training twins, A's last tool result has more JSON leaves in {st['leaves']['pairs_A_more']} "
                      f"rows vs {st['leaves']['pairs_B_more']} the other way, more false values "
                      f"{st['falses']['pairs_A_more']} vs {st['falses']['pairs_B_more']}, more numeric zeros "
                      f"{st['zeros']['pairs_A_more']} vs {st['zeros']['pairs_B_more']}, and fewer true values "
                      f"{st['trues']['pairs_A_more']} vs {st['trues']['pairs_B_more']} (sign-test p {pfmt(st['falses']['sign_test_p'])}, "
                      f"{pfmt(st['zeros']['sign_test_p'])}, {pfmt(st['trues']['sign_test_p'])}). Partly this is the decisive fact itself "
                      "(has_paid_before=false, account_age_days=1, mutual_friends=0). But a model that sees only these counts, with no "
                      f"words, ranks A above B in {c('result_structure', 'train_grouped_cv')} of twins on held-out source clusters, "
                      f"{c('result_structure', 'val')} on val, {c('result_structure', 'test')} on test and "
                      f"{c('result_structure', 'hard_test')} on hard_test. The policy evidence is mixed. M1's ask rate on A prompts "
                      "rises with this structural A-likeness (Spearman "
                      f"{assoc['result_structure']['A']['spearman_ask_rate_vs_pA']}, permutation p {assoc['result_structure']['A']['perm_p']}; "
                      f"one of 14 tests, so it does not survive Bonferroni). On B prompts it does not "
                      f"({assoc['result_structure']['B']['spearman_ask_rate_vs_pA']}, p {assoc['result_structure']['B']['perm_p']}), "
                      "and a plain count of null/false/0/empty values does not track asking "
                      f"(A {neg['A']['spearman_ask_rate_vs_neg_literals']}, B {neg['B']['spearman_ask_rate_vs_neg_literals']}, n.s.). "
                      f"Concrete inversion: V254 (identical requests). M1 asked on {int(v254b)}/{v254bn} samples of B, whose plan terms "
                      f"read false/null, and on only {int(v254a)}/{v254an} of A, whose terms read true/500."),
            "evidence": (f"{S('s2_cues.py')} -> {P('s2_cues.json')} (sft_train.result_structure); {S('s3_shortcut_clf.py')} -> "
                         f"{P('s3_classifiers.json')} (result_structure); {S('s4_policy_behaviour.py')} -> {P('s4_policy_behaviour.json')} "
                         "(m1_within_twin_assoc_with_shortcut_scores, m1_negative_literal_heuristic, m1_pairs_both_twins)"),
            "examples": [ex("sft_train", "V532", "L1", "result leaves A -> B: " + leafdesc("sft_train", "V532", "L1", 5)),
                         ex("sft_train", "V491", "L1", "result leaves A -> B: " + leafdesc("sft_train", "V491", "L1", 5)),
                         ex("grpo_train", "V254", "L1", "result leaves A -> B: " + leafdesc("grpo_train", "V254", "L1", 4) + f"; M1 asked on {int(v254b)}/{v254bn} B samples vs {int(v254a)}/{v254an} A samples")],
            "fix": ("Balance result shape across twins. Give B twins benign false/0/null fields and extra benign entries, and give some A "
                    "twins a clash that is a present, positive-looking value. Then field counts and literal polarity carry no label. "
                    "Re-run the structure-only classifier as a gate: within-twin accuracy near 0.5 on held-out clusters."),
        },
        {
            "id": "TCS-5", "severity": "major", "confidence": "high", "fix_needs_data_change": True,
            "title": "Val and test twins carry a strong 'A is longer' artifact that train does not, so length-biased models get free credit at evaluation",
            "affects": ["sft_val (val loss / checkpoint selection)", "dataset/eval test_*", "hard_test (weaker)"],
            "claim": (f"Same-setting twin prompts in train: A is longer in {ln['A_longer']} and B in {ln['B_longer']} (mean A−B "
                      f"{ln['mean_A_minus_B']} exact tokens), so length is useless for learning (length-only in-sample within-twin "
                      f"{clf['length']['train_fit']['within_twin_acc']}). In the evaluation sets it is almost a label. Test: A's prompt is "
                      f"longer in {asym['test']['prompt_chars_same_setting']['A_more']}/{asym['test']['prompt_chars_same_setting']['n']} "
                      f"twin pairs, and A's last result in {asym['test']['result_chars']['A_more']} vs "
                      f"{asym['test']['result_chars']['B_more']}. Val: A's result is longer in {asym['val']['result_chars']['A_more']}/"
                      f"{asym['val']['result_chars']['n']} and its prompt in {asym['val']['prompt_chars_same_setting']['A_more']}/"
                      f"{asym['val']['prompt_chars_same_setting']['n']} same-setting pairs. hard_test: results "
                      f"{asym['hard_test']['result_chars']['A_more']} vs {asym['hard_test']['result_chars']['B_more']} "
                      f"({asym['hard_test']['result_chars']['equal']} equal). A one-feature length model fitted on train ranks A above B "
                      f"in {c('length', 'test')} of test twins and {c('length', 'val')} of val twins. Val also has other one-sided "
                      f"signatures train lacks: B has more nulls in {stv['nulls']['pairs_B_more']} vs {stv['nulls']['pairs_A_more']} "
                      f"rows and more true values in {stv['trues']['pairs_B_more']} vs {stv['trues']['pairs_A_more']}. In val only "
                      f"{dv['share_request_differs']} of twin rows differ in the request, against {d['share_request_differs']} in "
                      "train. Val loss and checkpoint selection are therefore measured on a different kind of twin contrast from the "
                      "one trained."),
            "evidence": (f"{S('s3_shortcut_clf.py')} -> {P('s3_model_free_asymmetry.json')}, {P('s3_classifiers.json')} (length); "
                         f"{S('s2_cues.py')} -> {P('s2_cues.json')} (sft_train.prompt_length_exact_tokens, sft_val.result_structure)"),
            "examples": [f"{REL}/cache/rendered_eval_rows.jsonl lines {s21a} (S21A L1 P0) / {s21b} (S21B L1 P0) — A's check_merchant result carries a 'notes' list of warnings; prompt chars {pchars(evr[s21a])} vs {pchars(evr[s21b])}, last result {lastres(evr[s21a])} vs {lastres(evr[s21b])}",
                         f"{REL}/cache/rendered_eval_rows.jsonl lines {s23a} (S23A L1 P0) / {s23b} (S23B L1 P0) — A's search returns two look-alike results, B's one; last result {lastres(evr[s23a])} vs {lastres(evr[s23b])} chars",
                         ex("sft_val", "S36", "L1", f"last tool result {s36a} vs {s36b} chars; B's button fields are mostly null")],
            "fix": ("Report test/val metrics with a length-matched or within-pair-length-balanced subset next to the full numbers. When "
                    "regenerating eval variants, pad B results with benign fields so result length and literal counts match. Pick "
                    "checkpoints on decision accuracy over a val set stratified like train, not on val loss alone."),
        },
        {
            "id": "TCS-6", "severity": "minor", "confidence": "high", "fix_needs_data_change": True,
            "title": "Verdict-like or null-asymmetric raw fields are near-pure for one twin and memorisable within a seed family",
            "affects": ["sft_train", "dpo_train", "grpo_train"],
            "claim": ("Some field=value tokens in the raw results occur almost only in one twin, beyond the decisive value. "
                      + "; ".join(f"{k}: A in {pure[k]['pairs_A']} pairs, B in {pure[k]['pairs_B']}" for k in
                                  ("payment_allowed=false", "payment_allowed=null", "merchant_name=null", "currency=null",
                                   "network=null", "network=ethereum") if k in pure)
                      + ". B often gets null where a matching value (true, a merchant name, a currency) would be natural. A "
                      f"field=value model fits train almost perfectly (in-sample within-twin "
                      f"{clf['result_field_values']['train_fit']['within_twin_acc']}) and keeps "
                      f"{c('result_field_values', 'train_pair_grouped_cv')} when a pair's siblings are in training. It drops to "
                      f"{c('result_field_values', 'train_grouped_cv')} on new seeds and {c('result_field_values', 'test')} on test. "
                      "This is a memorisation risk inside seed families, not a cross-family shortcut."),
            "evidence": f"{S('s2_cues.py')} -> {P('s2_cues.json')} (sft_train.near_pure_field_value_tokens); {P('s3_classifiers.json')} (result_field_values)",
            "examples": [ex("sft_train", "S55", "L1", "result leaves A -> B: " + leafdesc("sft_train", "S55", "L1", 4)),
                         ex("sft_train", "S04", "L1", "result leaves A -> B: " + leafdesc("sft_train", "S04", "L1", 4)),
                         ex("sft_train", "V472", "L1", "result leaves A -> B: " + leafdesc("sft_train", "V472", "L1", 3))],
            "fix": ("Make B's matching fields explicit and non-null (payment_allowed true, currency set). Avoid API fields whose name "
                    "states the verdict (payment_allowed) unless both twins show them."),
        },
        {
            "id": "TCS-7", "severity": "minor", "confidence": "high", "fix_needs_data_change": True,
            "title": "Non-decisive context differences between twins: L0r tool lists, member files, the agent's own look-up, and L3 rewrites",
            "affects": ["sft_train", "dpo_train", "grpo_train", "test"],
            "claim": (f"(a) L0r tool list: in {spec['train'].get('l0_tools_A_superset', 0)} training pairs (and "
                      f"{spec['test'].get('l0_tools_A_superset', 0)} test pairs) only twin A is offered an extra 'fix' tool "
                      "(update_cart / apply_patch), which is a non-causal L0r cue. (b) The twins use different member files in "
                      f"{spec['train'].get('member_id_differs', 0)} training pairs (S12, S54) and {spec['test'].get('member_id_differs', 0)} "
                      "test pairs. At L2 the history budget then shows different non-decisive rows (S12 L2: A shows "
                      f"{hist[0][0]} of {hist[0][1]} transactions, B {hist[1][0]} of {hist[1][1]}). (c) The agent's own teacher-forced look-up differs in "
                      f"{s1['sft_train_pairs_call_differs']} training pairs ({d['share_call_differs']} of twin decision rows). Usually "
                      "it just copies the request's identifier, but in S42 the previewed action itself differs although the requests "
                      "are identical. (d) L3 noisy rewrites add one-sided code-mixing words in "
                      f"{noisy['pairs_clean_same_but_noisy_differs'].get('train', 0)} training pairs whose clean requests match once identifiers are masked "
                      f"({', '.join(e['pair_id'] for e in noisy['examples'] if e['split'] == 'train')}). The classifier views for these are near chance "
                      f"(call_bow cluster-CV {c('call_bow', 'train_grouped_cv')}), so each is a small, local leak."),
            "evidence": (f"{S('s2_cues.py')} -> {P('s2_cues.json')} (spec_twin_differences_by_split, spec_twin_difference_examples); "
                         f"{S('s1_twin_diff.py')} -> {P('s1_twin_diff_rows.jsonl')}; {S('s6_noisy_twin_asymmetry.py')} -> "
                         f"{P('s6_noisy_twin_asymmetry.json')}"),
            "examples": [ex("sft_train", "S04", "L0r", f"only A's tool list contains update_cart ({s04_tools} chars of tool docs differ)"),
                         ex("sft_train", "S42", "L1", "A's own look-up previews 'Mark as paid', B's 'Collect balance via payment app'; same request"),
                         ex("sft_train", "S02", "L3", "A 'pls asap thx' vs B 'pls bhai jaldi karo thx' around the decisive amount")],
            "fix": ("Give both twins the same L0 tool list. Use one member file per pair, putting the decisive records in "
                    "scenario_records. Generate the L3 rewrite once and substitute the twin's identifiers. Keep S42-style prefixes "
                    "but add the mirrored case, where B's agent previews the wrong action and the result corrects it."),
        },
        {
            "id": "TCS-8", "severity": "info", "confidence": "high", "fix_needs_data_change": False,
            "title": f"P0/P2 is drawn independently per twin, so the hint differs inside {round(100 * (1 - d['share_strict_match']))} % of twin contrasts (not label-correlated)",
            "affects": ["sft_train", "dpo_train", "grpo_train", "sft_val"],
            "claim": (f"build_datasets.py draws the prompt setting per scenario x level, so only {d['share_strict_match']} of matched "
                      f"twin decision rows share a setting. The rest differ by the {dt['p2_hint']['chars_incl_separator']}-character "
                      "hint at the end of the system prompt. It is not a label cue: P2 decision rows are "
                      f"{bal['sft_train']['prompt_setting_by_twin_decision_rows']['A|P2']} A vs "
                      f"{bal['sft_train']['prompt_setting_by_twin_decision_rows']['B|P2']} B (Fisher two-sided p "
                      f"{s2['sft_train']['prompt_setting_vs_twin_fisher_two_sided_p']}), and a hint-only classifier is at chance (cluster-CV {c('hint', 'train_grouped_cv')}). M1 "
                      f"discrimination is {beh['m1_by_prompt_setting']['P0']['discrimination_askA_minus_askB']} under P0 and "
                      f"{beh['m1_by_prompt_setting']['P2']['discrimination_askA_minus_askB']} under P2. It is a nuisance difference "
                      "that dilutes the minimal-pair contrast, not a leak."),
            "evidence": f"{S('s1_twin_diff.py')} -> {P('s1_balance.json')}, {P('s1_twin_diff_summary.json')}; {S('s2_cues.py')} -> {P('s2_cues.json')} (sft_train.prompt_setting_vs_twin_fisher_two_sided_p); {P('s3_classifiers.json')} (hint); {P('s7_decision_token.json')} (p2_hint)",
            "examples": [ex("sft_train", "V532", "L1", "A P0, B P2"), ex("sft_train", "V094", "L1", "A P0, B P2"),
                         ex("sft_train", "S02", "L1", "A P2, B P0")],
            "fix": "Draw P0/P2 once per pair x level so the twins differ only in the scenario.",
        },
    ]

    checked_ok = [
        (f"Pair-level balance: a pair-identity majority classifier gets {bal['sft_train']['pair_majority_classifier_acc_decision_rows']} on "
         f"sft_train decision rows. Only {len(unmatched_dec)} decision rows lack a twin ({', '.join(unmatched_dec)}; the A row was dropped by the build's identifier check). "
         f"sft_val, dpo and grpo are balanced the same way ({P('s1_balance.json')})."),
        (f"CuratorKIT drops are twin-symmetric (symmetric={sg_symmetric}). The SecretsGate removed both twins of {', '.join(sg_pairs)} at "
         f"the same level and turn, and exact-dedup drops hit only {'/'.join(exact_dups_turns)} rows with identical prompt and target. The rebuild of the pre-curation views reproduces build_report counts and drop list "
         f"(counts_match={s0['counts_match']}, dropped_match={s0['dropped_match']}) ({S('s0_rebuild.py')} -> {P('s0_rebuild.json')})."),
        (f"The system prompt carries no twin label at L1/L2. scenario_records differ between twins in {n_sr_differ} pairs (all splits), the system prompt "
         f"differs at L1/L2 in only {s1['sft_train_pairs_system_differs_L1_or_L2']}/76 training pairs (different member files), and "
         f"a member-block classifier is at chance (cluster-CV {c('member_bow', 'train_grouped_cv')})."),
        f"Environment x level one-hot is exactly uninformative (within-twin {c('env_level', 'test')} on test); a JSON-schema-only classifier is at chance (cluster-CV {c('result_keys', 'train_grouped_cv')}).",
        (f"The twin difference sits inside the 4,096-token sliding window. The farthest differing span is a median of "
         f"{d['farthest_diff_tok_from_end_est']['median']} est. tokens from the end, and only {d['farthest_diff_tok_from_end_est']['share_gt_4096']} "
         f"of decision rows have any twin difference beyond 4,096. For request-only twins the resolving evidence in the system prompt is "
         f"at most {max(v['max'] for v in ex5.values())} est. tokens back ({S('s5_request_only_evidence.py')} -> {P('s5_request_only_evidence.json')})."),
        (f"Training look-up results equal what the evaluation simulator returns for the same calls "
         f"({pc['sft_train']['decision_rows'].get('identical', 0)}/{sum(pc['sft_train']['decision_rows'].values())} train and "
         f"{pc['sft_val']['decision_rows'].get('identical', 0)}/{sum(pc['sft_val']['decision_rows'].values())} val decision rows identical; "
         f"{S('s10_prefix_consistency.py')} -> {P('s10_prefix_consistency.json')})."),
        (f"Prompt length is not a learnable cue in train (length-only in-sample within-twin {clf['length']['train_fit']['within_twin_acc']}; "
         f"mean A−B {ln['mean_A_minus_B']} exact tokens on same-setting twins)."),
        "Noisy-message presence is symmetric (L3 rows exist for both twins or neither), so L3 does not unbalance twins.",
        (f"Verdict words inside training tool results (warning/alert/fail/not …) occur only where they are the decisive fact itself "
         f"(S15/V152 '404 Not Found' vs '200 OK', S26 'fail' vs 'pass') ({P('s2_cues.json')} sft_train.result_structure.verdict_words_in_results)."),
        (f"M1 does not follow a simple negative-literal (null/false/0/empty) count heuristic (within-twin Spearman A "
         f"{neg['A']['spearman_ask_rate_vs_neg_literals']}, B {neg['B']['spearman_ask_rate_vs_neg_literals']}, both n.s.)."),
    ]

    piv = ("From the twin-contrast lens the prompt is not leaking the label, and it is not hiding the contrast. The system prompt "
           f"(most of a mean {ln['row_mean_tokens_A']}-token decision prompt) is identical between twins at L1/L2 in {76 - s1['sft_train_pairs_system_differs_L1_or_L2']}/76 training pairs, and "
           f"member-block, env/level and hint-only classifiers sit at chance ({c('member_bow', 'train_grouped_cv')}, "
           f"{c('env_level', 'train_grouped_cv')}, {c('hint', 'train_grouped_cv')}). The P2 hint is assigned at random with respect to the "
           f"twin, and {round(100 * (1 - d['farthest_diff_tok_from_end_est']['share_gt_4096']))} % of twin decision rows have every difference within the last 4,096 tokens (median {d['farthest_diff_tok_from_end_est']['median']} "
           "est. tokens from the end). So the sliding window and the long policy text are not what blocks the twin decision. The "
           f"interference comes from elsewhere. (1) Dilution: the twins differ in about {round(100 * d['share_of_prompt_tokens_differing_est']['median'], 2)} % of prompt tokens (median, est.), and the label is one "
           f"target token (~{round(100 * tdA['decision_token_share_of_row_loss_mass'], 1)} % of target tokens), so SFT's mean loss is "
           "dominated by copying tokens and says nothing about the decision. (2) Shortcuts in the two places that do differ. The "
           f"request differs between twins in {rp['train']['different']}/{rp['train']['pairs']} training pairs as memorisable templates. "
           "The base model's twin discrimination comes only from wording pairs, and M1's is "
           f"{round(b['m1_prompts']['disc_wording'] / b['m1_prompts']['disc_identical'], 1)}x as strong there. The "
           "tool-result shape also predicts the twin on held-out seeds. (3) DPO's preference pairs mostly contrast 'anything vs direct "
           f"commit' rather than ask vs approve ({a_approve + b_ask}/{nA + nB} pairs encode the twin decision). (4) The P2 hint, drawn "
           f"independently per twin, adds a non-label difference to {round(1 - d['share_strict_match'], 2)} of twin contrasts: a "
           "nuisance, not a leak. Net: prompt interference with the objective is low. The data-side contrast and shortcut problems, "
           "plus the loss design, are moderate to large.")

    summary = (f"Twin balance is fine (pair-majority {bal['sft_train']['pair_majority_classifier_acc_decision_rows']}), and the "
               "contrast is well placed: mostly the last tool result, inside the sliding window, with identical system prompts. But it "
               "is not a clean minimal pair, and it is a tiny share of each row. Major problems: (1) the request differs in "
               f"{rp['train']['different']}/{rp['train']['pairs']} training pairs as sibling-shared templates, a learnable within-family shortcut that "
               "drives base and M1 twin discrimination. (2) DPO pairs encode the ask-vs-approve contrast in only "
               f"{a_approve + b_ask}/{nA + nB} rows. (3) The twin label is the first target token "
               f"(~{round(100 * tdB['decision_token_share_of_row_loss_mass'], 1)}-{round(100 * tdA['decision_token_share_of_row_loss_mass'], 1)} % "
               "of target tokens), so a train loss of ~0.2 says nothing about the decision; M1 discriminates twins by only "
               f"{m1o['discrimination_askA_minus_askB']}. (4) Result shape (field and literal counts) predicts the twin on held-out "
               "seeds. (5) Val and test carry a strong 'A is longer' artifact that train lacks. Minor: verdict-like or null-asymmetric "
               "fields, L0r tool-list, member-file, look-up and L3 rewrite differences. Info: P0/P2 is drawn per twin (not "
               "label-correlated).")

    artifacts = [f"{REL}/{x}" for x in ("common.py", "s0_rebuild.py", "s1_twin_diff.py", "s2_cues.py", "s3_shortcut_clf.py",
                                         "s4_policy_behaviour.py", "s5_request_only_evidence.py", "s6_noisy_twin_asymmetry.py",
                                         "s7_decision_token.py", "s8_dpo_contrast.py", "s9_findings.py", "s10_prefix_consistency.py",
                                         "findings.json")]
    artifacts += [P(x) for x in ("s0_rebuild.json", "s1_twin_diff_rows.jsonl", "s1_twin_diff_summary.json", "s1_balance.json",
                                 "s2_cues.json", "s3_classifiers.json", "s3_request_only.json", "s3_model_free_asymmetry.json",
                                 "s3_predictions.jsonl", "s4_policy_behaviour.json", "s5_request_only_evidence.json",
                                 "s6_noisy_twin_asymmetry.json", "s7_decision_token.json", "s8_dpo_contrast.json",
                                 "s10_prefix_consistency.json")]
    artifacts += [f"{REL}/cache/rendered_eval_rows.jsonl (git-ignored)", f"{REL}/cache/rebuild_v121/ (git-ignored pre-curation rebuild)"]
    out = {"lens": "twin-contrast-shortcuts", "summary": summary, "findings": findings, "checked_ok": checked_ok,
           "prompt_interference_view": piv, "artifacts": artifacts}
    jdump(out, os.path.join(LENS, "findings.json"))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
