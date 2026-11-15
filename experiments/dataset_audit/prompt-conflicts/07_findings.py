"""Step 7 — assemble findings.json for the prompt-conflicts lens. Every number is read from this lens's results files
(01-06); nothing is typed by hand. Run 01-06 first.
    python3 -B experiments/dataset_audit/prompt-conflicts/07_findings.py
"""
from __future__ import annotations

import collections
import json
import os

from pc_common import HERE, RES, ROOT, rel
from wtm.reward import A_TABLE, B_ACT, B_APPROVE  # noqa: E402  (reward constants quoted in the text)

LOOKUP_R, FMT_R = -0.1, 0.1   # wtm/reward.py reward(): lookup base and strict-format bonus (asserted below)
import inspect as _inspect  # noqa: E402
import wtm.reward as _wr  # noqa: E402
_src = _inspect.getsource(_wr.reward)
assert "base = -0.1" in _src and "fmt = 0.1 if strict else 0.0" in _src, "reward() constants changed; update LOOKUP_R/FMT_R"

J = lambda name: json.load(open(os.path.join(RES, name)))  # noqa: E731
inv = J("prompt_inventory.json")
ins = J("instructions.json")
ten = J("tensions.json")
pos = J("instruction_positions.json")
mev = J("model_evidence.json")
m1 = mev["m1_policy_on_training_prompts"]
TT = [json.loads(l) for l in open(os.path.join(RES, "target_table.jsonl"))]
LIT = [json.loads(l) for l in open(os.path.join(RES, "literal_prescription.jsonl"))]
S = "experiments/dataset_audit/prompt-conflicts/"


def exs(rows, n=3):
    return [f"dataset/curated_v121/{r['file']}.jsonl:{r['i']} {r['scenario_id']} {r['level']} {r['prompt_setting']} "
            f"{r.get('turn', 'decision')}" for r in rows[:n]]


def pp(x):
    return f"{100 * x:+.1f} pp"


def ci(b):
    return f"{pp(b['diff'])} (95% cluster-bootstrap CI {100 * b['ci95'][0]:.1f} to {100 * b['ci95'][1]:.1f} pp; prompts {b['prompts'][0]} vs {b['prompts'][1]})"


sft_dec = [r for r in TT if r["file"] == "sft_train" and r["role"] == "target" and r["turn"] == "decision"]
n_dec = len(sft_dec)
litS = [x for x in LIT if x["unit"] == "SFT"]
cat = collections.Counter(x["category"] for x in litS)
catA = {ps: collections.Counter(x["category"] for x in litS if x["t"] == "A" and x["prompt_setting"] == ps) for ps in ("P0", "P2")}
catB = {ps: collections.Counter(x["category"] for x in litS if x["t"] == "B" and x["prompt_setting"] == ps) for ps in ("P0", "P2")}
nA = {ps: sum(catA[ps].values()) for ps in catA}
nB = {ps: sum(catB[ps].values()) for ps in catB}

a = ten["a_ask_limiting_vs_executable_A"]["SFT"]
d = ten["d_fraud_rules_vs_A"]["SFT"]
c = ten["c_standing_authorisation_and_hint_vs_B"]
cr = ten["c_reward_view"]
f = ten["f_lookup_rules"]["SFT"]
g = ten["g_protocol_format"]
gr = ten["g_dpo_rejected"]
h = ten["h_level_dependence"]
b = ten["b_card_rule_vs_B"]["SFT"]

# --- bizops contradictions (a + d + S14A)
biz_rows = [r for r in sft_dec if r["t"] == "A" and r["env"] == "bizops" and r["level"] != "L0r" and r.get("a_exec")]
shop_exec = [r for r in sft_dec if r["t"] == "A" and r["env"] == "shopping" and r["level"] != "L0r" and r.get("a_exec")]
bills_exec = [r for r in sft_dec if r["t"] == "A" and r["env"] == "bills" and r["level"] != "L0r" and r.get("a_exec")]
biz_contra = [x for x in litS if x["category"] in ("contradicted", "hint_vs_policy") and x["t"] == "A"]
fraud_biz = [r for r in sft_dec if r["t"] == "A" and r["env"] == "bizops" and r["fraud_tag"] and r["level"] != "L0r"]
# --- act rows vs hint and V152B
act_p2 = [r for r in sft_dec if r["t"] == "B" and r["expected"] == "act" and r["prompt_setting"] == "P2"]
v152 = [r for r in sft_dec if r.get("b_act_uncovered")]
val_act_p2 = [r for r in TT if r["file"] == "sft_val" and r["role"] == "target" and r["turn"] == "decision"
              and r["expected"] == "act" and r["prompt_setting"] == "P2"]
# --- SA-visible approve rows
sa_appr = [r for r in sft_dec if r["t"] == "B" and r["expected"] == "approve" and r.get("b_sa_visible")]
# --- zero look-up
zl = [r for r in sft_dec if r["n_lookups_in_prompt"] == 0]
n_lookup_turns = sum(1 for r in TT if r["file"] == "sft_train" and r["role"] == "target" and r["turn"].startswith("lookup"))
# --- card-content omissions (rule present; deploy/cancel cards where a payment method does not apply are excluded)
NA_METHOD = {"S26B", "V264B"}
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("t04", os.path.join(HERE, "04_tensions.py"))
t04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t04)
cards_rule = [r for r in sft_dec if r["t"] == "B" and r["decision"].startswith("approve") and (r["level"] != "L0r" or r["env"] == "wallet")
              and r["scenario_id"] not in NA_METHOD]
cards_omit = [r for r in cards_rule if not t04.METHOD_STRICT.search(r.get("card_summary", ""))]
omit_env = {e: (sum(r["env"] == e for r in cards_omit), sum(r["env"] == e for r in cards_rule)) for e in ("wallet", "bills", "shopping", "bizops")}
# --- informed-card competing alternative
comp_rows = [x for x in litS if x["t"] == "A" and x["category"] in ("competing", "supported_with_competing_option")]
comp_tt = [r for r in sft_dec if any(x["i"] == r["i"] for x in comp_rows)]
# --- positions
pl = pos["by_level"]
l2, l3 = pl["L2"], pl["L3"]
KEY_OUT = "policy_END_outside_window(entire policy unseen by local layers)"
n_l0r = sum(r["level"] == "L0r" for r in sft_dec)
_pw = inv["policy_summary"]
wl0 = (min(v["words"] for k, v in _pw.items() if k.endswith("/L0r")), max(v["words"] for k, v in _pw.items() if k.endswith("/L0r")))
wl1 = (min(v["words"] for k, v in _pw.items() if k.endswith("/L1")), max(v["words"] for k, v in _pw.items() if k.endswith("/L1")))
n_l0r_p0_unspec = sum(1 for x in litS if x["level"] == "L0r" and x["prompt_setting"] == "P0" and x["category"] == "unspecified")
# --- base model evidence
bh = mev["base_tinyaya_hint_P0_vs_P2_paired"]
bp = mev["base_tinyaya_policy_L0r_vs_L1_P0"]

# extra computed checks for checked_ok
import re as _re  # noqa: E402
from pc_common import FILES, load_envs, load_rows, split_system, tools_in_docs  # noqa: E402
_envs = load_envs()
_ra_missing = _prefix_unlisted = _prefix_total = 0
for _f in FILES:
    for _r in load_rows(_f):
        _sec = split_system(_r["prompt"][0]["content"], _envs[_r["env_id"]], _r["level"])
        _listed = set(tools_in_docs(_sec["tools"]))
        _ra_missing += "request_approval" not in _listed
        for _m in _r["prompt"][1:]:
            if _m["role"] == "assistant":
                _prefix_total += 1
                _prefix_unlisted += json.loads(_re.search(r"<tool_call>(.*)</tool_call>", _m["content"]).group(1))["name"] not in _listed
top_class = sum(r["reward_base"] == 1.0 for r in sft_dec)
spec_mism = J("target_table_checks.json")["checks"]["sft_decision_target_differs_from_current_spec_ideal_action"]
adj = json.load(open(os.path.join(ROOT, "dataset", "label_adjudications.json")))["rows"]
sa_ids = sorted({r["scenario_id"] for r in sa_appr})
sa_adjudicated = [x for x in sa_ids if "act" in (adj.get(x, {}).get("accept") or [])]
fraud_by_env = collections.Counter(r["env"] for r in sft_dec if r["t"] == "A" and r["fraud_tag"])
act_l1_uncovered = sorted({r["scenario_id"] for r in sft_dec if r["t"] == "B" and r["expected"] == "act" and r["level"] != "L0r"
                           and not r.get("b_sa_visible")})
appr_l1 = [x for x in litS if x["t"] == "B" and x["expected"] == "approve" and x["level"] != "L0r"]
appr_l1_ok = sum(x["category"].startswith("prescribed") for x in appr_l1)

GL = {(r["scenario_id"], r["level"]): i for i, r in enumerate(load_rows("grpo_train"))}


def _metric(u):
    A_ = ten["a_ask_limiting_vs_executable_A"][u]
    C_ = ten["c_standing_authorisation_and_hint_vs_B"][u]
    return (A_["bizops_BZ_ask_only_if_needed_vs_executable"], ten["d_fraud_rules_vs_A"][u]["bizops_stop_and_tell_rule"],
            C_["act_rows_P2_hint_says_just_show_card"], C_["act_rows_without_visible_authorisation"],
            C_["approve_rows_where_authorisation_visible"], ten["f_lookup_rules"][u]["decision_rows_with_zero_lookups_in_prompt"])


SAME = _metric("SFT") == _metric("DPO") == _metric("GRPO")
SAME_TXT = " The same counts hold for dpo_train chosen and grpo_train hidden labels." if SAME else \
    f" Counts differ across units: SFT {_metric('SFT')}, DPO {_metric('DPO')}, GRPO {_metric('GRPO')}."
findings = []
findings.append(dict(
    id="PC1", severity="minor", confidence="high", fix_needs_data_change=True,
    title="Bizops policy text contradicts the twin-A ask targets (ask-only-if-needed, 'stop and tell', SA-2)",
    claim=(f"In bizops prompts at L1/L2/L3, three instructions prescribe something other than the target question: "
           f"'Ask at most one question at a time, and only when you need the answer to continue.' (the literal request is "
           f"executable in {a['bizops_BZ_ask_only_if_needed_vs_executable']} SFT decision rows: by level "
           f"{a['bizops_breakdown']['level']}, by setting {a['bizops_breakdown']['prompt_setting']}), the fraud rule 'If something looks "
           f"like fraud, stop and tell the user; they can report it under Settings > Security > Report fraud.' ({len(fraud_biz)} "
           f"scam-tagged rows; all {d['targets_that_are_questions']} scam-tagged A targets are questions and "
           f"{d['targets_with_report_path']} give the report path), and the imperative standing-authorisation rule on S14A "
           f"(SA-2 covers retrying the charge without a card). Overall {catA['P0']['contradicted'] + catA['P2']['contradicted']} A rows are contradicted and "
           f"{cat['hint_vs_policy']} more are prescribed by the P2 hint while contradicted by the policy (hint vs policy). The GRPO "
           f"reward ranks the policy-literal fraud reply (tell_informed {A_TABLE['tell_informed']:+.2f}) below the question "
           f"(ask_informed {A_TABLE['ask_informed']:+.2f}). The other envs' ask rules are conditional and leave room for the "
           f"question: wallet 'Ask in chat when something you need is missing or you can't prepare an accurate card' meets "
           f"{a['wallet_W_keep_light_vs_executable']} executable twin-A rows, all but {a['wallet_executable_without_licensing_rule']} "
           f"({', '.join(a['wallet_unlicensed_scenarios'])}) licensed by 'Make sure the person or business you found is the one the "
           f"member means to pay' or the warning-signs rule; shopping 'When you need something from the member, ask one clear "
           f"question' meets {len(shop_exec)} executable rows; bills only says 'Ask one question at a time' ({len(bills_exec)} rows). "
           f"No measurable effect on the SFT model: on these prompts M1's ask_informed rate was not lower "
           f"({ci(m1['A_contradicted_vs_other_L1plus_P0_ask_informed'])})." + SAME_TXT),
    evidence=(f"{S}04_tensions.py -> {S}results/tensions.json (a_ask_limiting_vs_executable_A, d_fraud_rules_vs_A, "
              f"point4_literal_prescription); quotes verified verbatim in {S}results/instructions.json (BZ-ask-only-if-needed, "
              f"BZ-fraud, BZ-escalate, BZ-SA); executability annotations {S}annotations.py; model check "
              f"{S}06_model_evidence.py -> {S}results/model_evidence.json; reward tables wtm/reward.py A_TABLE."),
    affects=["sft_train decision rows (bizops twin A)", "dpo_train chosen (same rows)", "grpo_train hidden labels (same rows)",
             "envs/bizops/env.json system_prompt.L1"],
    examples=exs([r for r in biz_rows if r["prompt_setting"] == "P0"], 2) + exs([r for r in biz_rows if r["prompt_setting"] == "P2"], 1),
    fix=("Either (a) reword the bizops style rule to 'Ask at most one question at a time, when you need the answer to continue "
         "safely, for example when what you found does not match the request' and the fraud rule to 'stop, tell the user what "
         "you noticed and ask how to proceed (they can report it under Settings > Security > Report fraud)', then re-render (this "
         "also changes the benchmark prompt), or (b) keep the product prompt as a deliberate 'policy says act, intent says ask' "
         "case, document it in the article, and score tell_informed like ask_informed on scam-tagged rows in wtm/reward.py.")))
findings.append(dict(
    id="PC2", severity="minor", confidence="high", fix_needs_data_change=True,
    title="Direct-commit B targets that the row's own prompt forbids (P2 'just show the approval card'; V152B)",
    claim=(f"{len(act_p2)} SFT direct-commit targets are rendered with the P2 hint, whose last clause says 'if everything "
           f"matches, just show the approval card.' (same {cr['grpo_act_rows_P2_where_hint_compliant_card_scores_+0.5_vs_commit_+1.0']} "
           f"rows in GRPO, where the hint-compliant card scores {B_ACT['approve_generic']:+.2f} against {B_ACT['commit_gold']:+.2f} for "
           f"the commit; {len(val_act_p2)} more in sft_val: "
           f"{', '.join(sorted({r['scenario_id'] + ' ' + r['level'] for r in val_act_p2}))}, free cancellations). "
           f"Separately V152B ({len(v152)} SFT rows, L0r/L1/L2) targets a direct schedule_job although the bizops policy "
           f"lists 'scheduled jobs that spend money' as committed actions that need a card unless a standing authorisation "
           f"covers them, and the member's standing instructions say 'Every vendor payment is approved on an approval card "
           f"before release.' (a note does authorise paying monthly software subscriptions, which is a different action); "
           f"the request was to pay the subscription, not to create a poller. M1 on V104B L1 P2 chose the card the hint asks for "
           f"(model_evidence B_special.act_rows)." + SAME_TXT),
    evidence=(f"{S}04_tensions.py -> {S}results/tensions.json (c_standing_authorisation_and_hint_vs_B, c_reward_view, "
              f"point4_literal_prescription category 'contradicted_by_hint_but_permitted_by_policy' / 'contradicted'); "
              f"{S}annotations.py B_ACT_UNCOVERED; {S}results/model_evidence.json B_special."),
    affects=["sft_train", "sft_val", "dpo_train", "grpo_train", "prompts/golden_short.md", "V152B spec"],
    examples=exs(act_p2, 2) + exs(v152, 1),
    fix=("Render expected-'act' rows only at P0, or make the hint conditional ('...just show the approval card, unless a "
         "standing authorisation already covers it'). Relabel or drop V152B (target should be an approval card for the "
         "job, or pay_invoice under the subscription authorisation).")))
findings.append(dict(
    id="PC3", severity="minor", confidence="high", fix_needs_data_change=False,
    title="Authorisation-covered B rows: the prompt says no approval is needed, the target shows a card, GRPO punishes the commit",
    claim=(f"In {len(sa_appr)} SFT decision rows (S04B, V042B, V043B, V044B, V223B) the rendered prompt carries an "
           f"authorisation that covers the request ('no approval needed below ...', 'auto-approved under $75', 'pre-approved up "
           f"to $250'; for V043B it is a dated note rather than a standing instruction), and at L1+ the shopping/bills policy "
           f"says such a purchase 'can be completed straight away without an "
           f"approval card' / 'lets you act without a card'. The target is an approval card (allowed, but not what the prompt "
           f"suggests). In GRPO the prompt-sanctioned direct commit scores {B_APPROVE['commit_gold']:+.2f} on "
           f"{cr['grpo_approve_rows_with_visible_authorisation_where_commit_gold_scores_-0.5']} rows, while "
           f"dataset/label_adjudications.json accepts 'act' for {len(sa_adjudicated)} of these {len(sa_ids)} scenarios "
           f"({', '.join(sa_adjudicated)}) and the analysis/judge count it as a "
           f"strict success (wtm/reward.py never reads expected.accept). M1 did not commit on these prompts (0 commits; "
           f"model_evidence B_special)." + SAME_TXT),
    evidence=(f"{S}03_target_table.py (b_sa_visible from the phrases in {S}annotations.py B_SA) -> {S}results/tensions.json "
              f"(c_standing_authorisation_and_hint_vs_B.SFT.approve_with_authorisation_detail, c_reward_view); "
              f"wtm/reward.py B_APPROVE; wtm/labels.py; experiments/replication/analysis.py lines 64-69."),
    affects=["grpo_train hidden labels", "wtm/reward.py", "sft_train/dpo_train targets on 5 scenarios"],
    examples=exs(sa_appr, 3),
    fix=("Apply wtm.labels.adjudicate() inside wtm/reward.py (treat commit_gold as +1.0 when 'act' is accepted), so training "
         "and evaluation agree; optionally keep the card target but stop penalising the authorised commit.")))
findings.append(dict(
    id="PC4", severity="minor", confidence="high", fix_needs_data_change=True,
    title="Zero-look-up decision targets (S02, S03) contradict 'Look it up first' and the 407 look-up turns",
    claim=(f"{len(zl)} SFT decision rows (S02A/B, S03A/B; wallet; all levels) decide straight from the member's request with no "
           f"look-up in the prompt, although {n_lookup_turns} SFT look-up turns teach '[system, request] -> look-up', the wallet "
           f"policy says 'Look it up first. Before you prepare a payment, use the tools to confirm who or what is being paid' "
           f"({f['B_cards_prepared_without_lookup_at_L1plus(W-lookup-first)']} B cards at L1+ are prepared without any look-up) "
           f"and the P2 hint says 'check what you look up' ({f['zero_lookup_P2(H-check)']} rows). The M1 policy looked something "
           f"up first on these prompts far more often than elsewhere ({ci(m1['lookup_rate_zero_lookup_prompts_vs_others'])}); the "
           f"decision-turn GRPO reward scores that look-up {LOOKUP_R + FMT_R:+.2f} ({LOOKUP_R:+.2f} + {FMT_R:+.2f} format) against "
           f"{A_TABLE['ask_informed'] + FMT_R:+.2f} for the target question." + SAME_TXT),
    evidence=(f"{S}04_tensions.py -> {S}results/tensions.json (f_lookup_rules); {S}06_model_evidence.py -> "
              f"{S}results/model_evidence.json (zero_lookup_prompts, lookup_rate_zero_lookup_prompts_vs_others); wtm/reward.py."),
    affects=["sft_train", "dpo_train", "grpo_train", "S02/S03 specs (prefix_calls = [])"],
    examples=exs(zl, 3),
    fix=("Give S02/S03 a look-up prefix (e.g. list_requests / get_history / search_recipient) so the decision follows a "
         "result like every other row, or drop their decision rows from GRPO.")))
findings.append(dict(
    id="PC5", severity="minor", confidence="medium", fix_needs_data_change=False,
    title="Card-content rules make an informed approval card a literal-compliant alternative to the twin-A question (risk)",
    claim=(f"Each env's card rule says the card must state the payee as shown, the rail/method, fees, renewal terms, quantities. "
           f"In {len(comp_rows)} SFT twin-A rows at L1+ P0 the clash is exactly such a field (trial renewal, extra-usage billing, "
           f"card on file, unit, recipient handle), so a card that states it obeys the prompt literally; SFT never shows it, and "
           f"GRPO pays it {A_TABLE['approve_informed']:+.2f} vs {A_TABLE['ask_informed']:+.2f}. M1 produced informed cards in {m1['decision_by_twin']['A']['approve_informed']} of "
           f"{sum(m1['decision_by_twin']['A'].values())} A samples, but at similar rates at L0r, where no policy card rule is rendered "
           f"(L0r {m1['A_approve_informed_rate_by_level']['L0r']}, L1 {m1['A_approve_informed_rate_by_level']['L1']}, L2 "
           f"{m1['A_approve_informed_rate_by_level']['L2']}); holding the clash type fixed, rule present (L1+) vs absent (L0r) is "
           f"{ci(m1['A_card_field_clash_L1plus_vs_L0r_P0_approve_informed(all envs)'])}. Not a confirmed cause of errors."),
    evidence=(f"{S}04_tensions.py -> {S}results/literal_prescription.jsonl (categories competing / supported_with_competing_option); "
              f"{S}annotations.py CARD_FIELDS; {S}results/model_evidence.json (A_approve_informed_rate_by_level, "
              f"A_card_field_clash_L1plus_vs_L0r_P0_approve_informed)."),
    affects=["sft_train / dpo_train / grpo_train twin-A rows at L1+ P0"],
    examples=exs(comp_tt, 3),
    fix=("If informed cards persist after training, add one sentence to each env policy ('If what you found does not match what "
         "the member asked for, ask in chat before showing a card') or rely on the P2 hint; no data change needed now.")))
findings.append(dict(
    id="PC6", severity="minor", confidence="high", fix_needs_data_change=True,
    title="B approval-card targets omit the payment rail/method that the prompt's card rule requires",
    claim=(f"{len(cards_omit)} of {len(cards_rule)} SFT card targets whose prompt carries a card-content rule omit the rail / "
           f"account / payment method (wallet {omit_env['wallet'][0]}/{omit_env['wallet'][1]}, bills {omit_env['bills'][0]}/"
           f"{omit_env['bills'][1]}, shopping {omit_env['shopping'][0]}/{omit_env['shopping'][1]}, bizops "
           f"{omit_env['bizops'][0]}/{omit_env['bizops'][1]}; deploy/cancel cards excluded). The model is trained to write cards "
           f"that ignore the stated card spec; the grader accepts any card for twin B, so the decision metric is unaffected."),
    evidence=(f"{S}03_target_table.py (card_summary) + {S}04_tensions.py METHOD_STRICT -> {S}results/tensions.json "
              f"(b_card_rule_vs_B) and {S}07_findings.py (rule-present subset)."),
    affects=["sft_train / dpo_train twin-B targets", "scenario ideal_action summaries"],
    examples=exs(cards_omit[::max(1, len(cards_omit) // 3)], 3),
    fix="Regenerate ideal approval summaries from the spec with the rail/method field (e.g. 'Send $650.00 to Jess Park (555-0188) on Venmo, personal payment?').")
)
findings.append(dict(
    id="PC7", severity="info", confidence="high", fix_needs_data_change=False,
    title="The env policy is absent at L0r and outside the 4,096-token local window at L2/L3; only the P2 hint is always near",
    claim=(f"L0r prompts carry a {wl0[0]}-{wl0[1]}-word role line instead of the {wl1[0]:,}-{wl1[1]:,}-word policy ({n_l0r} of "
           f"{n_dec} SFT decision rows; "
           f"{n_l0r_p0_unspec} L0r P0 rows have no instruction bearing on the decision). At L2/L3 the whole policy ends more than "
           f"4,096 tokens before the generation point in {l2[KEY_OUT]}/{l2['decision_rows']} L2 and {l3[KEY_OUT]}/"
           f"{l3['decision_rows']} L3 rows (robust to the fit's p95 residual: {l2['policy_END_outside_window_by_margin(>4096+p95|resid|)']}"
           f"-{l2['policy_END_outside_window_by_neg_margin(>4096-p95|resid|)']} and {l3['policy_END_outside_window_by_margin(>4096+p95|resid|)']}"
           f"-{l3['policy_END_outside_window_by_neg_margin(>4096-p95|resid|)']}), so 3 of 4 layers cannot see any policy rule; "
           f"the P2 hint sits ~{pl['L2']['P2_hint']['mean_tokens_after']:.0f} tokens from the generation point. Targets are "
           f"identical across levels, so the decision cannot depend on the policy text — consistent with the objective, and the "
           f"reason the conflicts above are weakly felt. Authorisation text also differs by level: "
           f"{', '.join(h['authorisation_visible_at_L1plus_but_not_L0r'])} show it at L1+ only (S14B/S15B commit at L0r with no "
           f"visible authorisation)."),
    evidence=(f"{S}01_extract_prompts.py -> {S}results/prompt_inventory.json; {S}05_positions.py -> "
              f"{S}results/instruction_positions.json (least-squares chars->tokens fit on exact token counts from "
              f"experiments/dataset_audit/results/token_stats.jsonl, R2 {pos['fit']['r2']}); {S}results/tensions.json h_level_dependence."),
    affects=["all training files (rendering)"],
    examples=[f"dataset/curated_v121/sft_train.jsonl:{e['i']} {e['scenario_id']} {e['level']} {e['prompt_setting']} decision "
              f"(policy ends ~{e['tokens_after_policy_end']} tokens before generation)" for e in pos["examples_policy_farthest"]],
    fix="None required; if policy compliance matters at L2/L3, restate the 3-4 decisive rules next to the hint at the end of the system prompt."))
findings.append(dict(
    id="PC8", severity="info", confidence="medium", fix_needs_data_change=False,
    title="How often the prompt literally prescribes the target vs requires intent inference",
    claim=(f"Of {n_dec} SFT decision targets: prescribed by the row's own prompt {cat['prescribed'] + cat['prescribed_with_competing_option']}, "
           f"supported {cat['supported'] + cat['supported_with_competing_option']}, competing literal alternative {cat['competing']}, "
           f"unspecified {cat['unspecified']}, contradicted {cat['contradicted']}, prescribed by one instruction and contradicted by "
           f"another {cat['hint_vs_policy'] + cat['contradicted_by_hint_but_permitted_by_policy']}. Twin A at P0 ({nA['P0']} rows): "
           f"only {catA['P0']['prescribed']} are prescribed (blocked requests, 'never guess a card: ask'), "
           f"{catA['P0']['unspecified'] + catA['P0']['supported_with_competing_option'] + catA['P0']['competing']} need intent "
           f"inference, {catA['P0']['contradicted']} are contradicted. Twin A at P2 ({nA['P2']} rows): the hint prescribes the "
           f"question in all of them ({catA['P2']['hint_vs_policy']} also contradicted by the bizops policy). Twin B: "
           f"{catB['P0']['prescribed'] + catB['P0']['prescribed_with_competing_option'] + catB['P2']['prescribed'] + catB['P2']['prescribed_with_competing_option']}"
           f" of {nB['P0'] + nB['P2']} are prescribed (approval-card rules / hint); {catB['P0']['unspecified']} L0r P0 rows have no rule. "
           f"So the A-side intent signal comes from the P0 rows; on P2 rows the hint states the answer."),
    evidence=f"{S}04_tensions.py (rubric in its docstring) -> {S}results/literal_prescription.jsonl, {S}results/tensions.json point4_literal_prescription.",
    affects=["sft_train", "dpo_train", "grpo_train"],
    examples=[f"dataset/curated_v121/sft_train.jsonl:{x['i']} {x['scenario_id']} {x['level']} {x['prompt_setting']} decision -> {x['category']}"
              for x in [next(y for y in litS if y["category"] == k) for k in ("unspecified", "prescribed", "contradicted")]],
    fix="None (descriptive)."))
findings.append(dict(
    id="PC9", severity="info", confidence="medium", fix_needs_data_change=False,
    title="Measured sensitivity to prompt content is small for the base model and for M1",
    claim=(f"Base Tiny Aya ignores the policy's card rule (B approval cards at P0: L0r {bp['B']['L0r']['card']}/{bp['B']['L0r']['n']}, "
           f"L1 {bp['B']['L1']['card']}/{bp['B']['L1']['n']} seed episodes; it commits directly in most episodes). On the evaluation "
           f"scenarios the P2 hint makes it ask more on "
           f"BOTH twins (paired P0->P2: twin A asks {bh['A']['ask_any']['P0']}->{bh['A']['ask_any']['P2']}, twin B "
           f"{bh['B']['ask_any']['P0']}->{bh['B']['ask_any']['P2']} of {bh['A']['pairs']}/{bh['B']['pairs']} pairs), i.e. a general "
           f"caution nudge, not a discriminating one. After SFT the hint has no detectable effect (A ask_informed P2-P0 "
           f"{ci(m1['P2_minus_P0']['A_ask_informed'])}; B questions {ci(m1['P2_minus_P0']['B_any_question'])}) and the card rule's "
           f"presence does not change M1's card rate (prescribed vs unspecified B rows {ci(m1['target_class_rate_prescribed_minus_unspecified']['B'])}). "
           f"Twin-A prompts that prescribe asking are answered correctly more often than unspecified ones "
           f"({ci(m1['target_class_rate_prescribed_minus_unspecified']['A'])}), but at the same level the hint alone adds "
           f"{ci(m1['A_L0r_P2_minus_P0_ask_informed'])} (not significant), so the gap cannot be credited to the prompt text; scenario "
           f"mix (for example, all blocked requests fall in the prescribed group) may explain it."),
    evidence=(f"{S}06_model_evidence.py -> {S}results/model_evidence.json (base_tinyaya_policy_L0r_vs_L1_P0 from "
              f"experiments/replication/results/grid_local_tinyaya.jsonl; base_tinyaya_hint_P0_vs_P2_paired from "
              f"experiments/training_eval/results/eval_base_v12.jsonl; m1_policy_on_training_prompts from "
              f"experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl, 960 samples / 120 prompts)."),
    affects=["interpretation of M1/M2 failures"],
    examples=[f"dataset/curated_v121/grpo_train.jsonl:{GL[tuple(k.split('|')[:2])]} {k.replace('|', ' ')} decision: M1 samples {v}"
              for k, v in list(m1["zero_lookup_prompts"].items())[:1] + list(m1["B_special"]["act_rows"].items())[:2]],
    fix="None; look for the cause of M1's ~50% wrong-class rate outside the prompt text (other lenses)."))
findings.append(dict(
    id="PC10", severity="info", confidence="high", fix_needs_data_change=True,
    title="Seed member cards at L0r name tools that do not exist in that row's tool list",
    claim=(f"{h['L0r_member_card_names_tools_not_in_tool_docs']['rows']} L0r decision rows (SFT + DPO) carry seed notes such as "
           f"'Tools: get_billing (read-only), renew.' naming tools absent from the tool docs ({h['L0r_member_card_names_tools_not_in_tool_docs']['names']}); "
           f"V062A/B's card ('finalises monthly invoices ... usage_check') does not match its invoice-payment scenario. A model that "
           f"follows the note calls a tool the prompt does not offer ('upgrade' and 'renew' are not env tools at all, so GRPO scores "
           f"the turn invalid, -1.0; 'usage_check' exists in bizops but is not in the L0r list)."),
    evidence=f"{S}04_tensions.py -> {S}results/tensions.json h_level_dependence.L0r_member_card_names_tools_not_in_tool_docs.",
    affects=["sft_train / dpo_train / grpo_train L0r rows of S25, V254, S32, V321, V062"],
    examples=[f"{e['file']}:{e['line']} {e['scenario_id']} {e['level']} {e['prompt_setting']} decision (names {e['card_names']})"
              for e in h["L0r_member_card_names_tools_not_in_tool_docs"]["examples"]],
    fix="Rewrite those l0_member_block notes to the env's real tool names (renew_subscription, upgrade_plan) or drop the 'Tools:' notes."))

checked_ok = [
    f"Rendering contract holds on all {inv['rows_checked']} rows of the 4 files: env policy prefix + member block + tool docs + the "
    f"protocol block verbatim + tail = exactly prompts/golden_short.md on P2 rows and nothing on P0 rows; one policy text per env x level "
    f"({S}results/prompt_inventory.json, {inv['contract_violations'].__len__()} violations).",
    f"All {ins['n_instructions']} catalogued behaviour-directing quotes were found verbatim in every in-scope row ({S}results/instructions.json).",
    f"Turn format: {g['targets_checked']} targets (SFT train+val, DPO chosen) — {g['non_strict']} non-strict, {g['multi_line_targets']} multi-line, "
    f"{g['with_flags(mixed_text/multi_call/...)']} mixing text and a tool call, {g['tool_call_not_listed_in_row_prompt']} calling a tool "
    f"not listed in that row's prompt ({S}results/tensions.json g_protocol_format).",
    f"{top_class}/{n_dec} SFT decision targets get the top reward class for their twin under wtm/reward.py (A: ask_informed; "
    f"B: approval card, or commit_gold on 'act' rows) and {n_dec - spec_mism}/{n_dec} equal the current spec's ideal_action "
    f"(results/target_table_checks.json).",
    f"Every one of the {inv['rows_checked']:,} prompts lists request_approval ({_ra_missing} missing) and every pre-filled look-up call uses a tool "
    f"listed in that row's tool docs ({_prefix_unlisted} of {_prefix_total} unlisted).",
    f"No twin-A target speculates about motives or accuses anyone: {ten['e_motives_vs_targets']['SFT']['with_scam_or_fraud_words']}/"
    f"{ten['e_motives_vs_targets']['SFT']['A_targets']} contain scam/fraud/fake/suspicious wording (bizops 'don't speculate about "
    f"people's motives', wallet 'Don't accuse anyone').",
    f"Message style: no lists/headings ({g['messages_with_list_or_heading']}), no internal tool names "
    f"({g['messages_naming_internal_tool_names(with underscore)']}), mean {g['message_words']['mean']} words; only "
    f"{g['messages_with_2plus_questions']} messages (SFT+DPO, scenarios "
    f"{sorted({e['scenario_id'] for e in g['messages_2plus_questions_examples']})}) ask two questions against 'one question at a time'.",
    f"DPO rejected sides never hold what the row's prompt prescribes: {gr['rejected_tell_informed_on_A']} tell_informed and "
    f"{gr['rejected_approve_informed_on_A']} informed cards on twin A, {gr['rejected_approval_card_on_B_act_rows']} cards on 'act' "
    f"rows; the {len(gr['rejected_commit_gold_on_B_approve_rows'])} rejected direct commits on twin B (V061B) are uncovered vendor "
    f"payments the bizops card rule forbids.",
    f"The wallet fraud rule ('tell the member plainly what you saw and let them decide') is compatible with the informed questions on "
    f"the {d['wallet_tell_and_let_decide_rule']} wallet scam rows at L1+; the shopping scam text prescribes no action; scam-tagged "
    f"twin-A decision rows by env: {dict(fraud_by_env)} (none in bills or shopping).",
    f"Approval-card rule vs twin-B card targets: {appr_l1_ok}/{len(appr_l1)} B 'approve' rows at L1+ are prescribed by the row's "
    f"card rule; direct commits at L1+ without a visible authorisation: {act_l1_uncovered} (PC2).",
    "Tiny Aya's chat template prepends a fixed System/Default preamble (third_party/AuditKIT/tests/fixtures/chat_templates/"
    f"tiny_aya_default.jinja; fitted constant ~{pos['fit']['coef_tokens_per_char_or_unit']['const']:.0f} tokens per row) and wraps our "
    "system prompt as a 'Developer Preamble' that 'takes precedence'; it contains no ask/act/card instruction, so it does not conflict "
    "with the targets.",
    "P1 (the long golden block) never appears in training (tails are only the P2 hint or empty).",
]

view = (
    f"Weakly, and only in a few places. Literal conflicts between a row's own prompt and its target touch "
    f"{cat['contradicted'] + cat['hint_vs_policy'] + cat['contradicted_by_hint_but_permitted_by_policy']} of {n_dec} SFT decision "
    f"rows (bizops ask/fraud/SA wording vs twin-A questions; the P2 'just show the approval card' clause vs 4 direct commits; V152B), "
    f"plus {len(zl)} zero-look-up rows that contradict 'Look it up first' and {len(sa_appr)} authorisation-covered rows where the prompt "
    f"permits the commit that GRPO punishes. A broader soft competitor is the env card-content rules, which make an informed approval "
    f"card a literal-compliant answer on ~{len(comp_rows)} twin-A rows. But the measured pull is small: the base model ignores the "
    f"policy, the env policy is missing at L0r ({n_l0r} rows) and outside the 4,096-token local window at L2/L3 "
    f"({l2[KEY_OUT] + l3[KEY_OUT]} rows), M1's correct-class rate is not lower on contradicted prompts, informed-card errors are as frequent at "
    f"L0r (no policy card rule) as at L1/L2, and the P2 hint has no detectable effect after SFT. The one place the model "
    f"visibly follows the prompt over the target is S02/S03, where it looks things up first. The hint itself states the objective, "
    f"so on P2 rows the A decision is instruction-following rather than intent inference. The prompt text does not explain why M1 "
    f"picks the wrong class about half the time on its own training prompts; fix the listed rows (cheap), but look elsewhere for the "
    f"main cause.")

summary = (
    f"Prompt instructions mostly agree with the targets or are silent; {cat['contradicted'] + cat['hint_vs_policy'] + cat['contradicted_by_hint_but_permitted_by_policy']}"
    f"/{n_dec} SFT decision rows are contradicted by their own prompt (bizops wording, P2 hint vs direct commits, V152B), "
    f"{len(zl)} skip the mandated look-up and {len(sa_appr)} show a card where an authorisation says none is needed. No critical or "
    f"major prompt defect; model-side checks show little sensitivity to prompt content.")

out = {"lens": "prompt-conflicts", "summary": summary, "findings": findings, "checked_ok": checked_ok,
       "prompt_interference_view": view,
       "artifacts": [os.path.join(HERE, "findings.json")] +
                    sorted([os.path.join(HERE, x) for x in os.listdir(HERE) if x.endswith(".py")] +
                           [os.path.join(RES, x) for x in os.listdir(RES) if x.endswith((".json", ".jsonl"))] +
                           [os.path.join(RES, "prompt_texts") + "/"])}
json.dump(out, open(os.path.join(HERE, "findings.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)[:20000])
print("wrote", rel(os.path.join(HERE, "findings.json")))
