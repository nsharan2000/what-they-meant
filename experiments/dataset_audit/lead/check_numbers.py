"""Lead audit: re-read every headline number cited in experiments/dataset_audit/REPORT.md from the lens artifacts.

Each entry = (id, artifact path relative to experiments/dataset_audit/, key path, expected value as cited in the report).
Key paths use '/' between dict keys (or a list of parts when a key contains '/'); a trailing '[i]' indexes a list. Floats compare with a small tolerance.
Writes lead/results/headline_numbers.json and exits 1 if any cited value is missing or differs.
Run: python3 experiments/dataset_audit/lead/check_numbers.py   (numpy/pandas not needed; stdlib only)
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/dataset_audit

CHECKS = [
    # --- SFT objective / decision token (problem 1, section 4) ---
    ("sft_trained_tokens", "composition-memorization/results/token_composition.json", "sft_train/trained_tokens_total", 38160),
    ("first_token_share", "composition-memorization/results/token_composition.json", "sft_train/decision_bearing_share/first_token_only", 0.013),
    ("pivot_share", "composition-memorization/results/token_composition.json", "sft_train/decision_bearing_share/pivots_only", 0.0195),
    ("lookup_token_share", "composition-memorization/results/token_composition.json", "sft_train/lookup_token_share", 0.3646),
    ("decision_nats_per_token", "composition-memorization/results/token_composition.json", "loss_attribution/mean_per_token_loss_from_decision_class", 0.011),
    ("decision_nats_per_token_corrected", "composition-memorization_verify/results/v6_extra.json", "implied_nats_per_trained_token_type_nll_corrected", 0.0079),
    ("coinflip_global", "twin-contrast-shortcuts_verify/results/v5_decision_token.json", "coinflip_all_decisions_added_to_global_mean_loss", 0.0092),
    ("loss_epoch2_mean", "composition-memorization_verify/results/v4_sftlog.json", "loss_mean_epoch2", 0.2112),
    ("loss_epoch2_sd", "composition-memorization_verify/results/v4_sftlog.json", "loss_sd_epoch2", 0.1043),
    ("pivot_weight_w10", "composition-memorization/results/token_composition.json", "pivot_weighting/w10", 0.1662),
    ("pivot_weight_w20", "composition-memorization/results/token_composition.json", "pivot_weighting/w20", 0.2851),
    ("A_targets_start_with_word", "twin-contrast-shortcuts_verify/results/v5_decision_token.json", "first_token_is_lt/A_decision/no", 247),
    ("B_targets_start_with_lt", "twin-contrast-shortcuts_verify/results/v5_decision_token.json", "first_token_is_lt/B_decision/yes", 249),
    ("only_result_pair_levels", "composition-memorization_verify/results/v3_dupes.json", "sft_train/twin_pair_levels_only_result_differs_hint_stripped", 133),
    ("only_result_share", "composition-memorization_verify/results/v3_dupes.json", "sft_train/share_only_result_hint_stripped", 0.5385),
    ("only_result_diff_chars_median", "composition-memorization_verify/results/v3_dupes.json", "sft_train/only_result_pairs_result_diff_chars_median", 28.0),
    ("only_result_diff_share_prompt", "composition-memorization_verify/results/v3_dupes.json", "sft_train/only_result_pairs_diff_share_of_prompt_chars_median", 0.0029),
    ("pair_levels_hint_differs", "composition-memorization_verify/results/v3_dupes.json", "sft_train/twin_pair_levels_ps_differs", 113),
    ("pair_levels_only_result_raw", "composition-memorization_verify/results/v3_dupes.json", "sft_train/twin_pair_levels_only_result_differs_raw", 72),
    ("m1_acc_request_unique", "composition-memorization_verify/results/v3_dupes.json", "m1_acc_by_request_shared_with_twin/False[0]", 0.523),
    ("m1_acc_request_shared", "composition-memorization_verify/results/v3_dupes.json", "m1_acc_by_request_shared_with_twin/True[0]", 0.524),
    ("act_commit_rate", "composition-memorization_verify/results/v13_act.json", "commit_rate_act_prompts", 0.575),
    ("approve_commit_rate", "composition-memorization_verify/results/v13_act.json", "commit_rate_approve_prompts", 0.0241),
    ("sft_act_rows", "composition-memorization_verify/results/v13_act.json", "sft_act_rows", 18),
    # --- M1 on its own training prompts ---
    ("m1_gold_A", "grpo-dynamics_verify/results/v01_policy.json", "gold_rate_by_cell/A", 0.4741),
    ("m1_gold_B_approve", "grpo-dynamics_verify/results/v01_policy.json", "gold_rate_by_cell/B_approve", 0.5526),
    ("m1_gold_B_act", "grpo-dynamics_verify/results/v01_policy.json", "gold_rate_by_cell/B_act", 0.475),
    ("m1_groups_mixed", "grpo-dynamics_verify/results/v01_policy.json", "groups_0_lt_pass_lt_1", 108),
    ("m1_pass_ge_half", "grpo-dynamics_verify/results/v01_policy.json", "groups_pass_ge_half", 0.6667),
    ("m1_unique_plurality_gold", "grpo-dynamics_verify/results/v01_policy.json", "groups_plurality_class_is_gold_unique", 0.5667),
    ("m1_strict_acc_overall", "splits-eval-alignment_verify/results/v06_positions_and_m1.json", "part2_m1/overall/acc", 0.524),
    ("m1_acc_L0r", "splits-eval-alignment_verify/results/v06_positions_and_m1.json", "part2_m1/by_level/L0r/acc", 0.486),
    ("m1_acc_L1", "splits-eval-alignment_verify/results/v06_positions_and_m1.json", "part2_m1/by_level/L1/acc", 0.523),
    ("m1_acc_L2", "splits-eval-alignment_verify/results/v06_positions_and_m1.json", "part2_m1/by_level/L2/acc", 0.597),
    ("m1_acc_L3", "splits-eval-alignment_verify/results/v06_positions_and_m1.json", "part2_m1/by_level/L3/acc", 0.479),
    ("majority_of_8", "composition-memorization_verify/results/v6_extra.json", "majority_of_8_accuracy", 0.575),
    ("majority_of_8_A", "composition-memorization_verify/results/v6_extra.json", "majority_of_8_accuracy_by_twin/A", 0.5259),
    ("majority_of_8_B", "composition-memorization_verify/results/v6_extra.json", "majority_of_8_accuracy_by_twin/B", 0.621),
    ("betabinom_E_p", "composition-memorization_verify/results/v6_extra.json", "first_token_type_betabinomial/all/E_p_goldtype", 0.5862),
    ("betabinom_nll", "composition-memorization_verify/results/v6_extra.json", "first_token_type_betabinomial/all/E_neglog_p_goldtype_nats", 0.6069),
    ("m1_dprime", "grpo-dynamics_verify/results/v11_checks.json", "M1_dprime", 0.4819),
    ("base_dprime", "grpo-dynamics_verify/results/v11_checks.json", "base_tiny_aya_dprime", 0.2295),
    ("strong_dprime", "grpo-dynamics_verify/results/v11_checks.json", "strong_dprime_sampled_scenarios", 0.8392),
    ("m1_vs_strong_spearman", "grpo-dynamics_verify/results/v10_extras.json", "spearman_M1_A_pass_vs_strong_success", 0.3534),
    ("discrimination_samples", "twin-contrast-shortcuts_verify/results/v3_m1.json", "discrimination_samples", 0.186),
    ("discrimination_majority", "twin-contrast-shortcuts_verify/results/v3_m1.json", "discrimination_majority", 0.323),
    ("level_adjR2_A", "grpo-dynamics_verify/results/v08_levels.json", "A/adj_R2", 0.0144),
    ("level_adjR2_B", "grpo-dynamics_verify/results/v08_levels.json", "B/adj_R2", 0.004),
    ("mde_P2_A", "grpo-dynamics_verify/results/v08_levels.json", "min_detectable_level_effect_A_2se/ps_P2", 0.184),
    ("within_scen_realistic_minus_L0r_A", "realism-pressure_verify/results/v10_extra.json", "M1_paired_L0r_vs_realistic[0]/diff", 0.1786),
    ("zero_lookup_rows", "prompt-conflicts_verify/results/05_pc4_pc5_pc6.json", "pc4/sft_train/zero_lookup", 16),
    ("system_user_lookup_rows", "composition-memorization_verify/results/v6_extra.json", "system_user_prompt_turns/lookup1", 401),
    # --- validation loss (section 4) ---
    ("eval_loss_ep1", "composition-memorization_verify/results/v4_sftlog.json", "evals[0]/eval_loss", 0.6442),
    ("eval_loss_ep2", "composition-memorization_verify/results/v4_sftlog.json", "evals[1]/eval_loss", 0.7456),
    ("eval_entropy_change_pct", "composition-memorization_verify/results/v4_sftlog.json", "eval_entropy_change_pct", -24.63),
    ("eval_acc_ep1", "composition-memorization_verify/results/v4_sftlog.json", "evals[0]/eval_mean_token_accuracy", 0.8513),
    ("eval_acc_ep2", "composition-memorization_verify/results/v4_sftlog.json", "evals[1]/eval_mean_token_accuracy", 0.8547),
    ("val_rise_if_pivots_plus1", "composition-memorization_verify/results/v4_sftlog.json", "eval_loss_rise_if_every_pivot_token_rose_by_1_nat", 0.0183),
    ("val_rise_needed_per_pivot", "composition-memorization_verify/results/v4_sftlog.json", "decision_pivot_nll_rise_needed_to_explain_whole_rise_nats", 5.53),
    ("distinct_targets", "composition-memorization_verify/results/v3_dupes.json", "sft_train/distinct_targets", 216),
    ("distinct_decision_conversations", "composition-memorization_verify/results/v3_dupes.json", "sft_train/distinct_decision_conversations_after_system", 202),
    ("exposures_decision", "composition-memorization_verify/results/v3_dupes.json", "sft_train/exposures_2ep_per_decision_target", 6.67),
    ("exposures_lookup", "composition-memorization_verify/results/v3_dupes.json", "sft_train/exposures_2ep_per_lookup_target", 12.15),
    ("train_loss_drop_within_ep2_first10", "composition-memorization_verify/results/v4_sftlog.json", "first10_epoch2", 0.3168),
    ("train_loss_drop_within_ep2_last10", "composition-memorization_verify/results/v4_sftlog.json", "last10_epoch2", 0.1664),
    # --- step size (problem 2) ---
    ("grpo_dW_ratio", "grpo-dynamics/results/update_size.json", "efficiency/observed_ratio_dW_grpo25_vs_sft", 0.018372153604004408),
    ("grpo_sum_lr_ratio_30", "grpo-dynamics_verify/results/v03_update_size.json", "ratio_sum_lr_30", 0.02456140350877193),
    ("grpo_bulk_eff", "grpo-dynamics_verify/results/v03_update_size.json", "grpo_bulk_efficiency_ckpt25", 0.22399687089516776),
    ("adam_noise_floor", "grpo-dynamics_verify/results/v03_update_size.json", "adam_sim_25_steps/0.0/rms_over_sumlr", 0.2212),
    ("grpo_upper_bound", "grpo-dynamics_verify/results/v03_update_size.json", "grpo30_upper_bound_vs_sft_dW", 0.09985005571852405),
    ("pred_lr5e5_36", "grpo-dynamics_verify/results/v03_update_size.json", "pred_dW_vs_sft_lr5e-05_steps36", 0.2716),
    ("pred_lr2e5_30", "grpo-dynamics_verify/results/v03_update_size.json", "pred_dW_vs_sft_lr2e-05_steps30", 0.0895),
    ("pred_lr1e4_30", "grpo-dynamics_verify/results/v03_update_size.json", "pred_dW_vs_sft_lr0.0001_steps30", 0.4473),
    ("merge_residual", "grpo-dynamics/results/update_size.json", "stack_check/merge_residual_mean_abs_dlogprob", 0.01916),
    ("m2_effect", "grpo-dynamics/results/update_size.json", "stack_check/m2_adapter_effect_mean_abs_dlogprob", 0.0171),
    ("pred_effect_lr5e5_36", "grpo-dynamics/results/update_size.json", "stack_check/pred_effect_grid_linear/lr5e-05_steps36", 0.2076),
    ("dpo_total_lr_ratio", "dpo-pairs_verify/results/v6_lr_budget.json", "dpo_over_sft_total_lr_linear", 0.014035087719298246),
    ("dpo_pred_dW_linear", "dpo-pairs_verify/results/v6_lr_budget.json", "predicted_dpo_dW_over_sft_linear(calib=grpo)", 0.012780628594090022),
    ("dpo_pred_dW_constant", "dpo-pairs_verify/results/v6_lr_budget.json", "predicted_dpo_dW_over_sft_constant(calib=grpo)", 0.02476246790104942),
    ("dpo_lr_for_10pct", "dpo-pairs_verify/results/v6_lr_budget.json", "lr_needed_for_10pct_of_sft_budget_linear", 3.5625e-05),
    ("dpo_sft_step_same_budget", "dpo-pairs_verify/results/v6_lr_budget.json", "sft_step_where_cum_lr_reaches_dpo_total", 4),
    ("grpo_push_shared", "grpo-dynamics_verify/results/v10_extras.json", "shared_push_decomposition/logged/shared_total", -80.664),
    ("grpo_push_composition", "grpo-dynamics_verify/results/v10_extras.json", "shared_push_decomposition/logged/composition_part", -20.671),
    ("grpo_push_no_card_credit", "grpo-dynamics_verify/results/v10_extras.json", "shared_push_decomposition/no_card_credit_on_A/shared_total", -60.906),
    # --- GRPO step time (budget) ---
    ("grpo_step_mean_s", "grpo-dynamics/results/step_time.json", "seconds_mean", 718.7),
    ("grpo_step_random_s", "grpo-dynamics/results/step_time.json", "pool_pred/all_levels/pred_seconds_random_step", 735.1),
    ("grpo_step_bucketed_s", "grpo-dynamics/results/step_time.json", "pool_pred/all_levels/pred_seconds_length_bucketed", 526.5),
    ("grpo_step_fit_r2", "grpo-dynamics/results/step_time.json", "fit_max_prompt/r2", 0.998),
    # --- DPO pairs (problem 3) ---
    ("dpo_A_rejected_eq_B_chosen", "dpo-pairs_verify/results/v3_contrasts.json", "A_rejected_type_equals_B_chosen_type", 12),
    ("dpo_prompt_free_cv", "dpo-pairs_verify/results/v3_contrasts.json", "cv_accuracy_mean_std/action_type_prompt_free[0]", 0.9355),
    ("dpo_qmark_rule", "dpo-pairs_verify/results/v3_contrasts.json", "rule_qmark", 0.9335),
    ("dpo_rejected_P1", "dpo-pairs_verify/results/v2_context.json", "grid_prompt/P1", 380),
    ("dpo_fully_on_context", "dpo-pairs_verify/results/v2_context.json", "fully_on_context(P0 grid, P0 dpo, same sys length, same messages)", 7),
    ("dpo_identical_twin_levels", "dpo-pairs_verify/results/v2_context.json", "twin_levels_identical_rejected", 91),
    ("dpo_pairs_identical_twin_levels", "dpo-pairs_verify/results/v2_context.json", "pairs_in_identical_twin_levels", 182),
    ("base_P0_A_commit", "dpo-pairs_verify/results/v7_base_errors.json", ["P0/A", "commit[1]"], 0.742),
    ("base_P0_Bapprove_commit", "dpo-pairs_verify/results/v7_base_errors.json", ["P0/B_approve", "commit[1]"], 0.769),
    ("dpo_commit_pushed_down_A", "dpo-pairs_verify/results/v11_missed_checks.json", "commit_type_push/A:commit_rejected", 194),
    ("dpo_commit_pushed_down_B", "dpo-pairs_verify/results/v11_missed_checks.json", "commit_type_push/B_approve:commit_rejected", 182),
    ("dpo_commit_pushed_up", "dpo-pairs_verify/results/v11_missed_checks.json", "commit_type_push/B_act:commit_chosen", 18),
    ("dpo_L0r_no_policy", "dpo-pairs_verify/results/v11_missed_checks.json", ["approval_policy_stated_in_system_prompt_by_level(True/False)", "L0r", "false"], 133),
    # --- grader (problem 4) ---
    ("oracle_positive_targets", "grader-agreement/results/oracle_summary.json", "positive_target_disagreements/n_positive_targets", 1601),
    ("oracle_disagreements", "grader-agreement/results/oracle_summary.json", "positive_target_disagreements/n", 0),
    ("dump_vals_ask_grpo", "grader-agreement_verify/results/v_planted_summary.json", "by/train_grpo|dump_vals_ask|A/rclass/ask_informed", 191),
    ("quote_ask_grpo", "grader-agreement/results/keywords_summary.json", "planted/train_grpo|quote_ask|A/reward_class/ask_informed", 194),
    ("judge_precision_A", "grader-agreement/results/judge_agreement_summary.json", "A_messages/precision_kw_vs_judge", 0.8938),
    ("judge_recall_A", "grader-agreement/results/judge_agreement_summary.json", "A_messages/recall_kw_vs_judge", 0.9439),
    ("judge_precision_A_cards", "grader-agreement/results/judge_agreement_summary.json", "A_approval_cards/precision_kw_vs_judge", 0.1786),
    ("empty_card_reward_rows", "grader-agreement_verify/results/v_final_checks.json", "grpo_B_approve_card_rewards/empty_card_reward=1.1", 231),
    ("bact_commits", "grader-agreement_verify/results/v_bact_read_summary.json", "n_commits_on_B_act", 67),
    ("bact_right", "grader-agreement_verify/results/v_bact_read_summary.json", "B_act_right_by_my_reading", 25),
    ("B_success_graded", "grader-agreement_verify/results/v_bact_read_summary.json", "B_success_current_rule", 113),
    ("B_success_read", "grader-agreement_verify/results/v_bact_read_summary.json", "B_success_my_reading", 71),
    ("A_rows_term_in_lookup", "grpo-dynamics_verify/results/v10_extras.json", "A_rows_with_informed_term_in_request_or_lookup_result", 208),
    ("A_approve_informed_echo", "grpo-dynamics_verify/results/v10_extras.json", "A_approve_informed_samples_on_prompts_with_term_in_turns", 74),
    # --- evaluation side (problem 5) ---
    ("test_clusters", "splits-eval-alignment_verify/results/v07_eval_alignment.json", "d_clusters/test/source_post_clusters", 13),
    ("hard_test_rows_in_test_clusters", "splits-eval-alignment_verify/results/v07_eval_alignment.json", "d_clusters/hard_test_rows_sharing_a_cluster_with_test", 26),
    ("val_clusters", "splits-eval-alignment_verify/results/v07_eval_alignment.json", "d_clusters/val/source_post_clusters", 5),
    ("identical_first_prompt_test_L1", "splits-eval-alignment_verify/results/v02_l2s_reveal.json", "per_split/test/pairs_with_identical_first_prompt_P0/L1", 18),
    ("identical_first_prompt_test_L2s", "splits-eval-alignment_verify/results/v02_l2s_reveal.json", "per_split/test/pairs_with_identical_first_prompt_P0/L2s", 0),
    ("identical_first_prompt_hard_L2", "splits-eval-alignment_verify/results/v02_l2s_reveal.json", "per_split/hard_test/pairs_with_identical_first_prompt_P0/L2", 9),
    ("crossrun_changed_rules", "realism-pressure_verify/results/v8_crossrun_specs.json", "pairs_with_changed_tool_rules", 424),
    ("crossrun_pairs", "realism-pressure_verify/results/v8_crossrun_specs.json", "n_pairs", 652),
    ("hardset_proxy", "realism-pressure_verify/results/v11_hardset_sensitivity.json", "proxy_hard_as_logged", 81),
    ("hardset_proxy_wo_empty", "realism-pressure_verify/results/v11_hardset_sensitivity.json", "proxy_hard_excluding_needless_after_empty", 57),
    ("preamble_tokens", "splits-eval-alignment_verify/results/v10_preamble.json", "estimated_preamble_tokens_incl_specials", 384),
    ("preamble_ask_lines", "splits-eval-alignment_verify/results/v10_preamble.json", "lines_matching_ask_clarify_followup_question_confirm_check", []),
    # --- twin nuisances (problem 7) ---
    ("request_differs_pairs", "twin-contrast-shortcuts_verify/results/v1_twins.json", "request_differs_train_pairs", 31),
    ("request_identifier_only_pairs", "twin-contrast-shortcuts_verify/results/v1_twins.json", "request_kind_train_pairs/identifier_only", 13),
    ("request_wording_pairs", "twin-contrast-shortcuts_verify/results/v1_twins.json", "request_kind_train_pairs/wording", 18),
    ("same_setting_share", "twin-contrast-shortcuts_verify/results/v1_twins.json", "same_prompt_setting_share", 0.543),
    ("P2_share_A", "grpo-dynamics_verify/results/v11_checks.json", "P2_share_by_twin_whole_pool/A", 0.2794),
    ("P2_share_B", "grpo-dynamics_verify/results/v11_checks.json", "P2_share_by_twin_whole_pool/B", 0.3293),
    # --- prompt interference (section 3) ---
    ("m1_contradicted_vs_other_pp", "prompt-conflicts_verify/results/02_pc1.json", "m1_bizops_contradicted_vs_other_A_L1plus_P0_ask_informed/diff_pp", 15.1),
    ("base_hint_A_ask_P0", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_eval_P0_vs_P2_paired/A/ask_any_P0", 53),
    ("base_hint_A_ask_P2", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_eval_P0_vs_P2_paired/A/ask_any_P2", 70),
    ("base_hint_B_ask_P0", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_eval_P0_vs_P2_paired/B/ask_any_P0", 42),
    ("base_hint_B_ask_P2", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_eval_P0_vs_P2_paired/B/ask_any_P2", 61),
    ("base_hint_pairs", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_eval_P0_vs_P2_paired/A/pairs", 343),
    ("base_B_cards_L0r_P0", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_grid_P0/B|L0r/card", 2),
    ("base_B_cards_L1_P0", "prompt-conflicts_verify/results/07_pc9_checks.json", "base_grid_P0/B|L1/card", 0),
    ("m1_A_L1", "prompt-conflicts_verify/results/07_pc9_checks.json", "m1_correct_rate_by_twin_level/A|L1[2]", 0.518),
    ("m1_A_L2", "prompt-conflicts_verify/results/07_pc9_checks.json", "m1_correct_rate_by_twin_level/A|L2[2]", 0.586),
    ("m1_B_L1", "prompt-conflicts_verify/results/07_pc9_checks.json", "m1_correct_rate_by_twin_level/B|L1[2]", 0.513),
    ("m1_B_L2", "prompt-conflicts_verify/results/07_pc9_checks.json", "m1_correct_rate_by_twin_level/B|L2[2]", 0.608),
    ("card_rule_bills_L0r", "prompt-conflicts_verify/results/08_extra.json", "A_approve_informed_env_level/bills|L0r/rate", 0.312),
    ("card_rule_bills_L1plus", "prompt-conflicts_verify/results/08_extra.json", "A_approve_informed_env_level/bills|L1+/rate", 0.4),
    ("card_rule_shopping_L0r", "prompt-conflicts_verify/results/08_extra.json", "A_approve_informed_env_level/shopping|L0r/rate", 0.179),
    ("card_rule_shopping_L1plus", "prompt-conflicts_verify/results/08_extra.json", "A_approve_informed_env_level/shopping|L1+/rate", 0.179),
    ("rp3_opus", "realism-pressure_verify/results/v2_summary.json", "RP3_panel_hard_L0rP2_to_L2P2/claude-opus/D_ask/diff", -0.3111),
    ("rp3_sonnet", "realism-pressure_verify/results/v2_summary.json", "RP3_panel_hard_L0rP2_to_L2P2/claude-sonnet/D_ask/diff", -0.2667),
    ("rp3_haiku", "realism-pressure_verify/results/v2_summary.json", "RP3_panel_hard_L0rP2_to_L2P2/claude-haiku/D_ask/diff", -0.3333),
    ("rp3_fable", "realism-pressure_verify/results/v2_summary.json", "RP3_panel_hard_L0rP2_to_L2P2/claude-fable/D_ask/diff", -0.1778),
    ("train_rows_empty_tool_result", "realism-pressure/results/unscripted.json", "training_views/sft_train.jsonl/rows_with_empty_tool_result", 0),
    # --- SFT diagnosis extras (section 4) ---
    ("selector_ambiguous_share", "composition-memorization_verify/results/v3_dupes.json", "sft_train/selector/share_tok", 0.0483),
    ("loss_if_decision_perfect", "grpo-dynamics_verify/results/v02_loss_share.json", "gold_class/loss_if_decision_were_perfect_epoch2", 0.1999),
    ("decision_share_epoch2_loss", "grpo-dynamics_verify/results/v02_loss_share.json", "gold_class/share_of_epoch2_mean_loss", 0.0535),
    ("val_untrained_lookup_rows", "composition-memorization_verify/results/v5_val.json", "val_lookup_rows_untrained_tool", 52),
    ("val_4gram_cov", "composition-memorization_verify/results/v5_val.json", "val_4gram_cov_by_train_all", 0.1383),
    ("train_4gram_cov_loco", "composition-memorization_verify/results/v5_val.json", "train_4gram_cov_leave_one_cluster_out", 0.1405),
    ("train_4gram_cov_lopo", "composition-memorization_verify/results/v5_val.json", "train_4gram_cov_leave_one_pair_out", 0.3783),
]


def get(obj, path):
    # path: "a/b[0]/c" or, when a key itself contains '/', a list of key parts ["P0/A", "commit[1]"]
    for part in (path if isinstance(path, list) else path.split("/")):
        m = re.fullmatch(r"(.*?)((?:\[\d+\])*)", part)
        key, idx = m.group(1), m.group(2)
        if key != "":
            if not isinstance(obj, dict) or key not in obj:
                raise KeyError(path)
            obj = obj[key]
        for i in re.findall(r"\[(\d+)\]", idx):
            obj = obj[int(i)]
    return obj


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= max(1e-6, 1e-4 * abs(float(b)))
    return a == b


out, bad = [], 0
cache = {}
for cid, rel, path, expected in CHECKS:
    p = os.path.join(ROOT, rel)
    try:
        if p not in cache:
            cache[p] = json.load(open(p))
        val = get(cache[p], path)
        ok = same(val, expected)
    except Exception as e:  # missing file or key
        val, ok = f"ERROR {type(e).__name__}: {e}", False
    bad += not ok
    out.append({"id": cid, "artifact": f"experiments/dataset_audit/{rel}",
                "key": path if isinstance(path, str) else " | ".join(path), "value": val,
                "cited": expected, "ok": ok})
    if not ok:
        print(f"MISMATCH {cid}: {rel} :: {path} -> {val!r} (cited {expected!r})")

os.makedirs(os.path.join(ROOT, "lead", "results"), exist_ok=True)
dst = os.path.join(ROOT, "lead", "results", "headline_numbers.json")
json.dump({"n_checks": len(out), "n_ok": len(out) - bad, "checks": out}, open(dst, "w"), indent=1)
print(f"{len(out) - bad}/{len(out)} cited numbers match their artifacts -> {dst}")
sys.exit(1 if bad else 0)
