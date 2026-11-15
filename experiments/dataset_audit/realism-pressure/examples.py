"""Concrete row examples for each finding (file + 0-based line index + the decisive raw output, truncated).

Reads results/episode_indicators.csv, results/m1_samples.csv, results/hint_crossrun_rows.csv (written by the other scripts).
Output: results/examples.json
Run: python3 -B experiments/dataset_audit/realism-pressure/examples.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

from common import GRPO, RES, ROOT, SCORES, dump, first_text


def flips(ep, family, model, twin, a, b, dec_a, dec_b, k=3):
    """scenarios whose first decision is in dec_a at cell a and in dec_b at cell b (same family/model)."""
    g = ep[(ep.family == family) & (ep.model == model) & (ep.twin == twin)]
    xa = g[(g.level == a[0]) & (g.prompt == a[1]) & g.decision.isin(dec_a)].set_index("id")
    xb = g[(g.level == b[0]) & (g.prompt == b[1]) & g.decision.isin(dec_b)].set_index("id")
    out = []
    for sid in sorted(set(xa.index) & set(xb.index))[:k]:
        ra, rb = xa.loc[sid], xb.loc[sid]
        out.append({"scenario_id": sid, "model": model,
                    "from": {"cell": f"{a[0]}/{a[1]}", "decision": ra.decision, "ref": f"{ra.file}:{ra.line}",
                             "text": first_text(ra.file, int(ra.line))},
                    "to": {"cell": f"{b[0]}/{b[1]}", "decision": rb.decision, "ref": f"{rb.file}:{rb.line}",
                           "text": first_text(rb.file, int(rb.line))}})
    return out


def same_decision_all_levels(ep, family, model, twin, prompt, levels, decs, k=3):
    g = ep[(ep.family == family) & (ep.model == model) & (ep.twin == twin) & (ep.prompt == prompt)]
    ok = None
    for lv in levels:
        ids = set(g[(g.level == lv) & g.decision.isin(decs)].id)
        ok = ids if ok is None else ok & ids
    out = []
    for sid in sorted(ok)[:k]:
        rows = g[g.id == sid].set_index("level")
        out.append({"scenario_id": sid, "model": model, "prompt": prompt,
                    "cells": {lv: {"decision": rows.loc[lv].decision, "ref": f"{rows.loc[lv].file}:{rows.loc[lv].line}",
                                   "text": first_text(rows.loc[lv].file, int(rows.loc[lv].line), 140)} for lv in levels}})
    return out


def m1_mixed(m1, level, twin, k=2):
    pp = m1[(m1.level == level) & (m1.twin == twin)].groupby("grpo_line")
    out = []
    for gl, g in pp:
        if 3 <= g.success.sum() <= 5:
            out.append({"grpo_line": int(gl), "scenario_id": g.scenario_id.iloc[0], "level": level,
                        "prompt_setting": g.prompt_setting.iloc[0], "sft_line": int(g.sft_line.iloc[0]),
                        "sft_target_decision": g.target_decision.iloc[0],
                        "sample_decisions": g.decision.value_counts().to_dict(),
                        "scores_lines": [int(x) for x in g.score_line]})
        if len(out) >= k:
            break
    return out


def main():
    ep = pd.read_csv(os.path.join(RES, "episode_indicators.csv"))
    m1 = pd.read_csv(os.path.join(RES, "m1_samples.csv"))
    hx = pd.read_csv(os.path.join(RES, "hint_crossrun_rows.csv"))
    APPROVE = ["approve_generic", "approve_informed", "approve_gold"]
    COMMIT = ["commit_gold", "commit_other", "commit_harmful"]
    ASK = ["ask_generic", "ask_informed"]
    ex = {}
    ex["frontier_P2_realistic_needless_ask_on_B"] = (
        flips(ep, "panel_hard_v12", "claude-opus", "B", ("L0r", "P2"), ("L2", "P2"), APPROVE, ASK, 2)
        + flips(ep, "panel_hard_v12", "claude-sonnet", "B", ("L0r", "P2"), ("L2", "P2"), APPROVE, ASK, 1))
    ex["production_prompt_more_asking_P0"] = (
        flips(ep, "variants_v1", "DeepSeek-V3.2", "B", ("L0r", "P0"), ("L2", "P0"), APPROVE + COMMIT, ASK, 2)
        + flips(ep, "variants_v1", "DeepSeek-V3.2", "A", ("L0r", "P0"), ("L2", "P0"), APPROVE + COMMIT, ASK, 1))
    ex["toy_prompt_lacks_approval_policy_B_commit_at_L0r"] = flips(
        ep, "seed_v11_open", "aya-expanse-32b", "B", ("L0r", "P0"), ("L1", "P0"), COMMIT, APPROVE, 3)
    ex["base_tiny_aya_literal_at_every_level"] = same_decision_all_levels(
        ep, "eval_v12", "tiny-aya-global-local", "A", "P0", ["L0r", "L1", "L2"], COMMIT + ["approve_generic"], 3)
    ex["base_tiny_aya_informed_ask_lost_at_L2"] = flips(
        ep, "eval_v12", "tiny-aya-global-local", "A", ("L0r", "P0"), ("L2", "P0"), ["ask_informed", "tell_informed"],
        COMMIT + APPROVE + ["ask_generic", "statement"], 3)
    ex["P2_hint_generic_asks_tiny_aya"] = flips(
        ep, "eval_v12", "tiny-aya-global-local", "B", ("L0r", "P0"), ("L0r", "P2"), COMMIT + APPROVE, ASK, 3)
    ex["burial_L2s_vs_L2_qwen"] = flips(
        ep, "variants_v1", "Qwen3-235B-A22B-Instruct-2507", "A", ("L2s", "P1"), ("L2", "P1"),
        ["ask_informed", "tell_informed"], ["ask_generic", "statement"] + APPROVE + COMMIT, 3)
    q = hx[(hx.model == "Qwen3-235B-A22B-Instruct-2507") & (hx.twin == "B") & (hx.from_prompt == "P0") &
           hx.dec_from.isin(APPROVE) & hx.dec_to.isin(ASK)].head(3)
    ex["P2_hint_needless_ask_qwen_L2"] = [{"scenario_id": r.id, "L2/P0": {"decision": r.dec_from, "ref": r.from_ref},
                                           "L2/P2": {"decision": r.dec_to, "ref": r.to_ref,
                                                     "text": first_text(r.to_ref.rsplit(":", 1)[0], int(r.to_ref.rsplit(":", 1)[1]))}}
                                          for r in q.itertuples()]
    ex["m1_mixed_decisions_on_own_training_prompt"] = (m1_mixed(m1, "L0r", "A", 1) + m1_mixed(m1, "L2", "A", 1)
                                                       + m1_mixed(m1, "L2", "B", 1))
    # one completion text per M1 example (from scores we only have the class; the GRPO completions are not logged)
    ex["notes"] = ["M1 completions are not stored in scores.jsonl (only the grader class), so M1 examples cite the scores "
                   "lines + the grpo_train/sft_train line of the identical prompt",
                   f"panel_* refs point into the session scratchpad (not in the repo): {os.path.dirname(GRPO)} is in-repo"]
    dump(ex, "examples.json")
    print(json.dumps({k: len(v) for k, v in ex.items()}, indent=1))


if __name__ == "__main__":
    main()
