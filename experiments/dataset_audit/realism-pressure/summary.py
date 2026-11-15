"""Collects every number the findings cite from the saved result files into one place (results/summary_numbers.json).
Run after rates.py, pressure.py, m1_levels.py, m1_vs_base.py, hint_crossrun.py, discrimination.py, unscripted.py, examples.py:
python3 -B experiments/dataset_audit/realism-pressure/summary.py
"""
from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd

from common import RES, dump


def J(name):
    return json.load(open(os.path.join(RES, name)))


def pick(ct, family, model, frm, to, metrics):
    x = ct[(ct.family == family) & (ct.model == model) & (ct["from"] == frm) & (ct.to == to) & ct.metric.isin(metrics)]
    return {r.metric: {"from": r.rate_from, "to": r.rate_to, "diff": r.diff, "ci": [r.ci_lo, r.ci_hi],
                       "cluster_ci": [r.cl_ci_lo, r.cl_ci_hi], "n": int(r.n)} for r in x.itertuples()}


def main():
    ct = pd.read_csv(os.path.join(RES, "contrasts.csv"))
    head = J("pressure_headline.json")["direction_across_models"]
    out = {}
    keep = ["toy->production (L0r->L1) [L0r/P0 -> L1/P0]", "toy->buried (L0r->L2) [L0r/P0 -> L2/P0]",
            "toy->production (L0r->L1) [L0r/P1 -> L1/P1]", "toy->buried (L0r->L2) [L0r/P1 -> L2/P1]",
            "toy->buried (L0r->L2) [L0r/P2 -> L2/P2]", "production->buried (L1->L2) [L1/P0 -> L2/P0]",
            "salient->buried (L2s->L2) [L2s/P1 -> L2/P1]", "length only (L0r->L0p) [L0r/P1 -> L0p/P1]",
            "hint P0->P1 @L0r [L0r/P0 -> L0r/P1]", "hint P0->P1 @L2 [L2/P0 -> L2/P1]"]
    out["direction_across_models"] = {k: {m: {kk: v[m][kk] for kk in ("n_models", "sig_up", "sig_down", "median_diff")}
                                          for m in v} for k, v in head.items() if k in keep}
    mets = ["A_ask", "A_informed", "A_literal", "B_approve", "B_act", "B_needless"]
    ta = "tiny-aya-global-local"
    out["tiny_aya_eval_v12"] = {f"{a}->{b}": pick(ct, "eval_v12", ta, a, b, mets) for a, b in
                                [("L0r/P0", "L1/P0"), ("L0r/P0", "L2/P0"), ("L0r/P2", "L1/P2"), ("L0r/P2", "L2/P2"),
                                 ("L2s/P0", "L2/P0"), ("L2s/P2", "L2/P2"), ("L0r/P0", "L0r/P2"), ("L2/P0", "L2/P2"),
                                 ("L1/P0", "L1/P2")]}
    out["tiny_aya_seed_local"] = {f"{a}->{b}": pick(ct, "seed_v11_local", ta, a, b, mets) for a, b in
                                  [("L0r/P0", "L1/P0"), ("L0r/P0", "L2/P0"), ("L2s/P0", "L2/P0")]}
    out["panel_hard_P2_L0r_to_L2"] = {m: pick(ct, "panel_hard_v12", m, "L0r/P2", "L2/P2", mets) for m in
                                      ["claude-opus", "claude-sonnet", "claude-haiku", "claude-fable", "DeepSeek-V3.2",
                                       "Qwen3-235B-A22B-Instruct-2507"]}
    out["big_models_P0_toy_to_realistic"] = {
        f"{fam}|{m}|{a}->{b}": pick(ct, fam, m, a, b, mets) for fam, m, a, b in [
            ("seed_v11_open", "DeepSeek-V3.2", "L0r/P0", "L1/P0"), ("seed_v11_open", "DeepSeek-V3.2", "L0r/P0", "L2/P0"),
            ("variants_v1", "DeepSeek-V3.2", "L0r/P0", "L2/P0"), ("seed_v11_open", "Qwen3-235B-A22B-Instruct-2507", "L0r/P0", "L1/P0"),
            ("variants_v1", "Qwen3-235B-A22B-Instruct-2507", "L0r/P0", "L2/P0"), ("seed_v11_open", "aya-expanse-32b", "L0r/P0", "L1/P0"),
            ("variants_v1", "aya-expanse-32b", "L0r/P0", "L2/P0"), ("seed_v11_claude", "claude-opus", "L0r/P0", "L2/P0"),
            ("seed_v11_claude", "claude-sonnet", "L0r/P0", "L2/P0"), ("seed_v11_claude", "claude-haiku", "L0r/P0", "L2/P0")]}
    out["burial_salient_to_buried"] = {
        f"{fam}|{m}|{a}->{b}": pick(ct, fam, m, a, b, ["A_informed", "A_literal", "B_needless"]) for fam, m, a, b in [
            ("seed_v11_open", "Qwen3-235B-A22B-Instruct-2507", "L2s/P1", "L2/P1"),
            ("variants_v1", "Qwen3-235B-A22B-Instruct-2507", "L2s/P1", "L2/P1"),
            ("variants_v1", "DeepSeek-V3.2", "L2s/P1", "L2/P1"), ("eval_v12", ta, "L2s/P0", "L2/P0")]}
    hx = J("hint_crossrun.json")
    out["hint_crossrun_L2_P0_to_P2"] = {"render_check": hx["render_check_P0_vs_P2"],
                                        "rows": [r for r in hx["contrasts"] if r["contrast"] == "L2: P0 -> P2" and
                                                 r["metric"] in ("ask", "informed_chat", "needless", "approve")]}
    dc = pd.read_csv(os.path.join(RES, "discrimination_contrasts.csv"))
    dcs = dc[(dc.metric == "D_ask")]
    out["discrimination_D_ask_sig_changes"] = dcs[dcs.sig].to_dict(orient="records")
    dcell = pd.read_csv(os.path.join(RES, "discrimination.csv"))
    out["tiny_aya_D_ask_range"] = {"min": float(dcell[dcell.model.str.contains("tiny-aya-global")].D_ask.min()),
                                   "max": float(dcell[dcell.model.str.contains("tiny-aya-global")].D_ask.max()),
                                   "PP_max": float(dcell[dcell.model.str.contains("tiny-aya-global")].PP.max())}
    disc = J("discrimination.json")
    out["m1_ask_gap_by_level"] = disc["m1_unpaired_gap"]
    out["m1_paired_D"] = disc["m1_paired"]
    m1 = J("m1_levels.json")
    cells = pd.DataFrame(m1["cells"])
    out["m1_success_by_level"] = cells[cells.prompt_setting == "P0+P2"][["level", "twin", "n_prompts", "success",
                                                                           "success_ci"]].to_dict(orient="records")
    out["m1_success_by_setting"] = cells[cells.level == "all"][["prompt_setting", "twin", "n_prompts", "success",
                                                                 "success_ci"]].to_dict(orient="records")
    out["m1_level_contrasts_pooled"] = [r for r in m1["level_contrasts"] if r["prompt_setting"] == "P0+P2"]
    out["m1_token_effect"] = m1["token_effect"]
    out["m1_step_drift"] = m1["step_drift"]
    hist = {tw: m1["target_classes"][tw]["per_prompt_success_hist"] for tw in ("A", "B")}
    out["m1_per_prompt_success_hist"] = hist
    out["m1_share_prompts_uncertain_2to6_of_8"] = {
        tw: round(sum(v for k, v in h.items() if 2 <= int(k) <= 6) / sum(h.values()), 4) for tw, h in hist.items()}
    out["m1_vs_base"] = [r for r in J("m1_vs_base.json")["rows"] if r["metric"] in ("success", "needless", "literal_act")]
    un = J("unscripted.json")
    out["unscripted"] = {"training_views": un["training_views"],
                         "pooled_share_empty_by_level": un["pooled_share_empty_by_level"],
                         "B_needless_by_level_and_empty": un["B_needless_by_level_and_empty"],
                         "decomposition_selected": [r for r in un["decomposition"] if r["metric"] == "needless" and
                                                    r["model"] in ("DeepSeek-V3.2", "Qwen3-235B-A22B-Instruct-2507",
                                                                   "claude-opus", "claude-sonnet", "claude-haiku",
                                                                   "claude-fable")]}
    rates = pd.read_csv(os.path.join(RES, "rates_by_cell.csv"))
    out["coverage"] = {"prompt_settings_per_source": {s: sorted(set(g.prompt)) for s, g in rates.groupby("source")},
                       "levels_per_source": {s: sorted(set(g.level)) for s, g in rates.groupby("source")},
                       "models_with_P2": sorted(set(rates[rates.prompt == "P2"].model))}
    ta_cells = rates[rates.model.str.contains("tiny-aya-global") & (rates.source != "grid_open_seedtools_P0")]
    a, b = ta_cells[ta_cells.twin == "A"], ta_cells[ta_cells.twin == "B"]
    out["tiny_aya_global_ranges_all_cells"] = {
        "sources": sorted(set(ta_cells.source)), "n_cells_per_twin": int(len(a)),
        "A_literal_act": [float(a.literal_act.min()), float(a.literal_act.max())],
        "A_informed_chat": [float(a.informed_chat.min()), float(a.informed_chat.max())],
        "B_approve": [float(b.approve.min()), float(b.approve.max())],
        "B_act": [float(b.act.min()), float(b.act.max())], "B_commit": [float(b.commit.min()), float(b.commit.max())]}
    un2 = J("unscripted.json")
    out["unscripted"]["share_of_asks_after_empty_by_level"] = un2["share_of_asks_after_empty_by_level"]
    out["unscripted"]["B_needless_after_empty_by_model_level"] = [
        r for r in un2["B_needless_after_empty_by_model_level"] if r["level"] in ("L0r", "L2")]
    dr = J("dup_records.json")
    out["dup_records"] = {"spec_level": {g: {k: v for k, v in d.items() if k != "ids"} for g, d in dr["spec_level"].items()},
                          "training_rows_total": {fn: d["total"] for fn, d in dr["training_rows"].items()},
                          "m1_success_by_dup_any": dr["m1_success_by_dup_any"]}
    out["dup_mentions"] = J("dup_mentions.json")["rows"]
    ep = pd.read_csv(os.path.join(RES, "episode_indicators.csv"), usecols=["family", "model", "source"])
    arms = ep.drop_duplicates(["family", "model"])
    out["n_models_and_arms"] = {"distinct_models": int(ep.model.nunique()), "model_x_spec_family_arms": int(len(arms)),
                                "episodes": int(len(ep)), "models": sorted(ep.model.unique()),
                                "panel_rows_per_source": ep[ep.source.str.startswith("panel")].source.value_counts().to_dict()}
    dump(out, "summary_numbers.json")
    print(json.dumps({k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
