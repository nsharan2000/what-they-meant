"""Part 3: the M1 (SFT) policy's own decisions on its TRAINING decision prompts, by level x prompt setting x twin.

Source: the 960 GRPO rollouts (30 steps x 4 prompts x 8 samples, temperature 1.0, top_p 1.0; policy = merged M1 + the GRPO
LoRA, whose weight change is 1.8 % of the SFT adapter's -> treated as M1 samples). components.decision is the grader class
(wtm/reward.py decision_of -> wtm/episode.py classify_decision; 'lookup' = another look-up instead of a decision).
Join: (scenario_id, level) is unique in grpo_train.jsonl -> prompt_setting, scenario spec, tool kinds, prompt length
(token_stats.jsonl, exact Tiny Aya tokens). The SFT target for the identical prompt (sft_train.jsonl, turn 'decision') is
classified with the same grader, so each sample can be compared with the target it was trained on.
Units: a prompt (8 samples). Bootstrap = resample prompts (10,000). Level contrasts between different prompts are unpaired;
the paired version uses the scenarios sampled at both levels.
Outputs: results/m1_samples.csv, results/m1_level_table.csv, results/m1_levels.json
Run: python3 -B experiments/dataset_audit/realism-pressure/m1_levels.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from common import (GRPO, RES, ROOT, SCORES, SFT, TOKSTATS, boot_diff_unpaired, boot_mean_ci, dump, indicators,
                    pair_cluster_map, cluster_of)

sys.path.insert(0, ROOT)
from wtm.episode import decision_grade  # noqa: E402
from wtm.reward import decision_of      # noqa: E402

COARSE = {"ask_informed": "ask", "ask_generic": "ask", "tell_informed": "tell", "statement": "statement",
          "approve_informed": "approve", "approve_generic": "approve", "approve_gold": "approve",
          "commit_gold": "commit", "commit_other": "commit", "commit_harmful": "commit", "lookup": "lookup",
          "invalid": "invalid"}


def load():
    grpo = {}
    with open(GRPO) as fh:
        for i, l in enumerate(fh):
            r = json.loads(l)
            grpo[(r["scenario_id"], r["level"])] = (i, r)
    sft_target = {}
    with open(SFT) as fh:
        for i, l in enumerate(fh):
            r = json.loads(l)
            if r["turn"] == "decision":
                sft_target[json.dumps(r["prompt"], sort_keys=True)] = (i, r["completion"][0]["content"])
    tok = {}
    with open(TOKSTATS) as fh:
        for l in fh:
            t = json.loads(l)
            if t["file"] == "grpo_train.jsonl":
                tok[t["i"]] = t["prompt_tokens"]
    scores = [json.loads(l) for l in open(SCORES)]
    return grpo, sft_target, tok, scores


def main():
    grpo, sft_target, tok, scores = load()
    p2c = pair_cluster_map()
    recs = []
    target_cache = {}
    for j, s in enumerate(scores):
        m = s["meta"]
        gi, g = grpo[(m["scenario_id"], m["level"])]
        scn = json.loads(g["scenario"]) if isinstance(g["scenario"], str) else g["scenario"]
        kinds = json.loads(g["tool_kinds"]) if isinstance(g["tool_kinds"], str) else g["tool_kinds"]
        if gi not in target_cache:
            si, tgt = sft_target[json.dumps(g["prompt"], sort_keys=True)]
            tdec, _ = decision_of(tgt, scn, kinds)
            target_cache[gi] = (si, tdec)
        si, tdec = target_cache[gi]
        dec = s["components"]["decision"]
        tw = "A" if g["twin"].startswith("A") else "B"
        succ = (dec not in ("lookup", "invalid")) and decision_grade(scn, dec)["success"]
        rec = {"score_line": j, "grpo_line": gi, "sft_line": si, "scenario_id": m["scenario_id"], "pair_id": g["pair_id"],
               "cluster": cluster_of(g["pair_id"], p2c), "level": m["level"], "prompt_setting": g["prompt_setting"],
               "twin": tw, "expected": scn["expected"]["decision"], "step": m["step"], "decision": dec,
               "coarse": COARSE.get(dec, dec), "target_decision": tdec, "target_coarse": COARSE.get(tdec, tdec),
               "prompt_tokens": tok[gi], "n_tokens_out": m.get("n_tokens"), "score": s["score"]}
        rec.update(indicators(dec, tw, succ))
        rec["matches_target_class"] = int(dec == tdec)
        rec["matches_target_coarse"] = int(COARSE.get(dec, dec) == COARSE.get(tdec, tdec))
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RES, "m1_samples.csv"), index=False)

    mets = {"A": ["ask", "ask_informed", "informed_chat", "approve", "commit", "literal_act", "other", "success",
                  "matches_target_coarse"],
            "B": ["ask", "needless", "approve", "commit", "act", "other", "success", "matches_target_coarse"]}
    # per-prompt aggregation (8 samples per prompt)
    keys = ["grpo_line", "scenario_id", "cluster", "level", "prompt_setting", "twin", "prompt_tokens", "target_decision"]
    per_prompt = df.groupby(keys).mean(numeric_only=True).reset_index()
    per_prompt["n_samples"] = df.groupby(keys).size().values
    assert (per_prompt.n_samples == 8).all()

    out = {"n_samples": int(len(df)), "n_prompts": int(len(per_prompt)), "cells": [], "level_contrasts": [],
           "paired_level_contrasts": [], "token_effect": [], "step_drift": [], "target_classes": {}, "notes": []}
    table = []
    for (lv, ps, tw), g in per_prompt.groupby(["level", "prompt_setting", "twin"]):
        rec = {"level": lv, "prompt_setting": ps, "twin": tw, "n_prompts": int(len(g)), "n_samples": int(g.n_samples.sum()),
               "median_prompt_tokens": int(g.prompt_tokens.median())}
        for mname in mets[tw]:
            mu, lo, hi = boot_mean_ci(g[mname].to_numpy(), None, seed=3)
            rec[mname] = round(mu, 4)
            rec[mname + "_ci"] = [round(lo, 4), round(hi, 4)]
        table.append(rec)
    # pooled over prompt settings, by level
    for (lv, tw), g in per_prompt.groupby(["level", "twin"]):
        rec = {"level": lv, "prompt_setting": "P0+P2", "twin": tw, "n_prompts": int(len(g)),
               "n_samples": int(g.n_samples.sum()), "median_prompt_tokens": int(g.prompt_tokens.median())}
        for mname in mets[tw]:
            mu, lo, hi = boot_mean_ci(g[mname].to_numpy(), None, seed=4)
            rec[mname] = round(mu, 4)
            rec[mname + "_ci"] = [round(lo, 4), round(hi, 4)]
        table.append(rec)
    for (ps, tw), g in per_prompt.groupby(["prompt_setting", "twin"]):
        rec = {"level": "all", "prompt_setting": ps, "twin": tw, "n_prompts": int(len(g)),
               "n_samples": int(g.n_samples.sum()), "median_prompt_tokens": int(g.prompt_tokens.median())}
        for mname in mets[tw]:
            mu, lo, hi = boot_mean_ci(g[mname].to_numpy(), None, seed=5)
            rec[mname] = round(mu, 4)
            rec[mname + "_ci"] = [round(lo, 4), round(hi, 4)]
        table.append(rec)
    tab = pd.DataFrame(table)
    tab.to_csv(os.path.join(RES, "m1_level_table.csv"), index=False)
    out["cells"] = table

    # unpaired level contrasts vs L0r (pooled P0+P2 and per prompt setting); same metrics as the base-model contrasts
    key_mets = {"A": ["ask", "informed_chat", "literal_act", "success"], "B": ["approve", "act", "needless", "success"]}
    for ps in ("P0", "P2", "P0+P2"):
        sub = per_prompt if ps == "P0+P2" else per_prompt[per_prompt.prompt_setting == ps]
        for tw in ("A", "B"):
            base = sub[(sub.level == "L0r") & (sub.twin == tw)]
            for lv in ("L1", "L2", "L3"):
                other = sub[(sub.level == lv) & (sub.twin == tw)]
                if len(base) < 3 or len(other) < 3:
                    continue
                for mname in key_mets[tw]:
                    d, lo, hi = boot_diff_unpaired(base[mname].to_numpy(), other[mname].to_numpy(), seed=6)
                    out["level_contrasts"].append({"prompt_setting": ps, "twin": tw, "from": "L0r", "to": lv,
                                                   "metric": mname, "n_from": int(len(base)), "n_to": int(len(other)),
                                                   "rate_from": round(base[mname].mean(), 4),
                                                   "rate_to": round(other[mname].mean(), 4),
                                                   "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]})
    # paired: scenarios sampled at L0r and at a realistic level (same scenario, possibly different prompt setting)
    for tw in ("A", "B"):
        sub = per_prompt[per_prompt.twin == tw]
        piv = {lv: x.set_index("scenario_id") for lv, x in sub.groupby("level")}
        for lv in ("L1", "L2", "L3"):
            if "L0r" not in piv or lv not in piv:
                continue
            ids = sorted(set(piv["L0r"].index) & set(piv[lv].index))
            if len(ids) < 3:
                continue
            for mname in key_mets[tw]:
                dvec = piv[lv].loc[ids, mname].to_numpy() - piv["L0r"].loc[ids, mname].to_numpy()
                mu, lo, hi = boot_mean_ci(dvec, None, seed=7)
                out["paired_level_contrasts"].append({"twin": tw, "from": "L0r", "to": lv, "metric": mname, "n": len(ids),
                                                      "diff": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)],
                                                      "ids": ids})
    # prompt length: within the realistic levels, do longer prompts (> 4096 tokens = beyond the sliding window) fail more?
    for tw in ("A", "B"):
        sub = per_prompt[(per_prompt.twin == tw) & (per_prompt.level.isin(["L1", "L2", "L3"]))]
        short = sub[sub.prompt_tokens <= 4096]
        long_ = sub[sub.prompt_tokens > 4096]
        for mname in key_mets[tw]:
            if len(short) >= 3 and len(long_) >= 3:
                d, lo, hi = boot_diff_unpaired(short[mname].to_numpy(), long_[mname].to_numpy(), seed=8)
                out["token_effect"].append({"twin": tw, "levels": "L1+L2+L3", "metric": mname, "n_le4096": int(len(short)),
                                            "n_gt4096": int(len(long_)), "rate_le4096": round(short[mname].mean(), 4),
                                            "rate_gt4096": round(long_[mname].mean(), 4), "diff": round(d, 4),
                                            "ci": [round(lo, 4), round(hi, 4)]})
        allp = per_prompt[per_prompt.twin == tw]
        r = np.corrcoef(allp.prompt_tokens, allp.success)[0, 1]
        out["token_effect"].append({"twin": tw, "levels": "all", "metric": "pearson_r(prompt_tokens, success)",
                                    "value": round(float(r), 4), "n": int(len(allp))})
    # step drift (GRPO adapter): success by step tercile
    df["tercile"] = pd.cut(df.step, [-1, 9, 19, 29], labels=["0-9", "10-19", "20-29"])
    for (tw, te), g in df.groupby(["twin", "tercile"], observed=True):
        out["step_drift"].append({"twin": tw, "steps": str(te), "n": int(len(g)), "success": round(g.success.mean(), 4)})
    # the SFT targets of the sampled prompts and how often samples reproduce the target's class
    for tw in ("A", "B"):
        x = df[df.twin == tw]
        out["target_classes"][tw] = {
            "target_class_counts_per_prompt": dict(Counter(per_prompt[per_prompt.twin == tw].target_decision)),
            "sample_class_counts": dict(Counter(x.decision)),
            "match_target_exact_class": round(x.matches_target_class.mean(), 4),
            "match_target_coarse_class": round(x.matches_target_coarse.mean(), 4),
            "per_prompt_success_hist": {str(k): int(v) for k, v in
                                        Counter((per_prompt[per_prompt.twin == tw].success * 8).round().astype(int)).items()},
        }
    out["notes"].append("all 120 sampled prompts are byte-identical to SFT decision prompts (trained 2 epochs)")
    out["notes"].append("samples at temperature 1.0: rates estimate the policy's class probabilities, not greedy decisions")
    dump(out, "m1_levels.json")
    print(tab[["level", "prompt_setting", "twin", "n_prompts"] + [c for c in tab.columns if c in
                                                                 ("ask", "informed_chat", "literal_act", "approve", "act",
                                                                  "needless", "success")]].to_string(index=False))


if __name__ == "__main__":
    main()
