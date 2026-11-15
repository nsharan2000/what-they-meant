"""V3 (verifier): the M1 policy's sampled decisions during GRPO (scores.jsonl, 960 = 120 prompts x 8, temperature 1.0 per
training/train_grpo.py) re-counted from scratch: classes by twin, ask/correct rates, twin discrimination overall / by request
stratum / by difference locus / by prompt setting, plus two checks the analyst did not make:
  (1) are the sampled prompts byte-identical to SFT decision prompts that M1 was trained on (with the same target)?
  (2) per-prompt MAJORITY over the 8 samples (a proxy for greedy decoding, which the evaluation uses: wtm/run.py temperature 0.0).

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v3_m1.py -> results/v3_m1.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_twins import req_kind  # noqa: E402
from vcommon import ROOT, T, all_specs, curated, jl, parts, save  # noqa: E402

from wtm.episode import decision_grade  # noqa: E402
from wtm.labels import adjudicate  # noqa: E402

MSG = {"ask_informed", "ask_generic", "tell_informed", "statement"}


def main():
    out = {}
    sc = jl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl"))
    out["n_samples"] = len(sc)
    grpo = {(r["scenario_id"], r["level"]): r for r in curated("grpo_train")}
    sft = curated("sft_train")
    sft_dec = {(r["scenario_id"], r["level"]): r for r in sft if r["turn"] == "decision"}
    specs = all_specs()
    per = defaultdict(list)
    steps = defaultdict(set)
    for s in sc:
        m = s["meta"]
        per[(m["scenario_id"], m["level"])].append(s["components"]["decision"])
        steps[(m["scenario_id"], m["level"])].add(m["step"])
    out["n_prompts"] = len(per)
    out["samples_per_prompt"] = dict(Counter(len(v) for v in per.values()))
    out["prompts_repeated_across_steps"] = sum(len(v) > 1 for v in steps.values())
    cls = {"A": Counter(), "B": Counter()}
    prompts = []
    in_sft = same_prompt = 0
    missing = []
    for (sid, lv), decs in per.items():
        g = grpo[(sid, lv)]
        t = T(g)
        cls[t].update(decs)
        scn = adjudicate(json.loads(g["scenario"]))
        corr = [decision_grade(scn, d)["success"] if d not in ("lookup", "invalid") else False for d in decs]
        s_row = sft_dec.get((sid, lv))
        if s_row is not None:
            in_sft += 1
            same_prompt += s_row["prompt"] == g["prompt"]
        else:
            missing.append([sid, lv])
        maj = Counter(decs).most_common(1)[0][0]
        prompts.append({"sid": sid, "pair": g["pair_id"], "level": lv, "ps": g["prompt_setting"], "t": t, "n": len(decs),
                        "ask": float(np.mean([d in MSG for d in decs])), "correct": float(np.mean(corr)),
                        "maj_correct_share": float(np.mean(corr) > 0.5), "maj_class": maj,
                        "decs": dict(Counter(decs))})
    out["decision_class_by_twin"] = {t: dict(c.most_common()) for t, c in cls.items()}
    out["sampled_prompts_in_sft_train_decision_rows"] = in_sft
    out["sampled_prompts_identical_to_sft_prompt"] = same_prompt
    out["sampled_prompts_missing_from_sft"] = missing
    for t in "AB":
        sel = [p for p in prompts if p["t"] == t]
        out[f"{t}_rates"] = {"prompts": len(sel), "samples": sum(p["n"] for p in sel),
                             "ask_rate_samples": round(sum(p["ask"] * p["n"] for p in sel) / sum(p["n"] for p in sel), 3),
                             "correct_rate_samples": round(sum(p["correct"] * p["n"] for p in sel) / sum(p["n"] for p in sel), 3),
                             "prompts_majority_correct": round(float(np.mean([p["correct"] > 0.5 for p in sel])), 3),
                             "prompts_correct_ge_6of8": round(float(np.mean([p["correct"] >= 0.75 for p in sel])), 3),
                             "prompts_correct_le_2of8": round(float(np.mean([p["correct"] <= 0.25 for p in sel])), 3),
                             "prompts_majority_ask": round(float(np.mean([p["ask"] > 0.5 for p in sel])), 3)}
    out["discrimination_samples"] = round(out["A_rates"]["ask_rate_samples"] - out["B_rates"]["ask_rate_samples"], 3)
    out["discrimination_majority"] = round(out["A_rates"]["prompts_majority_ask"] - out["B_rates"]["prompts_majority_ask"], 3)
    # distribution of per-prompt correct share (8 samples)
    out["per_prompt_correct_hist"] = {t: dict(sorted(Counter(int(round(p["correct"] * p["n"])) for p in prompts if p["t"] == t).items()))
                                      for t in "AB"}

    # strata: request kind (spec-level), locus (A vs B grpo rows at the same level), prompt setting
    byp = defaultdict(dict)
    for s in specs.values():
        if T(s) in "AB":
            byp[s["pair_id"]][T(s)] = s
    rk = {p: req_kind(d["A"]["user_message"], d["B"]["user_message"]) for p, d in byp.items() if "A" in d and "B" in d}
    gm = defaultdict(dict)
    for (sid, lv), g in grpo.items():
        if T(g) in "AB":
            gm[(g["pair_id"], lv)][T(g)] = g
    locus = {}
    for k, d in gm.items():
        if "A" in d and "B" in d:
            pa, pb = parts(d["A"]["prompt"]), parts(d["B"]["prompt"])
            req = "req_same" if pa[1] == pb[1] else ("req_id" if req_kind(pa[1], pb[1]) == "identifier_only" else "req_word")
            res = "res_same" if pa[3] == pb[3] else "res_diff"
            call = "call_same" if pa[2] == pb[2] else "call_diff"
            locus[k] = f"{req}+{res}+{call}"

    def strat(key):
        tab = {}
        for v in sorted({key(p) for p in prompts}):
            a = [p for p in prompts if key(p) == v and p["t"] == "A"]
            b = [p for p in prompts if key(p) == v and p["t"] == "B"]
            e = {"nA": len(a), "nB": len(b)}
            if a:
                e["askA"] = round(float(np.mean([p["ask"] for p in a])), 3)
                e["askA_majority"] = round(float(np.mean([p["ask"] > 0.5 for p in a])), 3)
            if b:
                e["askB"] = round(float(np.mean([p["ask"] for p in b])), 3)
                e["askB_majority"] = round(float(np.mean([p["ask"] > 0.5 for p in b])), 3)
            if a and b:
                e["disc"] = round(e["askA"] - e["askB"], 3)
                e["disc_majority"] = round(e["askA_majority"] - e["askB_majority"], 3)
                e["pairs_A"] = sorted({p["pair"] for p in a})
                e["pairs_B"] = sorted({p["pair"] for p in b})
            tab[v] = e
        return tab
    out["by_request_kind"] = strat(lambda p: rk.get(p["pair"], "?"))
    out["by_locus"] = strat(lambda p: locus.get((p["pair"], p["level"]), "unmatched"))
    out["by_prompt_setting"] = strat(lambda p: p["ps"])
    out["by_level"] = strat(lambda p: p["level"])
    # pair-cluster bootstrap of the wording-vs-identical difference in discrimination
    rng = np.random.default_rng(5)
    pairs = sorted({p["pair"] for p in prompts})
    bp = defaultdict(list)
    for p in prompts:
        bp[p["pair"]].append(p)

    def disc(sel, kind):
        a = [p["ask"] for p in sel if rk.get(p["pair"]) == kind and p["t"] == "A"]
        b = [p["ask"] for p in sel if rk.get(p["pair"]) == kind and p["t"] == "B"]
        return np.mean(a) - np.mean(b) if a and b else np.nan
    boots = []
    for _ in range(4000):
        pick = rng.choice(len(pairs), len(pairs))
        sel = [x for k in pick for x in bp[pairs[k]]]
        boots.append(disc(sel, "wording") - disc(sel, "identical"))
    out["wording_minus_identical_disc"] = {"point": round(float(disc(prompts, "wording") - disc(prompts, "identical")), 3),
                                           "ci95": [round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)]}
    # specific example V254 (analyst: B asked 8/8, A 2/8)
    out["V254"] = [p for p in prompts if p["pair"] == "V254"]
    out["prompts"] = prompts
    save(out, "v3_m1.json")
    for k, v in out.items():
        if k not in ("prompts",):
            print(k, v)


if __name__ == "__main__":
    main()
