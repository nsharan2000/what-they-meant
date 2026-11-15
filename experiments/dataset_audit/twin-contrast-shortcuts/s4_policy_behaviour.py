"""Step 4: do the trained policy (M1 SFT, sampled during GRPO: 960 samples = 120 training decision prompts x 8) and the base
Tiny Aya actually discriminate twins, and does their behaviour follow the twin-contrast cues found in steps 1-3?

  - decision-class mix by twin (reproduces the numbers quoted in the brief);
  - correct / ask rates by pair stratum: request identical vs differs (identifier only / wording), twin-difference loci;
  - twin discrimination = P(ask-like | A) - P(ask-like | B) on pairs where both twins were sampled;
  - within-twin rank correlation of the policy's per-prompt ask rate with the out-of-fold shortcut scores of step 3
    (result_structure, request_bow, length, cue_counts, result_bow): if B prompts that LOOK like A (by a non-causal view) are asked
    about more often, the policy is using that view;
  - the same strata for the base model (experiments/training_eval/results/eval_base_v12.jsonl, Tiny Aya Global, greedy,
    every split, P0/P2, L0r/L1/L2/L2s).
Ask-like = ask_informed | ask_generic | tell_informed | statement. Correct = wtm.episode.decision_grade(...)["success"] on the spec
the reward used (label adjudications applied, wtm.labels.adjudicate).
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s4_policy_behaviour.py -> results/s4_policy_behaviour.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, ROOT, jdump, load_jsonl, load_rows, load_specs, load_split, twin_letter  # noqa: E402
from s1_twin_diff import request_kind, decisive_vocab  # noqa: E402

from wtm.episode import decision_grade  # noqa: E402
from wtm.labels import adjudicate  # noqa: E402

ASK = {"ask_informed", "ask_generic", "tell_informed", "statement"}
APPROVE = {"approve_generic", "approve_informed", "approve_gold"}


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def perm_p_spearman(x, y, n=5000, seed=7):
    """two-sided permutation p-value for Spearman rho."""
    r0 = spearman(x, y)
    if r0 is None:
        return None
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    cnt = 0
    for _ in range(n):
        r = spearman(x, rng.permutation(y))
        if r is not None and abs(r) >= abs(r0) - 1e-12:
            cnt += 1
    return round((cnt + 1) / (n + 1), 4)


def disc_boot(rows, a_key="wording", b_key="identical", n=4000, seed=11):
    """pair-bootstrap of discrimination (ask_A - ask_B) in stratum a_key minus stratum b_key; rows need pair_id, req, t, ask_rate."""
    rng = np.random.default_rng(seed)
    byp = defaultdict(list)
    for r in rows:
        byp[r["pair_id"]].append(r)
    pids = sorted(byp)

    def disc(sel, key):
        a = [r["ask_rate"] for r in sel if r["req"] == key and r["t"] == "A"]
        b = [r["ask_rate"] for r in sel if r["req"] == key and r["t"] == "B"]
        return (np.mean(a) - np.mean(b)) if a and b else np.nan
    point = {k: disc(rows, k) for k in (a_key, b_key)}
    da, db, dd = [], [], []
    for _ in range(n):
        pick = rng.choice(len(pids), len(pids), replace=True)
        sel = [r for k in pick for r in byp[pids[k]]]
        x, y = disc(sel, a_key), disc(sel, b_key)
        da.append(x)
        db.append(y)
        dd.append(x - y)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]  # noqa: E731
    return {f"disc_{a_key}": round(float(point[a_key]), 3), f"disc_{a_key}_ci95": ci(da),
            f"disc_{b_key}": round(float(point[b_key]), 3), f"disc_{b_key}_ci95": ci(db),
            "difference": round(float(point[a_key] - point[b_key]), 3), "difference_ci95": ci(dd),
            "share_boot_difference_le_0": round(float(np.nanmean(np.array(dd) <= 0)), 4), "n_pairs": len(pids)}


def main():
    specs, origin = load_specs()
    split = load_split()
    # pair strata from the specs
    pairs = defaultdict(dict)
    for s in specs.values():
        pairs[s["pair_id"]][twin_letter(s["twin"])] = s
    req_stratum = {}
    for pid, d in pairs.items():
        if "A" in d and "B" in d:
            req_stratum[pid] = request_kind(d["A"]["user_message"], d["B"]["user_message"], decisive_vocab(d["A"], d["B"]))[0]
    # twin-diff loci per (pair, level) from step 1 (grpo_train file)
    loci = {}
    for r in load_jsonl(os.path.join(RES, "s1_twin_diff_rows.jsonl")):
        if r["file"] == "grpo_train":
            loci[(r["pair_id"], r["level"])] = "+".join(r["loci"])
    # shortcut OOF scores (step 3)
    oof = defaultdict(dict)
    for p in load_jsonl(os.path.join(RES, "s3_predictions.jsonl")):
        if p["split"] == "train_oof":
            oof[(p["scenario_id"], p["level"])][p["view"]] = p["p_A_cluster_oof"]
    grpo_rows = {(r["scenario_id"], r["level"]): r for r in load_rows("grpo_train")}

    # ---------------------------------------------------------------- M1 policy samples
    sc = load_jsonl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl"))
    per_prompt = defaultdict(list)
    for s in sc:
        m = s["meta"]
        per_prompt[(m["scenario_id"], m["level"])].append(s["components"]["decision"])
    out = {"n_samples": len(sc), "n_prompts": len(per_prompt)}
    cls = Counter((k[0][-1], d) for k, v in per_prompt.items() for d in v)
    out["decision_class_by_twin"] = {t: dict(sorted(((d, c) for (tt, d), c in cls.items() if tt == t), key=lambda x: -x[1]))
                                     for t in "AB"}
    prompts = []
    for (sid, lv), decs in per_prompt.items():
        g = grpo_rows[(sid, lv)]
        scn = adjudicate(json.loads(g["scenario"]))
        t = twin_letter(scn["twin"])
        correct = [decision_grade(scn, d)["success"] if d not in ("lookup", "invalid") else False for d in decs]
        prompts.append({"scenario_id": sid, "pair_id": scn["pair_id"], "level": lv, "ps": g["prompt_setting"], "t": t,
                        "req": req_stratum.get(scn["pair_id"], "?"), "loci": loci.get((scn["pair_id"], lv), "unmatched"),
                        "ask_rate": float(np.mean([d in ASK for d in decs])),
                        "approve_rate": float(np.mean([d in APPROVE for d in decs])),
                        "correct_rate": float(np.mean(correct)), "n": len(decs), "oof": oof.get((sid, lv), {})})

    def strat(rows, key):
        tab = {}
        for val in sorted({r[key] for r in rows}):
            sel = [r for r in rows if r[key] == val]
            a = [r for r in sel if r["t"] == "A"]
            b = [r for r in sel if r["t"] == "B"]
            tab[val] = {"prompts_A": len(a), "prompts_B": len(b),
                        "ask_rate_A": round(float(np.mean([r["ask_rate"] for r in a])), 3) if a else None,
                        "ask_rate_B": round(float(np.mean([r["ask_rate"] for r in b])), 3) if b else None,
                        "correct_A": round(float(np.mean([r["correct_rate"] for r in a])), 3) if a else None,
                        "correct_B": round(float(np.mean([r["correct_rate"] for r in b])), 3) if b else None}
            if a and b:
                tab[val]["discrimination_askA_minus_askB"] = round(tab[val]["ask_rate_A"] - tab[val]["ask_rate_B"], 3)
        return tab
    out["m1_by_request_stratum"] = strat(prompts, "req")
    out["m1_by_loci"] = strat(prompts, "loci")
    out["m1_by_level"] = strat(prompts, "level")
    out["m1_by_prompt_setting"] = strat(prompts, "ps")
    out["m1_overall"] = strat([dict(r, all="all") for r in prompts], "all")
    # pair-level discrimination where both twins were sampled (any level)
    byp = defaultdict(lambda: {"A": [], "B": []})
    for r in prompts:
        byp[r["pair_id"]][r["t"]].append(r["ask_rate"])
    pl = []
    for pid, d in byp.items():
        if d["A"] and d["B"]:
            pl.append({"pair_id": pid, "req": req_stratum.get(pid), "askA": round(float(np.mean(d["A"])), 3),
                       "askB": round(float(np.mean(d["B"])), 3), "disc": round(float(np.mean(d["A"]) - np.mean(d["B"])), 3)})
    out["m1_pairs_both_twins"] = {"n": len(pl), "mean_discrimination": round(float(np.mean([p["disc"] for p in pl])), 3),
                                  "share_pairs_disc_gt_0.5": round(float(np.mean([p["disc"] > 0.5 for p in pl])), 3),
                                  "share_pairs_disc_le_0": round(float(np.mean([p["disc"] <= 0 for p in pl])), 3),
                                  "by_request_stratum": {k: {"n": len(v), "mean_disc": round(float(np.mean([p["disc"] for p in v])), 3)}
                                                         for k in ("identical", "identifier_only", "wording")
                                                         for v in [[p for p in pl if p["req"] == k]] if v},
                                  "pairs": sorted(pl, key=lambda p: p["disc"])}
    # within-twin association of ask rate with shortcut scores
    assoc = {}
    for view in ("result_structure", "request_bow", "length", "cue_counts", "result_bow", "result_field_values", "full_bow"):
        assoc[view] = {}
        for t in "AB":
            sel = [r for r in prompts if r["t"] == t and view in r["oof"]]
            x = [r["oof"][view] for r in sel]
            y = [r["ask_rate"] for r in sel]
            assoc[view][t] = {"n": len(sel), "spearman_ask_rate_vs_pA": spearman(x, y), "perm_p": perm_p_spearman(x, y)}
    out["m1_within_twin_assoc_with_shortcut_scores"] = assoc
    # direct test of a "negative literal" heuristic: false / null / 0 / empty values in the last tool result
    from common import components, flatten, parse_result
    pol = {}
    for r in prompts:
        g = grpo_rows[(r["scenario_id"], r["level"])]
        res = components(g["prompt"])["results"]
        obj = parse_result(res[-1])[1] if res else None
        vals = list(flatten(obj).values()) if isinstance(obj, (dict, list)) else []
        r["neg_literals"] = sum(v is None or v is False or v in ([], {}) or (not isinstance(v, bool) and isinstance(v, (int, float))
                                                                               and v == 0) for v in vals)
        r["true_literals"] = sum(v is True for v in vals)
    for t in "AB":
        sel = [r for r in prompts if r["t"] == t]
        pol[t] = {"n": len(sel),
                  "spearman_ask_rate_vs_neg_literals": spearman([r["neg_literals"] for r in sel], [r["ask_rate"] for r in sel]),
                  "perm_p_neg": perm_p_spearman([r["neg_literals"] for r in sel], [r["ask_rate"] for r in sel]),
                  "spearman_ask_rate_vs_true_literals": spearman([r["true_literals"] for r in sel], [r["ask_rate"] for r in sel]),
                  "perm_p_true": perm_p_spearman([r["true_literals"] for r in sel], [r["ask_rate"] for r in sel])}
    # pair-level: in pairs with both twins sampled at the SAME level, does the twin with more negative literals get asked more?
    lvl = defaultdict(dict)
    for r in prompts:
        lvl[(r["pair_id"], r["level"])][r["t"]] = r
    same = [(d["A"], d["B"]) for d in lvl.values() if "A" in d and "B" in d]
    pol["same_level_twin_pairs"] = [{"pair_id": a["pair_id"], "level": a["level"], "neg_A": a["neg_literals"],
                                     "neg_B": b["neg_literals"], "ask_A": a["ask_rate"], "ask_B": b["ask_rate"]} for a, b in same]
    out["m1_negative_literal_heuristic"] = pol
    out["m1_prompts"] = prompts

    # ---------------------------------------------------------------- base Tiny Aya (eval_base_v12)
    base = load_jsonl(os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl"))
    brows = []
    for r in base:
        t = twin_letter(r["twin"])
        if t == "P":
            continue
        d = r["decision_grade"]["decision"]
        sp = split.get(r["pair_id"]) or origin.get(r["id"], "?")
        brows.append({"t": t, "pair_id": r["pair_id"], "level": r["level"], "ps": r["prompt"], "split": sp,
                      "req": req_stratum.get(r["pair_id"], "?"), "ask_rate": float(d in ASK),
                      "correct_rate": float(r["decision_grade"]["success"]), "approve_rate": float(d in APPROVE)})
    out["base_n"] = len(brows)
    out["base_by_split"] = strat(brows, "split")
    out["base_by_request_stratum"] = strat(brows, "req")
    out["base_by_request_stratum_per_split"] = {sp: strat([r for r in brows if r["split"] == sp], "req")
                                                for sp in sorted({r["split"] for r in brows})}
    out["bootstrap_wording_vs_identical"] = {"m1_prompts": disc_boot(prompts), "base_all_splits": disc_boot(brows),
                                             "base_test": disc_boot([r for r in brows if r["split"] == "test"]),
                                             "base_hard_sets": disc_boot([r for r in brows if r["split"].startswith("hard")])}
    print(json.dumps(out["bootstrap_wording_vs_identical"], indent=1))
    jdump(out, os.path.join(RES, "s4_policy_behaviour.json"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("m1_prompts", "m1_pairs_both_twins", "base_by_request_stratum_per_split")},
                     indent=1))
    print(json.dumps({k: v for k, v in out["m1_pairs_both_twins"].items() if k != "pairs"}, indent=1))
    for p in out["m1_pairs_both_twins"]["pairs"]:
        print("  ", p)
    print(json.dumps(out["base_by_request_stratum_per_split"], indent=1))


if __name__ == "__main__":
    main()
