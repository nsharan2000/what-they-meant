"""Step 6 — model-side evidence: does the prompt's literal content move the decisions the models actually make?

(1) M1 (SFT) policy samples drawn during GRPO on TRAINING decision prompts
    (experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl, 960 samples = 120 prompts x 8; decision class from
    wtm/reward.py). Joined to grpo_train.jsonl by (scenario_id, level) — unique per prompt — and to this lens's
    literal-prescription category (results/literal_prescription.jsonl, unit GRPO) and annotations.
    Cluster bootstrap over prompts (8 samples of one prompt are not independent), 2,000 resamples, seed 0.
(2) Base Tiny Aya, paired P0 vs P2 on the same scenario x level (experiments/training_eval/results/eval_base_v12.jsonl):
    what the 1-2 line hint alone does to asking / carding / committing.
(3) Base Tiny Aya, replication grid (experiments/replication/results/grid_local_tinyaya.jsonl): L0r (no policy) vs L1 (policy
    with the approval-card rule) at P0 — does the policy text move the base model toward approval cards?
Output: results/model_evidence.json
    python3 -B experiments/dataset_audit/prompt-conflicts/06_model_evidence.py
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from pc_common import RES, ROOT, dump, load_rows, rel

ASK = ("ask_informed", "ask_generic")
CARD = ("approve_generic", "approve_informed", "approve_gold")
COMMIT = ("commit_gold", "commit_other", "commit_harmful")


def rate(samples, classes):
    return round(sum(s["decision"] in classes for s in samples) / len(samples), 3) if samples else None


def boot_diff(groups_a, groups_b, classes, n=2000, seed=0):
    """groups_*: list of per-prompt sample lists. Returns mean diff (a-b) of class rate and 95% CI (cluster bootstrap)."""
    if not groups_a or not groups_b:
        return None
    rng = np.random.default_rng(seed)
    fa = np.array([sum(s["decision"] in classes for s in g) for g in groups_a], float)
    na = np.array([len(g) for g in groups_a], float)
    fb = np.array([sum(s["decision"] in classes for s in g) for g in groups_b], float)
    nb = np.array([len(g) for g in groups_b], float)
    obs = fa.sum() / na.sum() - fb.sum() / nb.sum()
    ia = rng.integers(0, len(fa), (n, len(fa)))
    ib = rng.integers(0, len(fb), (n, len(fb)))
    d = fa[ia].sum(1) / na[ia].sum(1) - fb[ib].sum(1) / nb[ib].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(float(obs), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "prompts": [len(fa), len(fb)], "samples": [int(na.sum()), int(nb.sum())]}


def dist(samples):
    return dict(collections.Counter(s["decision"] for s in samples).most_common())


def m1_evidence():
    scores = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
    grpo = {(r["scenario_id"], r["level"]): (i, r) for i, r in enumerate(load_rows("grpo_train"))}
    lit = {(x["scenario_id"], x["level"]): x for x in map(json.loads, open(f"{RES}/literal_prescription.jsonl")) if x["unit"] == "GRPO"}
    tt = {(x["scenario_id"], x["level"]): x for x in map(json.loads, open(f"{RES}/target_table.jsonl"))
          if x["file"] == "grpo_train"}
    samples = []
    for s in scores:
        m = s["meta"]
        k = (m["scenario_id"], m["level"])
        i, r = grpo[k]
        x = tt[k]
        samples.append({"key": f"{k[0]}|{k[1]}", "grpo_line": i, "scenario_id": k[0], "level": k[1], "t": m["twin"][0],
                        "env": r["env_id"], "prompt_setting": r["prompt_setting"], "decision": s["components"]["decision"],
                        "category": lit[k]["category"], "fraud": x["fraud_tag"], "exec": x.get("a_exec"),
                        "clash": x.get("a_clash"), "card_states_clash": x.get("a_card_states_clash"),
                        "expected": x["expected"], "sa_visible": x.get("b_sa_visible"), "step": m["step"]})
    by_prompt = collections.defaultdict(list)
    for s in samples:
        by_prompt[s["key"]].append(s)

    def groups(pred):
        return [g for g in by_prompt.values() if pred(g[0])]

    def flat(gs):
        return [s for g in gs for s in g]

    out = {"n_samples": len(samples), "n_prompts": len(by_prompt),
           "decision_by_twin": {t: dist([s for s in samples if s["t"] == t]) for t in "AB"}}
    # A twins: by literal-prescription category
    A_cat = {}
    for cat in sorted({s["category"] for s in samples if s["t"] == "A"}):
        gs = groups(lambda s, c=cat: s["t"] == "A" and s["category"] == c)
        f = flat(gs)
        A_cat[cat] = {"prompts": len(gs), "samples": len(f), "ask_informed": rate(f, ("ask_informed",)),
                      "approve_informed": rate(f, ("approve_informed",)), "any_card": rate(f, CARD), "dist": dist(f)}
    out["A_by_literal_category"] = A_cat
    # A: env policy contradicts asking (bizops executable/fraud, P0) vs everything else at L1+ P0
    out["A_contradicted_vs_other_L1plus_P0_ask_informed"] = boot_diff(
        groups(lambda s: s["t"] == "A" and s["category"] == "contradicted"),
        groups(lambda s: s["t"] == "A" and s["level"] != "L0r" and s["prompt_setting"] == "P0" and s["category"] != "contradicted"),
        ("ask_informed",))
    # A: card-content rule names the clashing field (competing informed card) vs not, L1+ P0: approve_informed rate
    out["A_card_states_clash_vs_not_L1plus_P0_approve_informed"] = boot_diff(
        groups(lambda s: s["t"] == "A" and s["level"] != "L0r" and s["prompt_setting"] == "P0" and s["card_states_clash"]),
        groups(lambda s: s["t"] == "A" and s["level"] != "L0r" and s["prompt_setting"] == "P0" and not s["card_states_clash"]),
        ("approve_informed",))
    out["A_card_states_clash_vs_not_L0r_P0_approve_informed(control: no card-content rule at L0r)"] = boot_diff(
        groups(lambda s: s["t"] == "A" and s["level"] == "L0r" and s["prompt_setting"] == "P0" and s["card_states_clash"]),
        groups(lambda s: s["t"] == "A" and s["level"] == "L0r" and s["prompt_setting"] == "P0" and not s["card_states_clash"]),
        ("approve_informed",))
    # rule present vs absent, holding the clash type fixed: card-field clashes at L1+ P0 (card-content rule in the prompt) vs
    # L0r P0 (no card-content rule; wallet excluded because its L0r tool doc also describes the card content)
    out["A_card_field_clash_L1plus_vs_L0r_P0_approve_informed(non-wallet)"] = boot_diff(
        groups(lambda s: s["t"] == "A" and s["level"] != "L0r" and s["prompt_setting"] == "P0" and s["card_states_clash"]
               and s["env"] != "wallet"),
        groups(lambda s: s["t"] == "A" and s["level"] == "L0r" and s["prompt_setting"] == "P0" and s["card_states_clash"]
               and s["env"] != "wallet"), ("approve_informed",))
    out["A_card_field_clash_L1plus_vs_L0r_P0_approve_informed(all envs)"] = boot_diff(
        groups(lambda s: s["t"] == "A" and s["level"] != "L0r" and s["prompt_setting"] == "P0" and s["card_states_clash"]),
        groups(lambda s: s["t"] == "A" and s["level"] == "L0r" and s["prompt_setting"] == "P0" and s["card_states_clash"]),
        ("approve_informed",))
    out["A_approve_informed_rate_by_level"] = {lv: rate([s for s in samples if s["t"] == "A" and s["level"] == lv],
                                                        ("approve_informed",)) for lv in ("L0r", "L1", "L2", "L3")}
    # the hint at the same level (L0r: the hint is the ONLY rule) after SFT
    out["A_L0r_P2_minus_P0_ask_informed"] = boot_diff(
        groups(lambda s: s["t"] == "A" and s["level"] == "L0r" and s["prompt_setting"] == "P2"),
        groups(lambda s: s["t"] == "A" and s["level"] == "L0r" and s["prompt_setting"] == "P0"), ("ask_informed",))
    out["B_L0r_P2_minus_P0_card"] = boot_diff(
        groups(lambda s: s["t"] == "B" and s["level"] == "L0r" and s["prompt_setting"] == "P2" and s["expected"] == "approve"),
        groups(lambda s: s["t"] == "B" and s["level"] == "L0r" and s["prompt_setting"] == "P0" and s["expected"] == "approve"), CARD)
    # B twins by literal-prescription category
    B_cat = {}
    for cat in sorted({s["category"] for s in samples if s["t"] == "B"}):
        gs = groups(lambda s, c=cat: s["t"] == "B" and s["category"] == c)
        f = flat(gs)
        B_cat[cat] = {"prompts": len(gs), "samples": len(f), "card": rate(f, CARD), "any_question": rate(f, ASK + ("tell_informed",)),
                      "commit_gold": rate(f, ("commit_gold",)), "dist": dist(f)}
    out["B_by_literal_category"] = B_cat
    # correct-class rate (the class the target has) by category, both twins
    def correct(s):
        if s["t"] == "A":
            return s["decision"] == "ask_informed"
        return s["decision"].startswith("approve") if s["expected"] == "approve" else s["decision"] == "commit_gold"
    out["target_class_rate_by_category"] = {
        cat: {"prompts": len(groups(lambda s, c=cat: s["category"] == c)),
              "rate": round(sum(correct(s) for s in samples if s["category"] == cat) /
                            max(1, sum(s["category"] == cat for s in samples)), 3)}
        for cat in sorted({s["category"] for s in samples})}
    out["target_class_rate_prescribed_minus_unspecified"] = {
        t: boot_diff(groups(lambda s, t=t: s["t"] == t and s["category"] == "prescribed"),
                     groups(lambda s, t=t: s["t"] == t and s["category"] == "unspecified"),
                     ("ask_informed",) if t == "A" else CARD) for t in "AB"}
    # P2 hint effect after SFT (unpaired: different scenarios carry P0/P2)
    out["P2_minus_P0"] = {
        "A_ask_informed": boot_diff(groups(lambda s: s["t"] == "A" and s["prompt_setting"] == "P2"),
                                    groups(lambda s: s["t"] == "A" and s["prompt_setting"] == "P0"), ("ask_informed",)),
        "B_any_question": boot_diff(groups(lambda s: s["t"] == "B" and s["prompt_setting"] == "P2"),
                                    groups(lambda s: s["t"] == "B" and s["prompt_setting"] == "P0"), ASK + ("tell_informed",)),
        "B_card": boot_diff(groups(lambda s: s["t"] == "B" and s["prompt_setting"] == "P2" and s["expected"] == "approve"),
                            groups(lambda s: s["t"] == "B" and s["prompt_setting"] == "P0" and s["expected"] == "approve"), CARD)}
    out["by_level"] = {lv: {t: {"prompts": len(groups(lambda s, lv=lv, t=t: s["t"] == t and s["level"] == lv)),
                                "dist": dist(flat(groups(lambda s, lv=lv, t=t: s["t"] == t and s["level"] == lv)))}
                            for t in "AB"} for lv in ("L0r", "L1", "L2", "L3")}
    out["by_env"] = {e: {t: dist([s for s in samples if s["t"] == t and s["env"] == e]) for t in "AB"}
                     for e in ("wallet", "bills", "shopping", "bizops")}
    # fraud-tagged A rows: does the policy's 'tell' pull show up?
    out["A_fraud"] = {"bizops_L1plus": dist([s for s in samples if s["t"] == "A" and s["fraud"] and s["env"] == "bizops"
                                             and s["level"] != "L0r"]),
                      "wallet_L1plus": dist([s for s in samples if s["t"] == "A" and s["fraud"] and s["env"] == "wallet"
                                             and s["level"] != "L0r"]),
                      "all_A_tell_informed_samples": sum(s["t"] == "A" and s["decision"] == "tell_informed" for s in samples)}
    # B rows with special prompt status
    out["B_special"] = {
        "authorisation_visible_approve_rows": {s["key"]: dist(g) for s0 in [None] for g in by_prompt.values()
                                               for s in [g[0]] if s["t"] == "B" and s["expected"] == "approve" and s["sa_visible"]},
        "act_rows": {g[0]["key"] + "|" + g[0]["prompt_setting"]: dist(g) for g in by_prompt.values()
                     if g[0]["t"] == "B" and g[0]["expected"] == "act"}}
    # zero-lookup prompts (S02/S03): does the policy's 'look it up first' pull show up?
    zl = {"S02A", "S02B", "S03A", "S03B"}
    out["zero_lookup_prompts"] = {g[0]["key"] + "|" + g[0]["prompt_setting"]: dist(g) for g in by_prompt.values()
                                  if g[0]["scenario_id"] in zl}
    out["lookup_rate_zero_lookup_prompts_vs_others"] = boot_diff(
        [g for g in by_prompt.values() if g[0]["scenario_id"] in zl],
        [g for g in by_prompt.values() if g[0]["scenario_id"] not in zl], ("lookup",))
    return out


def base_hint_paired():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl"))]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["id"], r["level"])][r["prompt"]] = r["decision_grade"]["decision"]
    pairs = [(k, v["P0"], v["P2"]) for k, v in by.items() if "P0" in v and "P2" in v]
    out = {"pairs": len(pairs)}
    for t in "AB":
        P = [p for p in pairs if p[0][0][-1] == t]
        out[t] = {"pairs": len(P)}
        for name, cls in (("ask_any", ASK), ("ask_informed", ("ask_informed",)), ("card", CARD), ("commit", COMMIT)):
            a0 = sum(p[1] in cls for p in P)
            a2 = sum(p[2] in cls for p in P)
            up = sum(p[1] not in cls and p[2] in cls for p in P)
            down = sum(p[1] in cls and p[2] not in cls for p in P)
            out[t][name] = {"P0": a0, "P2": a2, "P0->P2 gained": up, "P0->P2 lost": down}
    return out


def base_policy_effect():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/replication/results/grid_local_tinyaya.jsonl"))]
    out = {}
    for t in "AB":
        out[t] = {}
        for lv in ("L0r", "L1", "L2"):
            R = [r for r in rows if r["twin"][0] == t and r["level"] == lv and r["prompt"] == "P0"]
            d = collections.Counter(r["decision_grade"]["decision"] for r in R)
            out[t][lv] = {"n": len(R), "card": sum(d[c] for c in CARD), "commit": sum(d[c] for c in COMMIT),
                          "ask": sum(d[c] for c in ASK), "dist": dict(d.most_common())}
    return out


def main():
    res = {"m1_policy_on_training_prompts": m1_evidence(), "base_tinyaya_hint_P0_vs_P2_paired": base_hint_paired(),
           "base_tinyaya_policy_L0r_vs_L1_P0": base_policy_effect()}
    p = dump("model_evidence.json", res)
    m = res["m1_policy_on_training_prompts"]
    print(json.dumps({k: m[k] for k in ("A_by_literal_category", "A_contradicted_vs_other_L1plus_P0_ask_informed",
                                        "A_card_states_clash_vs_not_L1plus_P0_approve_informed", "P2_minus_P0")}, indent=1))
    print("wrote", rel(p))


if __name__ == "__main__":
    main()
