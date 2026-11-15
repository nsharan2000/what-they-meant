"""What the M1 SFT policy does on its OWN training prompts: the 960 GRPO samples (temperature 1, top_p 1; 30 steps x 4 prompts x 8
generations; meta.step = GRPO steps already applied, and GRPO moved the weights 1.8 % as much as SFT did, so these are ~M1 samples).

Per sample: correct class (twin A: ask_informed | tell_informed; twin B expected approve: approve_*; twin B expected act:
commit_gold, lenient also approve_*), first-token error (A -> any tool call; B -> a chat message), joined with
  grpo_train.jsonl (prompt setting, env, expected decision; every GRPO prompt is byte-identical to an SFT-train decision prompt)
  results/twin_contrast.csv (where the pair's twin prompts differ: request text vs only the look-up result).
Per prompt: p_correct over its 8 samples -> class-level NLL -ln p (a LOWER bound on the target sequence's NLL, because the
target is one member of its class), compared with the SFT per-row loss implied by the logged per-token train loss.

    python3 experiments/dataset_audit/composition-memorization/grpo_decisions.py
Output: results/grpo_decisions.json, results/grpo_prompt_table.csv
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np
import pandas as pd

from common import GRPO_SCORES, OUT, dump, load_jsonl, load_view, source_of

A_OK = {"ask_informed", "tell_informed"}


def main():
    sc = load_jsonl(GRPO_SCORES)
    grpo = load_view("grpo_train")
    meta = {}
    for i, r in enumerate(grpo):
        s = json.loads(r["scenario"])
        meta[(r["scenario_id"], r["level"])] = {"grpo_row": i, "ps": r["prompt_setting"], "env": r["env_id"], "pair_id": r["pair_id"],
                                                "expected": s["expected"]["decision"],
                                                "accept": "|".join(sorted(set(s["expected"].get("accept") or [s["expected"]["decision"]]))),
                                                "prompt_msgs": len(r["prompt"])}
    tc = pd.read_csv(os.path.join(OUT, "twin_contrast.csv"))
    tc = tc[tc.split == "sft_train"]
    cue = {(r.pair_id, r.level): r.cue for r in tc.itertuples()}
    recs = []
    for k, x in enumerate(sc):
        m = x["meta"]
        key = (m["scenario_id"], m["level"])
        g = meta[key]
        d = x["components"]["decision"]
        tw = m["twin"][:1]
        if tw == "A":
            ok = d in A_OK
            ok_len = ok
            first_tok_wrong = not (d.startswith(("ask", "tell")) or d == "statement")      # chose a tool call (card/commit/lookup)
        elif g["expected"] == "act":
            ok = d == "commit_gold"
            ok_len = ok or d.startswith("approve")
            first_tok_wrong = d.startswith(("ask", "tell")) or d == "statement"
        else:
            ok = d.startswith("approve")
            ok_len = ok
            first_tok_wrong = d.startswith(("ask", "tell")) or d == "statement"
        recs.append({"k": k, "scenario_id": m["scenario_id"], "level": m["level"], "twin": tw, "step": m["step"], "decision": d,
                     "score": x["score"], "ok": ok, "ok_lenient": ok_len, "first_token_wrong": first_tok_wrong,
                     "n_tokens": m["n_tokens"], "source": source_of(m["scenario_id"]), "cue": cue.get((g["pair_id"], m["level"]), "no_twin_row"),
                     **g})
    df = pd.DataFrame(recs)
    out = {"n_samples": len(df), "n_prompts": int(df.groupby(["scenario_id", "level"]).ngroups),
           "samples_per_prompt": dict(collections.Counter(df.groupby(["scenario_id", "level"]).size())),
           "steps": [int(df.step.min()), int(df.step.max())]}
    out["decision_by_twin"] = {tw: dict(collections.Counter(g.decision).most_common()) for tw, g in df.groupby("twin")}
    out["decision_by_twin_expected"] = {f"{a}|{b}": dict(collections.Counter(g.decision).most_common())
                                        for (a, b), g in df.groupby(["twin", "expected"])}

    def rate(g):
        return {"n": int(len(g)), "prompts": int(g.groupby(["scenario_id", "level"]).ngroups), "acc": round(float(g.ok.mean()), 3),
                "acc_lenient": round(float(g.ok_lenient.mean()), 3), "first_token_wrong": round(float(g.first_token_wrong.mean()), 3)}
    out["overall"] = rate(df)
    for col in ("twin", "level", "ps", "source", "env", "cue", "expected"):
        out[f"by_{col}"] = {str(v): rate(g) for v, g in df.groupby(col)}
    out["by_twin_x_cue"] = {f"{a}|{b}": rate(g) for (a, b), g in df.groupby(["twin", "cue"])}
    out["by_twin_x_ps"] = {f"{a}|{b}": rate(g) for (a, b), g in df.groupby(["twin", "ps"])}
    out["by_step_block"] = {f"steps_{lo}-{lo + 9}": rate(df[(df.step >= lo) & (df.step <= lo + 9)]) for lo in (0, 10, 20)}
    # error composition: share of wrong samples that are a first-token (message vs tool-call) error
    wrong = df[~df.ok_lenient]
    out["wrong_lenient_n"] = int(len(wrong))
    out["wrong_that_are_first_token_errors"] = round(float(wrong.first_token_wrong.mean()), 3)
    a_s, b_s = df[df.twin == "A"], df[df.twin == "B"]
    out["twinA_samples_approval_card"] = {"n": int(a_s.decision.str.startswith("approve").sum()), "of": int(len(a_s)),
                                          "share": round(float(a_s.decision.str.startswith("approve").mean()), 3)}
    out["twinB_samples_chat_message"] = {"n": int(b_s.decision.str.startswith(("ask", "tell")).sum() + (b_s.decision == "statement").sum()),
                                         "of": int(len(b_s)),
                                         "share": round(float((b_s.decision.str.startswith(("ask", "tell")) | (b_s.decision == "statement")).mean()), 3)}
    out["lookup_samples"] = int((df.decision == "lookup").sum())
    # decision prompts without a prior look-up ([system, user], the same shape as every look-up-turn prompt) vs the rest
    out["by_prompt_shape"] = {f"{n}_msgs": {"samples": int(len(g)), "prompts": int(g.groupby(["scenario_id", "level"]).ngroups),
                                            "decisions": dict(collections.Counter(g.decision).most_common()),
                                            "lookup_samples": int((g.decision == "lookup").sum())}
                              for n, g in df.groupby("prompt_msgs")}
    # ---- per prompt -----------------------------------------------------------------------------------------------------
    pp = df.groupby(["scenario_id", "level"]).agg(twin=("twin", "first"), ps=("ps", "first"), cue=("cue", "first"),
                                                  expected=("expected", "first"), source=("source", "first"), n=("ok", "size"),
                                                  k=("ok", "sum"), k_len=("ok_lenient", "sum"), grpo_row=("grpo_row", "first"),
                                                  first_wrong=("first_token_wrong", "sum")).reset_index()
    pp["p"] = pp.k / pp.n
    pp.to_csv(os.path.join(OUT, "grpo_prompt_table.csv"), index=False)
    out["per_prompt_correct_histogram"] = {f"{int(k)}/8": int(n) for k, n in sorted(collections.Counter(pp.k).items())}
    out["per_prompt_correct_histogram_lenient"] = {f"{int(k)}/8": int(n) for k, n in sorted(collections.Counter(pp.k_len).items())}
    out["prompts_all_wrong"] = int((pp.k == 0).sum())
    out["prompts_all_right"] = int((pp.k == pp.n).sum())
    out["prompts_mixed"] = int(((pp.k > 0) & (pp.k < pp.n)).sum())
    # bimodality check: observed variance of per-prompt k vs a binomial with the pooled p (overdispersion ratio)
    pbar = pp.k.sum() / pp.n.sum()
    out["overdispersion_ratio"] = round(float(pp.k.var(ddof=1) / (8 * pbar * (1 - pbar))), 2)
    # implied class NLL at the decision (Jeffreys-smoothed p), a lower bound on the target's sequence NLL
    ph = (pp.k + 0.5) / (pp.n + 1.0)
    nll = -np.log(ph)
    out["class_nll_per_decision_row"] = {"mean": round(float(nll.mean()), 3), "median": round(float(nll.median()), 3),
                                         "mean_twinA": round(float(nll[pp.twin == "A"].mean()), 3),
                                         "mean_twinB": round(float(nll[pp.twin == "B"].mean()), 3),
                                         "smoothing": "(k+0.5)/(n+1)"}
    examples = {}
    for name, sub in (("all_wrong", pp[pp.k == 0]), ("mixed", pp[(pp.k > 0) & (pp.k < pp.n)]), ("all_right", pp[pp.k == pp.n])):
        examples[name] = sub.head(4)[["scenario_id", "level", "ps", "cue", "k", "n", "grpo_row"]].to_dict("records")
    out["prompt_examples"] = examples
    # prompt-level accuracy by cue (each prompt counted once)
    out["prompt_level_by_cue"] = {c: {"prompts": int(len(g)), "mean_p": round(float(g.p.mean()), 3),
                                      "all_wrong": int((g.k == 0).sum()), "all_right": int((g.k == g.n).sum())}
                                  for c, g in pp.groupby("cue")}
    out["prompt_level_by_twin_x_cue"] = {f"{a}|{b}": {"prompts": int(len(g)), "mean_p": round(float(g.p.mean()), 3)}
                                         for (a, b), g in pp.groupby(["twin", "cue"])}
    # twin-symmetric confusion: when the pair's twins differ only in the look-up result, does the policy give the same
    # class to both twins? (pairs where both twins of the same level were sampled)
    both = pp.assign(pair=pp.scenario_id.str[:-1]).groupby(["pair", "level"]).filter(lambda g: set(g.twin) == {"A", "B"})
    out["pair_levels_with_both_twins_sampled"] = int(both.groupby(["pair", "level"]).ngroups)
    # ---- robustness: a look-up is not a decision in the evaluation driver (decision mode continues after it) -----------------
    dd = df[df.decision != "lookup"]
    out["excluding_lookup_samples"] = {"n": int(len(dd)), "acc": round(float(dd.ok.mean()), 3),
                                       "first_token_wrong": round(float(dd.first_token_wrong.mean()), 3),
                                       "by_twin": {tw: {"n": int(len(g)), "acc": round(float(g.ok.mean()), 3),
                                                        "first_token_wrong": round(float(g.first_token_wrong.mean()), 3)}
                                                   for tw, g in dd.groupby("twin")}}
    # ---- discrimination: does P(tool-call first token) separate twin B prompts from twin A prompts? ----------------------
    df["tool_first"] = ~df.decision.str.startswith(("ask", "tell")) & (df.decision != "statement")
    pt = df[df.decision != "lookup"].groupby(["scenario_id", "level"]).agg(twin=("twin", "first"), p_tool=("tool_first", "mean"),
                                                                          ps=("ps", "first")).reset_index()
    a, b = pt[pt.twin == "A"].p_tool.values, pt[pt.twin == "B"].p_tool.values
    auc = float(((b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum()) / (len(a) * len(b)))
    out["first_token_discrimination"] = {"mean_p_tool_twinA": round(float(a.mean()), 3), "mean_p_tool_twinB": round(float(b.mean()), 3),
                                         "auc_B_over_A": round(auc, 3), "n_A": int(len(a)), "n_B": int(len(b)),
                                         "note": "AUC 0.5 = the first token ignores which twin the prompt is; 1.0 = perfect"}
    pair_rows = []
    for (pair, lv), g in pt.assign(pair=pt.scenario_id.str[:-1]).groupby(["pair", "level"]):
        if set(g.twin) == {"A", "B"}:
            pair_rows.append({"pair": pair, "level": lv, "p_tool_A": round(float(g[g.twin == "A"].p_tool.iloc[0]), 3),
                              "p_tool_B": round(float(g[g.twin == "B"].p_tool.iloc[0]), 3)})
    out["paired_twin_levels"] = {"n": len(pair_rows), "B_more_tool_calls_than_A": sum(r["p_tool_B"] > r["p_tool_A"] for r in pair_rows),
                                 "equal": sum(r["p_tool_B"] == r["p_tool_A"] for r in pair_rows), "rows": pair_rows}
    # ---- P0 vs P2 at the prompt level: permutation test on mean per-prompt accuracy -------------------------------------
    rng = np.random.default_rng(0)
    obs = pp[pp.ps == "P2"].p.mean() - pp[pp.ps == "P0"].p.mean()
    lab = (pp.ps == "P2").to_numpy().copy()
    perm = []
    for _ in range(20000):
        rng.shuffle(lab)
        perm.append(pp.p.values[lab].mean() - pp.p.values[~lab].mean())
    perm = np.array(perm)
    out["P2_minus_P0_prompt_accuracy"] = {"diff": round(float(obs), 3), "perm_p_two_sided": round(float((np.abs(perm) >= abs(obs)).mean()), 3),
                                          "n_P2": int((pp.ps == "P2").sum()), "n_P0": int((pp.ps == "P0").sum())}
    # same for level L0r vs L2 (realism): mean per-prompt accuracy
    out["per_prompt_mean_p_by_level"] = {lv: round(float(g.p.mean()), 3) for lv, g in pp.assign(level=pp.level).groupby("level")}
    dump(out, "grpo_decisions.json")
    print(json.dumps({k: out[k] for k in ("overall", "by_twin", "by_cue", "by_twin_x_cue", "per_prompt_correct_histogram",
                                          "overdispersion_ratio", "class_nll_per_decision_row", "prompt_level_by_cue",
                                          "by_ps", "by_level", "by_source", "by_step_block", "wrong_that_are_first_token_errors")},
                     indent=0))


if __name__ == "__main__":
    main()
