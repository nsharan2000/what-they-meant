"""Step 5 — where do the instructions sit relative to the generation point, in Tiny Aya tokens?

Tiny Aya (Cohere2) config: sliding_window 4096, sliding_window_pattern 4 (3 of every 4 layers attend only to the last 4,096
tokens). We have exact prompt token counts per row (experiments/dataset_audit/results/token_stats.jsonl, chat template incl.)
but no tokenizer locally (no transformers), so section token counts are ESTIMATED: a least-squares fit (numpy) of
prompt_tokens on the character counts of each prompt section (policy / member block / tool docs / protocol+hint /
conversation) over sft_train + sft_val + dpo_train rows (grpo_train prompts duplicate dpo_train's). The fit's residuals are
reported; distances are then computed per row as prompt_tokens - estimated tokens up to the end of the instruction quote.
Output: results/instruction_positions.json
    python3 -B experiments/dataset_audit/prompt-conflicts/05_positions.py
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from pc_common import HINT, ROOT, dump, load_envs, load_rows, rel, split_system

SW = 4096
QUOTES = {   # instruction id -> verbatim quote (subset of instructions.json), searched in the policy block
    "W-card-required": "Every payment, transfer, refund, invoice payment, QR payment, crypto withdrawal and swap needs the member's approval on a card.",
    "W-keep-light": "Ask in chat when something you need is missing or you can't prepare an accurate card.",
    "W-SA": "A request that falls entirely within an active standing authorisation may be carried out without a new card",
    "W-new-payee": "Make sure the person or business you found is the one the member means to pay.",
    "W-warning-signs": "If you notice one, tell the member plainly what you saw and let them decide.",
    "BL-card-content": "any recurring price, renewal date, fee or other term the member is agreeing to.",
    "BL-SA": "A standing authorisation lets you act without a card, but only inside its scope and limits.",
    "BL-tone": "Ask one question at a time.",
    "SH-card-content": "the total in the member's currency and the payment method), plus anything they would want to know before agreeing",
    "SH-SA": "can be completed straight away without an approval card",
    "SH-ask": "When you need something from the member, ask one clear question.",
    "BZ-card-required": "Before a committed action, get the user's approval on an approval card (request_approval), unless a standing authorisation covers it.",
    "BZ-SA": "When a request falls entirely inside a standing authorisation, carry it out without an approval card",
    "BZ-fraud": "If something looks like fraud, stop and tell the user",
    "BZ-ask-only-if-needed": "Ask at most one question at a time, and only when you need the answer to continue.",
}


def main():
    envs = load_envs()
    ts = {}
    for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")):
        x = json.loads(l)
        ts[(x["file"].replace(".jsonl", ""), x["i"])] = x
    feats, ys, meta = [], [], []
    mismatch = 0
    for f in ("sft_train", "sft_val", "dpo_train", "grpo_train"):
        for i, r in enumerate(load_rows(f)):
            t = ts[(f, i)]
            if (t["scenario_id"], t["level"], t["prompt_setting"]) != (r["scenario_id"], r["level"], r["prompt_setting"]):
                mismatch += 1
                continue
            sysm = r["prompt"][0]["content"]
            sec = split_system(sysm, envs[r["env_id"]], r["level"])
            conv = sum(len(m["content"]) for m in r["prompt"][1:])
            x = [len(sec["policy"]), len(sec["member"]), len(sec["tools"]), len(sec["protocol"]) + len(sec["tail"]), conv,
                 len(r["prompt"]), 1.0]
            meta.append((f, i, r, sec))
            if f != "grpo_train":
                feats.append(x)
                ys.append(t["prompt_tokens"])
    X, y = np.array(feats, float), np.array(ys, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    names = ["policy_chars", "member_chars", "tooldoc_chars", "protocol_hint_chars", "conversation_chars", "n_messages", "const"]
    fit = {"n_rows": len(y), "coef_tokens_per_char_or_unit": dict(zip(names, [round(float(c), 4) for c in coef])),
           "chars_per_token_implied": {n: (round(1 / float(c), 2) if c > 0 and n.endswith("chars") else None)
                                       for n, c in zip(names, coef)},
           "residual_tokens": {"mean_abs": round(float(np.abs(resid).mean()), 1), "p95_abs": round(float(np.percentile(np.abs(resid), 95)), 1),
                               "max_abs": round(float(np.abs(resid).max()), 1)},
           "r2": round(1 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
           "token_stats_join_mismatches": mismatch}
    k_pol, k_mem, k_tool, k_proto, k_conv, k_msg, k0 = coef
    rows = []
    for f, i, r, sec in meta:
        if f not in ("sft_train",) or r.get("turn") != "decision":
            continue
        T = ts[(f, i)]["prompt_tokens"]
        # conservative: the whole template constant (BOS, role headers) is counted BEFORE the policy, which makes every
        # "tokens after" distance smaller, i.e. the outside-window counts below are lower bounds under the fitted rates
        before_policy_end = k0 + len(sec["policy"]) * k_pol
        d = {"i": i, "scenario_id": r["scenario_id"], "env": r["env_id"], "level": r["level"],
             "prompt_setting": r["prompt_setting"], "prompt_tokens": T,
             "tokens_after_policy_start": round(T - k0),
             "tokens_after_policy_end": round(T - before_policy_end)}
        for qid, q in QUOTES.items():
            j = sec["policy"].find(q)
            if j >= 0:
                d[f"after:{qid}"] = round(T - (k0 + (j + len(q)) * k_pol))
        if r["prompt_setting"] == "P2":
            conv = sum(len(m["content"]) for m in r["prompt"][1:])
            d["after:P2-hint"] = round(conv * k_conv + (len(r["prompt"]) - 1) * k_msg)
        rows.append(d)
    summ = {}
    margin = float(np.percentile(np.abs(resid), 95))
    for lv in ("L0r", "L1", "L2", "L3"):
        R = [d for d in rows if d["level"] == lv]
        if not R:
            continue
        s = {"decision_rows": len(R), "prompt_tokens_mean": round(float(np.mean([d["prompt_tokens"] for d in R])), 1),
             "whole_prompt_beyond_window(>4096)": sum(d["prompt_tokens"] > SW for d in R),
             "policy_START_outside_window": sum(d["tokens_after_policy_start"] > SW for d in R),
             "policy_END_outside_window(entire policy unseen by local layers)": sum(d["tokens_after_policy_end"] > SW for d in R),
             "policy_END_outside_window_by_margin(>4096+p95|resid|)": sum(d["tokens_after_policy_end"] > SW + margin for d in R),
             "policy_END_outside_window_by_neg_margin(>4096-p95|resid|)": sum(d["tokens_after_policy_end"] > SW - margin for d in R),
             "tokens_after_policy_end": {"min": min(d["tokens_after_policy_end"] for d in R),
                                         "mean": round(float(np.mean([d["tokens_after_policy_end"] for d in R])), 1),
                                         "max": max(d["tokens_after_policy_end"] for d in R)}}
        per_q = collections.OrderedDict()
        for qid in QUOTES:
            v = [d[f"after:{qid}"] for d in R if f"after:{qid}" in d]
            if v:
                per_q[qid] = {"rows": len(v), "outside_window": sum(x > SW for x in v), "mean_tokens_after": round(float(np.mean(v)), 1)}
        s["instructions"] = per_q
        h = [d["after:P2-hint"] for d in R if "after:P2-hint" in d]
        if h:
            s["P2_hint"] = {"rows": len(h), "outside_window": sum(x > SW for x in h), "mean_tokens_after": round(float(np.mean(h)), 1)}
        summ[lv] = s
    examples = sorted([d for d in rows if d["level"] in ("L2", "L3")], key=lambda d: -d["tokens_after_policy_end"])[:3]
    p = dump("instruction_positions.json", {"sliding_window": SW, "fit": fit, "by_level": summ,
                                            "examples_policy_farthest": examples,
                                            "caveat": "section token counts are regression estimates (see fit.residual_tokens); "
                                                      "global layers (1 in 4) still attend to the whole prompt"})
    print(json.dumps(fit, indent=1))
    print(json.dumps({lv: {k: v for k, v in s.items() if k != "instructions"} for lv, s in summ.items()}, indent=1))
    print("wrote", rel(p))


if __name__ == "__main__":
    main()
