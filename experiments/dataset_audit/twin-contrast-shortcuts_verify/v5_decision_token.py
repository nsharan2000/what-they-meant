"""V5 (verifier): is the ask/act choice made at the first target token, and how much of the SFT loss can it carry?
Uses the exact Tiny Aya token counts (experiments/dataset_audit/results/token_stats.jsonl: completion_tokens, completion_first5).
Also: the char-level share of each decision prompt that differs from its twin (difflib on the hint-stripped serialized prompt),
converted to tokens with the row's own chars/token ratio, and the end-of-turn token the trainer adds (not in completion_tokens).

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v5_decision_token.py -> results/v5_decision_token.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import difflib  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import T, curated, nohint, save, token_stats  # noqa: E402


def ser(prompt):
    return "\n".join([nohint(prompt[0]["content"])] + [m["content"] for m in prompt[1:]])


def main():
    out = {}
    ts = token_stats()
    sft = curated("sft_train")
    first = defaultdict(Counter)
    ntok = defaultdict(list)
    for r in sft:
        st = ts[("sft_train", r["_i"])]
        assert st["scenario_id"] == r["scenario_id"] and st["turn"] == r["turn"]
        key = f"{T(r)}_{'decision' if r['turn'] == 'decision' else 'lookup'}"
        first[key][st["completion_first5"][0] == "<"] += 1
        ntok[key].append(st["completion_tokens"])
    out["first_token_is_lt"] = {k: {"yes": c[True], "no": c[False]} for k, c in first.items()}
    out["share_rows_first_token_lt"] = round(sum(c[True] for c in first.values()) / len(sft), 3)
    allt = sum(sum(v) for v in ntok.values())
    out["target_tokens"] = {k: {"n": len(v), "mean": round(float(np.mean(v)), 2), "mean_inv": round(float(np.mean([1 / x for x in v])), 4)}
                            for k, v in ntok.items()}
    out["total_target_tokens"] = allt
    ln2 = math.log(2)
    for k in ("A_decision", "B_decision"):
        v = ntok[k]
        out[f"coinflip_added_mean_loss_{k}"] = {
            "ln2_over_mean_tokens": round(ln2 / np.mean(v), 4),
            "mean_of_ln2_over_tokens": round(float(np.mean([ln2 / x for x in v])), 4),
            "with_end_of_turn_token": round(ln2 / (np.mean(v) + 1), 4)}
    # if EVERY decision row's first token were a coin flip, the token-weighted mean loss over the whole SFT set rises by:
    n_dec = len(ntok["A_decision"]) + len(ntok["B_decision"])
    out["coinflip_all_decisions_added_to_global_mean_loss"] = round(n_dec * ln2 / (allt + len(sft)), 4)
    out["decision_token_share_of_all_target_tokens"] = round(n_dec / (allt + len(sft)), 4)

    # twin-differing share of the decision prompt (char diff, converted with each row's chars/token)
    m = defaultdict(dict)
    for r in sft:
        if r["turn"] == "decision" and T(r) in "AB":
            m[(r["pair_id"], r["level"])][T(r)] = r
    shares, tok_diff, far = [], [], []
    for (pid, lv), d in m.items():
        if "A" not in d or "B" not in d:
            continue
        sa, sb = ser(d["A"]["prompt"]), ser(d["B"]["prompt"])
        sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
        diff_a = sum(i2 - i1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
        diff_b = sum(j2 - j1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
        first_diff_a = min([i1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"] or [len(sa)])
        cpt_a = len(sa) / ts[("sft_train", d["A"]["_i"])]["prompt_tokens"]
        cpt_b = len(sb) / ts[("sft_train", d["B"]["_i"])]["prompt_tokens"]
        shares.append((diff_a / len(sa) + diff_b / len(sb)) / 2)
        tok_diff.append(diff_a / cpt_a + diff_b / cpt_b)
        far.append((len(sa) - first_diff_a) / cpt_a)      # tokens from the earliest differing char to the end (A side)
    out["twin_prompt_diff"] = {"n_pairs": len(shares), "median_share": round(float(np.median(shares)), 4),
                               "p90_share": round(float(np.percentile(shares, 90)), 4),
                               "median_tokens_both_sides": round(float(np.median(tok_diff)), 1),
                               "median_tokens_from_first_diff_to_end": round(float(np.median(far)), 1),
                               "share_pairs_first_diff_beyond_4096_tokens_from_end": round(float(np.mean([x > 4096 for x in far])), 3)}
    save(out, "v5_decision_token.json")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
