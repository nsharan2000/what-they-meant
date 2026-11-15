"""Where do a pair's two decision prompts (twin A should-ask vs twin B should-act, same level) differ? That is the only signal the
decision can be learned from. Per (pair, level) in SFT train + val: is the member request identical, are the look-up calls
identical, do the look-up results differ, does the system prompt differ once the P2 hint is removed (and by how many characters)?

    python3 experiments/dataset_audit/composition-memorization/twin_contrast.py
Output: results/twin_contrast.csv (one line per pair x level x split) and results/twin_contrast.json (summary).
"""
from __future__ import annotations

import collections
import difflib
import os
import sys

import pandas as pd

from common import OUT, ROOT, dump, load_view, target_of

sys.path.insert(0, ROOT)
from wtm.render import golden_text  # noqa: E402

HINT = golden_text("P2")


def strip_hint(s: str) -> str:
    return s.replace("\n\n" + HINT, "").replace(HINT, "")


def char_diff(a: str, b: str):
    """characters inserted/deleted/replaced between a and b (line-level opcodes, then char-level inside replaced lines)."""
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False)
    n, first_from_end = 0, None
    la = a.splitlines()
    off = [0]
    for line in la:
        off.append(off[-1] + len(line) + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        n += max(sum(len(x) for x in la[i1:i2]), sum(len(x) for x in b.splitlines()[j1:j2]))
        pos = off[i1]
        d = len(a) - pos
        first_from_end = d if first_from_end is None else max(first_from_end, d)   # earliest difference, as chars before the end
    return n, first_from_end


def main():
    recs = []
    for split in ("sft_train", "sft_val"):
        rows = load_view(split)
        dec = {}
        for i, r in enumerate(rows):
            if r["turn"] == "decision" and r["twin"] in ("A_should_ask", "B_should_act"):
                dec[(r["pair_id"], r["twin"][:1], r["level"])] = (i, r)
        for (pid, tw, lv), (ia, a) in sorted(dec.items()):
            if tw != "A" or (pid, "B", lv) not in dec:
                continue
            ib, b = dec[(pid, "B", lv)]
            pa, pb = a["prompt"], b["prompt"]
            sys_a, sys_b = strip_hint(pa[0]["content"]), strip_hint(pb[0]["content"])
            sys_n, sys_first = char_diff(sys_a, sys_b) if sys_a != sys_b else (0, None)
            calls_a = [m["content"] for m in pa[2::2]]
            calls_b = [m["content"] for m in pb[2::2]]
            res_a = [m["content"] for m in pa[3::2]]
            res_b = [m["content"] for m in pb[3::2]]
            rec = {"split": split, "pair_id": pid, "level": lv, "row_A": ia, "row_B": ib, "ps_A": a["prompt_setting"],
                   "ps_B": b["prompt_setting"], "seed": not pid.startswith("V"), "env": a["env_id"],
                   "request_same": pa[1]["content"] == pb[1]["content"], "calls_same": calls_a == calls_b,
                   "results_same": res_a == res_b, "system_same": sys_a == sys_b, "system_diff_chars": sys_n,
                   "system_first_diff_chars_before_end": sys_first, "system_chars": len(sys_a),
                   "target_A": target_of(a)[:140], "target_B": target_of(b)[:140],
                   "result_chars_A": sum(len(x) for x in res_a),
                   "result_diff_chars": (sum(1 for op in difflib.SequenceMatcher(None, "".join(res_a), "".join(res_b),
                                                                                 autojunk=False).get_opcodes() if op[0] != "equal")
                                         if res_a != res_b else 0),
                   "prompt_chars_A": sum(len(m["content"]) for m in pa)}
            # where can the decision be read? (first matching rule)
            if not rec["request_same"]:
                cue = "request_differs"
            elif not rec["calls_same"]:
                cue = "lookup_call_differs"
            elif not rec["results_same"] and rec["system_same"]:
                cue = "only_lookup_result_differs"
            elif not rec["results_same"]:
                cue = "lookup_result_and_system_differ"
            elif not rec["system_same"]:
                cue = "only_system_differs"
            else:
                cue = "IDENTICAL_PROMPTS"
            rec["cue"] = cue
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "twin_contrast.csv"), index=False)
    summ = {}
    for split, g in df.groupby("split"):
        pairs = g.groupby("pair_id").cue.agg(lambda s: collections.Counter(s).most_common(1)[0][0])
        summ[split] = {"pair_levels": int(len(g)), "pairs": int(g.pair_id.nunique()),
                       "cue_by_pair_level": dict(collections.Counter(g.cue)),
                       "cue_by_pair_majority": dict(collections.Counter(pairs)),
                       "request_same_share": round(float(g.request_same.mean()), 3),
                       "only_system_or_result_differs_share": round(float(g.cue.isin(
                           ["only_lookup_result_differs", "lookup_result_and_system_differ", "only_system_differs"]).mean()), 3),
                       "system_diff_chars_when_system_differs": g[~g.system_same].system_diff_chars.describe().round(1).to_dict(),
                       "only_result_differs": {
                           "result_chars_median": float(g[g.cue == "only_lookup_result_differs"].result_chars_A.median()),
                           "result_share_of_prompt_chars_median": round(float((g[g.cue == "only_lookup_result_differs"].result_chars_A
                                                                               / g[g.cue == "only_lookup_result_differs"].prompt_chars_A).median()), 4),
                           "differing_char_runs_median": float(g[g.cue == "only_lookup_result_differs"].result_diff_chars.median())},
                       "examples": {c: g[g.cue == c][["pair_id", "level", "row_A", "row_B"]].head(3).to_dict("records")
                                    for c in sorted(g.cue.unique())}}
    dump(summ, "twin_contrast.json")
    print(df.groupby(["split", "cue"]).size())


if __name__ == "__main__":
    main()
