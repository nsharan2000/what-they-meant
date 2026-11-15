"""Do the M1 policy's GRPO samples reproduce MEMORISED targets verbatim? Completions were not logged (log_completions=False), but each
score record keeps meta.n_tokens = the generated ids incl. the closing <|END_RESPONSE|><|END_OF_TURN_TOKEN|> (end_id 6 on every
sample). A verbatim copy of a training target has exactly target_tokens + 2 ids (training/guards.py: trained span = target +
those two tokens). So per sample we compare n_tokens with
  own   = this prompt's own SFT target length + 2          (correct twin, memorised wording)
  twin  = the other twin's target length + 2 (same pair)    (wrong twin, memorised wording)
Exact-length matches by chance are measured on the same samples against targets of OTHER pairs (same decision kind).

    python3 experiments/dataset_audit/composition-memorization/regurgitation.py
Output: results/regurgitation.json
"""
from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd

from common import GRPO_SCORES, TOKEN_STATS, dump, load_jsonl, load_view, target_kind, target_of


def main():
    rows = load_view("sft_train")
    ts = [t for t in load_jsonl(TOKEN_STATS) if t["file"] == "sft_train.jsonl"]
    tgt_len, tgt_kind = {}, {}
    for t, r in zip(ts, rows):
        if r["turn"] == "decision":
            tgt_len[r["scenario_id"]] = t["completion_tokens"] + 2     # the same target at every level of a scenario
            tgt_kind[r["scenario_id"]] = target_kind(target_of(r), r["env_id"]).split(":")[0]
    # sanity: one target length per scenario
    per = collections.defaultdict(set)
    for t, r in zip(ts, rows):
        if r["turn"] == "decision":
            per[r["scenario_id"]].add(t["completion_tokens"])
    multi = [k for k, v in per.items() if len(v) > 1]
    sc = load_jsonl(GRPO_SCORES)
    recs = []
    for x in sc:
        m = x["meta"]
        sid = m["scenario_id"]
        twin = sid[:-1] + ("B" if sid.endswith("A") else "A")
        d = x["components"]["decision"]
        a_ok = d in ("ask_informed", "tell_informed")
        if sid.endswith("A"):
            chose = "own" if d.startswith(("ask", "tell")) or d == "statement" else ("lookup" if d == "lookup" else "other_twin_kind")
        else:
            chose = "own" if d.startswith(("approve", "commit")) else ("lookup" if d == "lookup" else "other_twin_kind")
        recs.append({"sid": sid, "level": m["level"], "decision": d, "n": m["n_tokens"], "end_id": m["end_id"], "chose": chose,
                     "own": tgt_len.get(sid), "twin": tgt_len.get(twin), "own_kind": tgt_kind.get(sid), "twin_kind": tgt_kind.get(twin)})
    df = pd.DataFrame(recs)
    df = df[df.own.notna() & df.twin.notna()].copy()
    df["eq_own"] = df.n == df.own
    df["eq_twin"] = df.n == df.twin
    # chance: same samples vs targets of other pairs with the same kind as the sample's chosen side
    pool = collections.defaultdict(list)
    for s, k in tgt_kind.items():
        pool[k].append((s[:-1], tgt_len[s]))
    rng = np.random.default_rng(0)

    def chance(r, kind):
        cands = [L for p, L in pool[kind] if p != r.sid[:-1]]
        return float(np.mean([r.n == L for L in cands])) if cands else np.nan
    df["chance_own_kind"] = [chance(r, r.own_kind) for r in df.itertuples()]
    df["chance_twin_kind"] = [chance(r, r.twin_kind) for r in df.itertuples()]
    out = {"samples": int(len(df)), "scenarios_with_varying_target_length_across_levels": multi,
           "end_id_counts": dict(collections.Counter(df.end_id.astype(str)))}
    for chose, g in df.groupby("chose"):
        out[f"chose_{chose}"] = {"n": int(len(g)), "len_equals_own_target": round(float(g.eq_own.mean()), 3),
                                 "len_equals_twin_target": round(float(g.eq_twin.mean()), 3),
                                 "chance_match_other_pairs_own_kind": round(float(g.chance_own_kind.mean()), 3),
                                 "chance_match_other_pairs_twin_kind": round(float(g.chance_twin_kind.mean()), 3)}
    # pairs where own and twin lengths differ (so a match identifies which target was copied)
    dist = df[df.own != df.twin]
    out["distinct_length_pairs"] = {
        "n": int(len(dist)),
        "own_side_samples_matching_own_len": round(float(dist[dist.chose == "own"].eq_own.mean()), 3),
        "own_side_samples_matching_twin_len": round(float(dist[dist.chose == "own"].eq_twin.mean()), 3),
        "flipped_samples_matching_twin_len": round(float(dist[dist.chose == "other_twin_kind"].eq_twin.mean()), 3),
        "flipped_samples_matching_own_len": round(float(dist[dist.chose == "other_twin_kind"].eq_own.mean()), 3),
        "n_flipped": int((dist.chose == "other_twin_kind").sum()), "n_own": int((dist.chose == "own").sum())}
    out["examples_flipped_exact_twin_len"] = dist[(dist.chose == "other_twin_kind") & dist.eq_twin].head(5)[
        ["sid", "level", "decision", "n", "own", "twin"]].to_dict("records")
    out["note"] = ("length equality is necessary, not sufficient, for a verbatim copy; the chance columns give the base rate of an "
                   "exact length match against other pairs' targets of the same kind")
    dump(out, "regurgitation.json")
    print(json.dumps({k: v for k, v in out.items() if k != "examples_flipped_exact_twin_len"}, indent=0))


if __name__ == "__main__":
    main()
