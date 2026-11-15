"""CM-1 / CM-3 / CM-4 token composition, re-derived independently.

Inputs: dataset/curated_v121/{sft_train,sft_val}.jsonl and experiments/dataset_audit/results/token_stats.jsonl (exact Tiny Aya
token counts + first 5 target tokens per row). Trained tokens per row = target tokens + 2 (<|END_RESPONSE|>, <|END_OF_TURN_TOKEN|>),
checked against training guards.json (trained_tokens_mean). Keyword token counts are ESTIMATES (chars / chars-per-token of the
row's own target), because no tokenizer is available here.
Output: results/v1_tokens.json
"""
import json
import os
import re

import numpy as np

from vcommon import ROOT, cur, dump, load_jsonl, target, is_call, call_name

ts = load_jsonl(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))
guards = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m1_sft/guards.json")))
grpo = cur("grpo_train")
scn_by_id = {}
for g in grpo:
    s = json.loads(g["scenario"])
    scn_by_id[s["id"]] = s
# scenario specs for val too (informed_any keywords)
import glob
for f in sorted(glob.glob(os.path.join(ROOT, "dataset/scenarios/*.jsonl"))) + sorted(glob.glob(os.path.join(ROOT, "dataset/scenarios_synth/*.jsonl"))):
    for s in load_jsonl(f):
        scn_by_id.setdefault(s["id"], s)

out = {}
for fname in ("sft_train", "sft_val"):
    rows = cur(fname)
    st = [t for t in ts if t["file"] == fname + ".jsonl"]
    assert len(st) == len(rows), (fname, len(st), len(rows))
    for i, (r, t) in enumerate(zip(rows, st)):
        assert t["i"] == i and t["scenario_id"] == r["scenario_id"] and t["level"] == r["level"] and \
            t["prompt_setting"] == r["prompt_setting"] and t["turn"] == r["turn"], (fname, i)
    comp = np.array([t["completion_tokens"] for t in st])
    trained = comp + 2
    T = int(trained.sum())
    turn = np.array([r["turn"] for r in rows])
    dec = turn == "decision"
    look = ~dec
    tgt = [target(r) for r in rows]
    call = np.array([is_call(x) for x in tgt])
    first_tok = [t["completion_first5"][0] for t in st]
    # sanity: every tool-call target starts with token "<"; no message target starts with "<"
    assert all(ft == "<" for ft, c in zip(first_tok, call) if c)
    n_msg_lt = sum(ft == "<" for ft, c in zip(first_tok, call) if not c)
    dec_call = dec & call
    dec_msg = dec & ~call
    # keyword estimate: first informed_any keyword hit in twin-A message decision targets, tokens ~ chars / (row chars per token)
    kw_tok = 0.0
    kw_rows = 0
    kw_missing = []
    for i in np.where(dec_msg)[0]:
        r = rows[i]
        s = scn_by_id.get(r["scenario_id"])
        txt = tgt[i]
        cpt = len(txt) / max(1, comp[i])
        best = None
        for kw in (s or {}).get("expected", {}).get("informed_any", []):
            m = re.search(re.escape(kw), txt, re.I)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), kw)
        if best:
            kw_rows += 1
            kw_tok += max(1.0, len(best[1]) / cpt)
        else:
            kw_missing.append(r["scenario_id"])
    pivots = int(dec.sum()) + int(dec_call.sum())
    res = {
        "rows": len(rows), "trained_tokens_per_epoch": T, "trained_tokens_mean": round(float(trained.mean()), 2),
        "guards_trained_tokens_mean": guards[fname]["trained_tokens_mean"],
        "decision_rows": int(dec.sum()), "lookup_rows": int(look.sum()),
        "decision_rows_tool_call": int(dec_call.sum()), "decision_rows_message": int(dec_msg.sum()),
        "message_targets_starting_with_lt_token": int(n_msg_lt),
        "share_first_token_of_decision_rows": round(int(dec.sum()) / T, 4),
        "share_first_token_plus_tool_name_token": round(pivots / T, 4),
        "keyword_tokens_est": round(kw_tok, 1), "keyword_rows": kw_rows, "keyword_not_found_rows": len(kw_missing),
        "share_pivots_plus_keyword_est": round((pivots + kw_tok) / T, 4),
        "lookup_trained_tokens": int(trained[look].sum()), "share_lookup_tokens": round(float(trained[look].sum()) / T, 4),
        "decision_trained_tokens": int(trained[dec].sum()), "decision_row_trained_mean": round(float(trained[dec].mean()), 2),
        "decision_msg_row_trained_mean": round(float(trained[dec_msg].mean()), 2) if dec_msg.any() else None,
        "decision_call_row_trained_mean": round(float(trained[dec_call].mean()), 2) if dec_call.any() else None,
        "end_tokens_share": round(2 * len(rows) / T, 4),
        "max_prompt_plus_trained": int(max(t["prompt_tokens"] + c + 2 for t, c in zip(st, comp))),
        "guards_max_len": guards[fname]["max_len"],
        "first_token_distribution_decision_msg": dict(sorted(
            {k: sum(1 for i in np.where(dec_msg)[0] if first_tok[i] == k) for k in set(first_tok[i] for i in np.where(dec_msg)[0])}.items(),
            key=lambda kv: -kv[1])[:15]),
    }
    if fname == "sft_train":
        # tool names in decision targets and in lookup targets
        res["decision_call_names"] = dict(sorted({n: sum(1 for i in np.where(dec_call)[0] if call_name(tgt[i]) == n)
                                                  for n in {call_name(tgt[i]) for i in np.where(dec_call)[0]}}.items()))
        # pivot weighting arithmetic
        P = pivots
        res["pivot_weighting"] = {str(w): round(w * P / (w * P + (T - P)), 4) for w in (1, 5, 10, 20)}
        # if only one lookup row per scenario were kept
        seen = set()
        keep = np.ones(len(rows), bool)
        for i, r in enumerate(rows):
            if r["turn"].startswith("lookup"):
                k = (r["scenario_id"], r["turn"])
                if k in seen:
                    keep[i] = False
                seen.add(k)
        res["one_lookup_row_per_scenario_turn"] = {"rows": int(keep.sum()), "trained_tokens": int(trained[keep].sum()),
                                                    "pivot_share": round(P / float(trained[keep].sum()), 4)}
    out[fname] = res
dump(out, "v1_tokens.json")
print(json.dumps(out, indent=1))
