"""VERIFY RP-1: unscripted / empty look-ups before the first decision, by level; does a needless B question follow them?

Own classification of each pre-decision tool result (every tool_call event that is not the decision itself):
  fallback   == {"result": "No matching records found."}  (wtm/sim.py default handler: no tool_rule, no view handler)
  view_empty == a view-handler page {"total": 0, "offset": .., "items": []} (search over member records matched nothing)
  unknown    == {"error": "Unknown tool ..."}  (tool not offered at this level, or a hallucinated name)
  scripted_empty == any other empty-looking result (e.g. a tool_rule whose result is {"items": [], "total": 0})
The analyst's flag = fallback | view_empty | unknown | (total==0 and no items) | 'No matching records found' in a string.
Outputs: results/v1_unscripted.json, results/v1_unscripted_arms.csv
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vcommon import ROOT, CUR, boot, cluster, cluster_map, dump, episodes, ind, ref, OUT

NOTFOUND_RE = re.compile(
    r"(can'?t|cannot|could ?n'?o?t|unable to|wasn'?t able to|was not able to|did ?n'?o?t|do ?n'?o?t|does ?n'?o?t|"
    r"is ?n'?o?t|are ?n'?o?t|not) (find|locate|see|show|seeing|finding|showing|appear|match|pull up|turn up|returning|return)"
    r"|no (matching|record|records|results?|match|trace|sign|entry|listing)|not (on file|listed|in your|found|showing up|"
    r"able to find)|came back empty|returned nothing|nothing (came|turned) up|unknown tool", re.I)


def classify(res):
    if isinstance(res, dict):
        if res == {"result": "No matching records found."}:
            return "fallback"
        if "error" in res and str(res["error"]).startswith("Unknown tool"):
            return "unknown"
        if set(res) == {"total", "offset", "items"} and res.get("total") == 0:
            return "view_empty"
        if res.get("total") == 0 and not res.get("items"):
            return "scripted_empty"
    if isinstance(res, str) and "No matching records found" in res:
        return "scripted_empty"
    return None


def decision_text(evs):
    for e in evs:
        if e.get("kind") == "message" and not e.get("narration"):
            return (e.get("text") or e.get("raw") or "").replace("\u2019", "'").replace("\u2018", "'")
    return ""


def main():
    cm = cluster_map()
    rows = []
    for e in episodes(keep_events=True):
        if e["dec"] == "error":
            continue
        evs = e.pop("events")
        kinds = Counter()
        n_look = 0
        for ev in evs:
            if ev.get("kind") != "tool_call":
                continue
            tk = ev.get("tool_kind")
            if tk in ("ui", "commit"):      # the decision itself
                break
            n_look += 1
            c = classify(ev.get("result"))
            if c:
                kinds[c] += 1
        e.update({"n_look": n_look, "fallback": int(kinds["fallback"] > 0), "view_empty": int(kinds["view_empty"] > 0),
                  "unknown": int(kinds["unknown"] > 0), "scripted_empty": int(kinds["scripted_empty"] > 0)})
        e["any_empty"] = int(sum(kinds.values()) > 0)                       # analyst-equivalent flag
        e["sim_gap"] = int(kinds["fallback"] > 0 or kinds["view_empty"] > 0)  # simulator/world-state only
        e.update(ind(e["dec"], e["twin"]))
        txt = decision_text(evs)
        e["notfound_text"] = int(bool(NOTFOUND_RE.search(txt)))
        e["ref"] = ref(e["path"], e["line"])
        e["cluster"] = cluster(e["pair"], cm)
        del e["path"]
        rows.append(e)
    df = pd.DataFrame(rows)
    for c in ("needless", "literal"):
        df[c] = df[c].fillna(0).astype(int)      # needless only meaningful on B rows, literal only on A rows
    out = {"n_episodes": int(len(df))}

    # (1) pooled share of B episodes with an empty result before the decision, by level (analyst: 2.1 / 15.3 / 19.5 %)
    pooled = []
    for (lv, tw), g in df.groupby(["level", "twin"]):
        rec = {"level": lv, "twin": tw, "n": int(len(g)), "share_any_empty": round(g.any_empty.mean(), 4),
               "share_fallback": round(g.fallback.mean(), 4), "share_view_empty": round(g.view_empty.mean(), 4),
               "share_unknown": round(g.unknown.mean(), 4), "share_scripted_empty": round(g.scripted_empty.mean(), 4),
               "mean_lookups": round(g.n_look.mean(), 3)}
        if tw == "B":
            k = int(g.needless.sum())
            rec.update({"needless_if_empty": round(g[g.any_empty == 1].needless.mean(), 4) if g.any_empty.sum() else None,
                        "needless_if_not": round(g[g.any_empty == 0].needless.mean(), 4),
                        "n_needless": k, "n_needless_after_empty": int((g.needless & g.any_empty).sum()),
                        "share_needless_after_empty": round(float((g.needless & g.any_empty).sum()) / max(1, k), 4),
                        "share_needless_after_simgap": round(float((g.needless & g.sim_gap).sum()) / max(1, k), 4)})
        pooled.append(rec)
    out["pooled_by_level"] = pooled
    ex_st = df[(df.src != "grid_open_seedtools_P0") & (df.twin == "B")]
    out["pooled_B_without_seedtools"] = []
    for lv, g in ex_st.groupby("level"):
        k = int(g.needless.sum())
        out["pooled_B_without_seedtools"].append({
            "level": lv, "n": int(len(g)), "share_any_empty": round(g.any_empty.mean(), 4),
            "needless_if_empty": round(g[g.any_empty == 1].needless.mean(), 4) if g.any_empty.sum() else None,
            "needless_if_not": round(g[g.any_empty == 0].needless.mean(), 4), "n_needless": k,
            "n_needless_after_empty": int((g.needless * g.any_empty).sum()),
            "share_needless_after_empty": round(float((g.needless * g.any_empty).sum()) / max(1, k), 4)})

    # (2) composition confound: which sources feed each level (pooled numbers mix different model sets)
    comp = df[df.twin == "B"].groupby(["level", "fam"]).size().unstack(fill_value=0)
    out["B_episodes_by_level_and_family"] = {lv: {k: int(v) for k, v in r.items() if v} for lv, r in comp.iterrows()}

    # (3) within-arm (same model, same family, same prompt, same scenario ids) share of empties L0r -> L1 / L2
    arms = []
    for (fam, model, pr), g in df[df.twin == "B"].groupby(["fam", "model", "prompt"]):
        base = g[g.level == "L0r"].set_index("id")
        for lv in ("L1", "L2"):
            oth = g[g.level == lv].set_index("id")
            ids = sorted(set(base.index) & set(oth.index))
            if len(ids) < 10:
                continue
            dn = oth.loc[ids, "needless"].to_numpy(float) - base.loc[ids, "needless"].to_numpy(float)
            de = (oth.loc[ids, "needless"] * oth.loc[ids, "any_empty"]).to_numpy(float) - \
                 (base.loc[ids, "needless"] * base.loc[ids, "any_empty"]).to_numpy(float)
            mu, lo, hi = boot(dn, oth.loc[ids, "cluster"].to_numpy(), seed=11)
            mue, loe, hie = boot(de, oth.loc[ids, "cluster"].to_numpy(), seed=12)
            arms.append({"fam": fam, "model": model, "prompt": pr, "to": lv, "n": len(ids),
                         "empty_L0r": round(base.loc[ids, "any_empty"].mean(), 4),
                         "empty_to": round(oth.loc[ids, "any_empty"].mean(), 4),
                         "simgap_to": round(oth.loc[ids, "sim_gap"].mean(), 4),
                         "needless_L0r": round(base.loc[ids, "needless"].mean(), 4),
                         "needless_to": round(oth.loc[ids, "needless"].mean(), 4),
                         "d_needless": round(mu, 4), "ci_cluster": [round(lo, 4), round(hi, 4)],
                         "d_via_empty": round(mue, 4), "ci_via_empty_cluster": [round(loe, 4), round(hie, 4)],
                         "d_without_empty": round(mu - mue, 4)})
    arms = pd.DataFrame(arms)
    arms.to_csv(os.path.join(OUT, "v1_unscripted_arms.csv"), index=False)
    out["arms_n"] = int(len(arms))
    out["arms_empty_rises"] = int((arms.empty_to > arms.empty_L0r).sum())
    out["arms_median_empty_L0r"] = float(arms.empty_L0r.median())
    out["arms_median_empty_to"] = {lv: float(arms[arms.to == lv].empty_to.median()) for lv in ("L1", "L2")}
    dsv = arms[(arms.model == "DeepSeek-V3.2") & (arms.fam == "variants") & (arms.prompt == "P0") & (arms.to == "L2")]
    out["deepseek_variants_P0_L2"] = dsv.to_dict(orient="records")

    # (4) the panel (L2/P2 hard-set mining, 6 models): share of B episodes with empties and of needless asks after one
    pm = []
    for model, g in df[(df.fam == "panel_mined") & (df.twin == "B")].groupby("model"):
        k = int(g.needless.sum())
        pm.append({"model": model, "n_B": int(len(g)), "share_empty": round(g.any_empty.mean(), 4),
                   "share_simgap": round(g.sim_gap.mean(), 4), "B_needless": round(g.needless.mean(), 4),
                   "share_needless_after_empty": round(float((g.needless & g.any_empty).sum()) / max(1, k), 4)})
    out["panel_mined_B"] = pm

    # (5) mechanism check: do needless asks that follow an empty look-up talk about not finding something?
    b = df[(df.twin == "B") & (df.needless == 1) & df.level.isin(["L1", "L2", "L2s", "L3"])]
    out["needless_text_mentions_notfound"] = {
        "after_empty": {"n": int(b.any_empty.sum()), "share": round(b[b.any_empty == 1].notfound_text.mean(), 4)},
        "after_simgap": {"n": int(b.sim_gap.sum()), "share": round(b[b.sim_gap == 1].notfound_text.mean(), 4)},
        "no_empty": {"n": int((b.any_empty == 0).sum()), "share": round(b[b.any_empty == 0].notfound_text.mean(), 4)}}
    # same for L0r
    b0 = df[(df.twin == "B") & (df.needless == 1) & (df.level == "L0r")]
    out["needless_text_mentions_notfound_L0r"] = {
        "after_empty": {"n": int(b0.any_empty.sum()), "share": round(b0[b0.any_empty == 1].notfound_text.mean(), 4)},
        "no_empty": {"n": int((b0.any_empty == 0).sum()), "share": round(b0[b0.any_empty == 0].notfound_text.mean(), 4)}}

    # (6) look-up volume as a confound: needless rate by number of look-ups, within L2 B episodes, split by empty
    l2b = df[(df.twin == "B") & (df.level == "L2")].copy()
    l2b["n_look_c"] = l2b.n_look.clip(upper=4)
    vol = l2b.groupby(["n_look_c", "any_empty"]).needless.agg(["mean", "size"]).reset_index()
    out["L2_B_needless_by_nlookups_and_empty"] = vol.round(4).to_dict(orient="records")

    # (7) A side: share of informed vs generic asks after an empty result (does an empty look-up also break A?)
    a = df[(df.twin == "A") & df.level.isin(["L1", "L2"])]
    out["A_L1L2_by_empty"] = {str(k): {"n": int(len(g)), "ask": round(g.ask.mean(), 4), "informed": round(g.informed.mean(), 4),
                                       "literal": round(g.literal.mean(), 4)} for k, g in a.groupby("any_empty")}

    # (8) examples cited by the analyst
    want = ["experiments/replication/results/grid_open_variants.jsonl:4974", "experiments/replication/results/grid_open_variants.jsonl:4965"]
    ex = df[df.ref.isin(want) | df.ref.str.endswith("panel_hard_claude.jsonl:113") | df.ref.str.endswith("panel_hard_claude.jsonl:118")]
    out["analyst_examples"] = ex[["ref", "model", "id", "level", "prompt", "dec", "n_look", "fallback", "view_empty", "unknown",
                                  "any_empty", "notfound_text"]].to_dict(orient="records")
    # fresh examples: B needless after a fallback result at L1/L2 whose text says it found nothing
    fr = df[(df.twin == "B") & (df.needless == 1) & (df.fallback == 1) & (df.notfound_text == 1) & df.level.isin(["L1", "L2"])]
    out["fresh_examples"] = fr.sample(min(6, len(fr)), random_state=3)[["ref", "model", "id", "level", "prompt", "dec"]].to_dict(orient="records")

    # (9) training views: any empty or not-found tool result in any prompt?
    pat = re.compile(r"No matching records found|\"total\": 0|\"items\": \[\]|Unknown tool|not found|No matching", re.I)
    strict_pat = re.compile(r"No matching records found|\"total\": 0[,}]|Unknown tool")
    tv = {}
    for fn in ("sft_train", "sft_val", "grpo_train", "dpo_train"):
        n = hit = 0
        hits = []
        with open(os.path.join(CUR, fn + ".jsonl")) as fh:
            for i, l in enumerate(fh):
                r = json.loads(l)
                n += 1
                trs = [m["content"] for m in r["prompt"] if m["role"] == "user" and "<tool_result" in m["content"]]
                h = [pat.search(t).group(0) for t in trs if pat.search(t)]
                if h:
                    hit += 1
                    if len(hits) < 5:
                        hits.append({"line": i, "id": r["scenario_id"], "match": h[:2]})
        n_strict = 0
        with open(os.path.join(CUR, fn + ".jsonl")) as fh:
            for l in fh:
                r = json.loads(l)
                n_strict += int(any(strict_pat.search(m["content"]) for m in r["prompt"]
                                    if m["role"] == "user" and "<tool_result" in m["content"]))
        tv[fn] = {"rows": n, "rows_with_fallback_or_emptyview_or_unknown": n_strict,
                  "rows_with_any_empty_or_notfound_text": hit, "first_hits": hits}
    out["training_views"] = tv
    # which look-up tools do training views call, and how many of them have a tool_rule (scripted)? (all of them should)
    used = Counter()
    with open(os.path.join(CUR, "grpo_train.jsonl")) as fh:
        for l in fh:
            r = json.loads(l)
            for m in r["prompt"]:
                if m["role"] == "assistant":
                    mm = re.search(r'"name":\s*"([a-z_]+)"', m["content"])
                    if mm:
                        used[mm.group(1)] += 1
    out["grpo_prefix_lookup_tools"] = dict(used.most_common())
    dump(out, "v1_unscripted.json")
    df.drop(columns=[]).to_csv(os.path.join(OUT, "v1_episode_flags.csv"), index=False)
    print(json.dumps({k: out[k] for k in ("pooled_B_without_seedtools", "panel_mined_B", "needless_text_mentions_notfound",
                                          "needless_text_mentions_notfound_L0r", "training_views", "deepseek_variants_P0_L2",
                                          "analyst_examples", "arms_n", "arms_empty_rises", "arms_median_empty_L0r",
                                          "arms_median_empty_to", "A_L1L2_by_empty")}, indent=1, default=float))


if __name__ == "__main__":
    main()
