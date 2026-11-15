"""6: wording. How much of a validation target's wording occurs in the training targets, compared with a training target's own
coverage by OTHER training clusters (leave-one-cluster-out = the same "new source post" condition the val split has); template
phrases; style of seed targets (user-authored) vs variant targets (open teacher model); within-cluster near-duplication
(a variant's target vs its seed's target).

Word n-grams on lower-cased words/numbers/punctuation (common.words). Decision targets only unless stated.

    python3 experiments/dataset_audit/composition-memorization/wording_overlap.py
Output: results/wording_overlap.json, results/wording_per_target.csv
"""
from __future__ import annotations

import collections
import difflib
import json
import os
import re

import numpy as np
import pandas as pd

from common import OUT, ROOT, dump, load_jsonl, load_view, ngrams, source_of, target_kind, target_of, words

TS = os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")


def free_text(t: str) -> str:
    if t.startswith("<tool_call>"):
        try:
            return " ".join(str(v) for v in json.loads(t.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0])["arguments"].values())
        except Exception:  # noqa: BLE001
            return t
    return t


def main():
    sp = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    clus = {p: cid for cid, c in sp["clusters"].items() for p in c["pairs"]}
    ts = {(t["file"], t["i"]): t for t in load_jsonl(TS)}
    recs = []
    for view in ("sft_train", "sft_val"):
        for i, r in enumerate(load_view(view)):
            t = target_of(r)
            k = target_kind(t, r["env_id"]).split(":")[0]
            recs.append({"view": view, "i": i, "scenario_id": r["scenario_id"], "pair_id": r["pair_id"], "cluster": clus[r["pair_id"]],
                         "twin": r["twin"][:1], "turn": r["turn"], "kind": k, "source": source_of(r["scenario_id"]),
                         "target": t, "free": free_text(t), "tokens": ts[(f"{view}.jsonl", i)]["completion_tokens"]})
    df = pd.DataFrame(recs)
    d = df[df.turn == "decision"].drop_duplicates(["view", "target"]).copy()
    d["w"] = d.free.map(words)
    out = {"distinct_decision_targets": {v: int(n) for v, n in d.groupby("view").size().items()}}

    def gramset(sub, n):
        s = set()
        for w in sub.w:
            s.update(ngrams(w, n))
        return s

    def coverage(w, ref, n):
        g = ngrams(w, n)
        return np.nan if not g else sum(x in ref for x in g) / len(g)

    tr = d[d.view == "sft_train"]
    va = d[d.view == "sft_val"]
    cov = {}
    for n in (3, 4):
        ref_all = gramset(tr, n)
        va_cov = va.w.map(lambda w: coverage(w, ref_all, n))
        # leave-one-cluster-out / leave-one-pair-out baselines for train targets
        loco, lopo = [], []
        by_c = {c: gramset(tr[tr.cluster != c], n) for c in tr.cluster.unique()}
        by_p = {p: gramset(tr[tr.pair_id != p], n) for p in tr.pair_id.unique()}
        for r in tr.itertuples():
            loco.append(coverage(r.w, by_c[r.cluster], n))
            lopo.append(coverage(r.w, by_p[r.pair_id], n))
        tr_loco, tr_lopo = pd.Series(loco, index=tr.index), pd.Series(lopo, index=tr.index)
        if n == 4:
            d.loc[va.index, "cov4_vs_train"] = va_cov
            d.loc[tr.index, "cov4_loco"] = tr_loco
            d.loc[tr.index, "cov4_lopo"] = tr_lopo
        row = {}
        for kind in ("message", "ui", "commit", "all"):
            mv = va.kind == kind if kind != "all" else va.kind.notna()
            mt = tr.kind == kind if kind != "all" else tr.kind.notna()
            row[kind] = {"val_vs_train": round(float(va_cov[mv].mean()), 3), "train_leave_cluster_out": round(float(tr_loco[mt].mean()), 3),
                         "train_leave_pair_out": round(float(tr_lopo[mt].mean()), 3), "n_val": int(mv.sum()), "n_train": int(mt.sum())}
        cov[f"{n}gram"] = row
    out["ngram_coverage_mean"] = cov
    # ---- within-cluster near-duplication ------------------------------------------------------------------------------------
    sims = []
    for r in tr[tr.source == "variant"].itertuples():
        same = tr[(tr.cluster == r.cluster) & (tr.source == "seed") & (tr.twin == r.twin)]
        other = tr[(tr.cluster != r.cluster) & (tr.twin == r.twin) & (tr.kind == r.kind)]
        s_same = max((difflib.SequenceMatcher(None, r.w, w2, autojunk=False).ratio() for w2 in same.w), default=np.nan)
        s_oth = max((difflib.SequenceMatcher(None, r.w, w2, autojunk=False).ratio() for w2 in other.w), default=np.nan)
        sims.append({"scenario_id": r.scenario_id, "kind": r.kind, "sim_to_own_seed": s_same, "max_sim_other_cluster": s_oth,
                     "target": r.target[:120]})
    sd = pd.DataFrame(sims).dropna(subset=["sim_to_own_seed"])
    out["variant_vs_own_seed_similarity"] = {
        "n_variant_targets_with_seed_in_train": int(len(sd)),
        "mean_sim_to_own_seed": round(float(sd.sim_to_own_seed.mean()), 3),
        "mean_max_sim_other_cluster": round(float(sd.max_sim_other_cluster.mean()), 3),
        "share_sim_to_own_seed_ge_0.6": round(float((sd.sim_to_own_seed >= 0.6).mean()), 3),
        "share_sim_to_own_seed_ge_0.8": round(float((sd.sim_to_own_seed >= 0.8).mean()), 3),
        "by_kind": {k: round(float(g.sim_to_own_seed.mean()), 3) for k, g in sd.groupby("kind")},
        "top_examples": sd.sort_values("sim_to_own_seed", ascending=False).head(4).to_dict("records")}
    # ---- template phrases in chat-message targets -------------------------------------------------------------------------
    msgs = d[d.kind == "message"]
    cnt = collections.Counter()
    for w in msgs[msgs.view == "sft_train"].w:
        cnt.update(set(ngrams(w, 4)))
    tmpl = [(" ".join(g), n) for g, n in cnt.most_common(40) if not all(re.fullmatch(r"\W", x) for x in g)][:15]
    out["top_train_message_4grams"] = [{"phrase": p, "train_targets": n,
                                         "val_targets": int(sum(1 for w in msgs[msgs.view == "sft_val"].w if tuple(p.split()) in set(ngrams(w, 4))))}
                                        for p, n in tmpl]
    # ---- style: seed (user-authored) vs variant (open teacher) ---------------------------------------------------------------
    def style(sub):
        txt = sub.free
        return {"n": int(len(sub)), "tokens_mean": round(float(sub.tokens.mean()), 1), "words_mean": round(float(sub.w.map(len).mean()), 1),
                "chars_mean": round(float(txt.str.len().mean()), 1),
                "ends_with_question_mark": round(float(txt.str.strip().str.endswith("?").mean()), 3),
                "question_marks_mean": round(float(txt.str.count(r"\?").mean()), 2),
                "sentences_mean": round(float(txt.map(lambda s: len([x for x in re.split(r"(?<=[.?!])\s+", s.strip()) if x])).mean()), 2),
                "starts_The": round(float(txt.str.startswith("The ").mean()), 3),
                "has_should_i": round(float(txt.str.lower().str.contains("should i").mean()), 3),
                "has_do_you_want": round(float(txt.str.lower().str.contains("do you want").mean()), 3),
                "has_did_you_mean": round(float(txt.str.lower().str.contains("did you mean").mean()), 3),
                "has_digit": round(float(txt.str.contains(r"\d").mean()), 3)}
    sty = {}
    for view in ("sft_train", "sft_val"):
        for kind in ("message", "ui"):
            for src in ("seed", "variant"):
                sub = d[(d.view == view) & (d.kind == kind) & (d.source == src)]
                if len(sub):
                    sty[f"{view}|{kind}|{src}"] = style(sub)
    out["style"] = sty
    # val decision targets with the lowest 4-gram coverage (most novel wording)
    vv = d[d.view == "sft_val"].sort_values("cov4_vs_train")
    out["val_most_novel"] = vv.head(4)[["scenario_id", "kind", "cov4_vs_train", "target"]].to_dict("records")
    out["val_least_novel"] = vv.tail(3)[["scenario_id", "kind", "cov4_vs_train", "target"]].to_dict("records")
    # composition of the val loss: rows and trained tokens by turn (val eval loss is token-averaged over these)
    v_all = df[df.view == "sft_val"]
    t_all = df[df.view == "sft_train"]
    out["val_vs_train_trained_token_mix"] = {
        "val_lookup_token_share": round(float((v_all[v_all.turn != "decision"].tokens + 2).sum() / (v_all.tokens + 2).sum()), 3),
        "train_lookup_token_share": round(float((t_all[t_all.turn != "decision"].tokens + 2).sum() / (t_all.tokens + 2).sum()), 3),
        "val_decision_kind_rows": dict(collections.Counter(v_all[v_all.turn == "decision"].kind)),
        "train_decision_kind_rows": dict(collections.Counter(t_all[t_all.turn == "decision"].kind))}
    d.drop(columns=["w"]).to_csv(os.path.join(OUT, "wording_per_target.csv"), index=False)
    dump(out, "wording_overlap.json")
    print(json.dumps({k: out[k] for k in ("ngram_coverage_mean", "variant_vs_own_seed_similarity", "style", "top_train_message_4grams",
                                          "val_vs_train_trained_token_mix")}, indent=0, default=str)[:9000])


if __name__ == "__main__":
    main()
