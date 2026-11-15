"""1-3: row composition, target repetition, look-up share, and an oracle check (every SFT decision target graded by wtm.reward).

    python3 experiments/dataset_audit/composition-memorization/composition.py
Output: results/composition.json (+ results/targets_train.csv: one line per distinct SFT-train target).
"""
from __future__ import annotations

import collections
import statistics

import pandas as pd

from common import OUT, dump, grade, load_view, scenarios, source_of, target_kind, target_of

C = collections.Counter


def dist(values):
    v = sorted(values)
    if not v:
        return None
    return {"n": len(v), "min": v[0], "p25": v[len(v) // 4], "median": statistics.median(v), "mean": round(statistics.mean(v), 2),
            "p75": v[(3 * len(v)) // 4], "max": v[-1]}


def main():
    scn = scenarios()
    views = {k: load_view(k) for k in ("sft_train", "sft_val", "dpo_train", "grpo_train")}
    res = {}
    for name in ("sft_train", "sft_val"):
        rows = views[name]
        df = pd.DataFrame([{"i": i, "scenario_id": r["scenario_id"], "pair_id": r["pair_id"], "twin": r["twin"][:1],
                            "twin_full": r["twin"], "env": r["env_id"], "level": r["level"], "ps": r["prompt_setting"],
                            "turn": r["turn"], "source": source_of(r["scenario_id"]), "target": target_of(r),
                            "tkind": target_kind(target_of(r), r["env_id"])} for i, r in enumerate(rows)])
        df["tkind_coarse"] = df["tkind"].str.split(":").str[0]
        df["is_lookup"] = df["turn"] != "decision"
        out = {"n_rows": len(df)}
        for col in ("twin_full", "env", "level", "ps", "turn", "source", "tkind_coarse"):
            out[f"by_{col}"] = dict(sorted(C(df[col]).items()))
        out["by_turn_x_tkind"] = {f"{a}|{b}": int(n) for (a, b), n in df.groupby(["turn", "tkind_coarse"]).size().items()}
        out["by_level_x_ps"] = {f"{a}|{b}": int(n) for (a, b), n in df.groupby(["level", "ps"]).size().items()}
        out["by_source_x_turn"] = {f"{a}|{b}": int(n) for (a, b), n in df.groupby(["source", "turn"]).size().items()}
        out["n_scenarios"] = int(df.scenario_id.nunique())
        out["n_pairs"] = int(df.pair_id.nunique())
        out["rows_per_scenario"] = dist(df.groupby("scenario_id").size().tolist())
        out["rows_per_pair"] = dist(df.groupby("pair_id").size().tolist())
        out["decision_rows_per_scenario"] = dist(df[~df.is_lookup].groupby("scenario_id").size().tolist())
        # cluster = source post (variants share their seed's cluster); weight of each cluster in training
        clus = {}
        import json, os
        from common import ROOT
        sp = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
        for cid, c in sp["clusters"].items():
            for p in c["pairs"]:
                clus[p] = cid
        df["cluster"] = df.pair_id.map(lambda p: clus.get(p, "?"))
        per_cluster = df.groupby("cluster").agg(rows=("i", "size"), pairs=("pair_id", "nunique")).sort_values("rows", ascending=False)
        out["n_clusters"] = int(len(per_cluster))
        out["rows_per_cluster"] = dist(per_cluster.rows.tolist())
        out["top_clusters_by_rows"] = [{"cluster": c, "rows": int(r.rows), "pairs": int(r.pairs),
                                        "share": round(r.rows / len(df), 3)} for c, r in per_cluster.head(8).iterrows()]
        out["lookup_share_rows"] = round(float(df.is_lookup.mean()), 4)
        # ---- target repetition --------------------------------------------------------------------------------------------
        g = df.groupby("target")
        tg = pd.DataFrame({"count": g.size(), "turns": g["turn"].agg(lambda s: "|".join(sorted(set(s)))),
                           "tkind": g["tkind"].first(), "n_scen": g["scenario_id"].nunique(), "n_pairs": g["pair_id"].nunique(),
                           "twins": g["twin"].agg(lambda s: "".join(sorted(set(s)))), "levels": g["level"].agg(lambda s: "".join(sorted(set(s)))),
                           "envs": g["env"].agg(lambda s: "|".join(sorted(set(s)))), "first_row": g["i"].min(),
                           "sources": g["source"].agg(lambda s: "|".join(sorted(set(s))))}).sort_values("count", ascending=False)
        out["n_distinct_targets"] = int(len(tg))
        dec_t = tg[tg.turns == "decision"]
        lk_t = tg[tg.turns != "decision"]
        mixed = tg[tg.turns.str.contains("decision") & tg.turns.str.contains("lookup")]
        out["n_distinct_decision_targets"] = int(len(dec_t))
        out["n_distinct_lookup_targets"] = int(len(lk_t))
        out["n_targets_used_as_both_lookup_and_decision"] = int(len(mixed))
        out["repeats_per_decision_target"] = dist(dec_t["count"].tolist())
        out["repeats_per_lookup_target"] = dist(lk_t["count"].tolist())
        out["decision_targets_shared_by_2plus_scenarios"] = int((dec_t.n_scen > 1).sum())
        out["lookup_targets_shared_by_2plus_scenarios"] = int((lk_t.n_scen > 1).sum())
        out["lookup_targets_shared_by_both_twins"] = int((lk_t.twins == "AB").sum())
        out["lookup_rows_whose_target_is_shared_by_both_twins"] = int(lk_t[lk_t.twins == "AB"]["count"].sum())
        out["top_targets"] = [{"count": int(r["count"]), "target": t[:160], "turns": r.turns, "kind": r.tkind,
                               "n_scenarios": int(r.n_scen), "n_pairs": int(r.n_pairs), "twins": r.twins, "levels": r.levels,
                               "envs": r.envs, "sources": r.sources, "example_row": int(r.first_row)}
                              for t, r in tg.head(15).iterrows()]
        out["top_decision_targets"] = [{"count": int(r["count"]), "target": t[:160], "kind": r.tkind, "n_scenarios": int(r.n_scen),
                                        "levels": r.levels, "example_row": int(r.first_row)} for t, r in dec_t.head(6).iterrows()]
        # who carries the 46x target
        top = tg.index[0]
        sub = df[df.target == top]
        out["max_target_detail"] = {"target": top, "count": int(len(sub)), "turns": dict(C(sub.turn)),
                                    "scenarios": sorted(sub.scenario_id.unique().tolist()),
                                    "pairs": sorted(sub.pair_id.unique().tolist()), "by_level": dict(C(sub.level)),
                                    "by_twin": dict(C(sub.twin)), "env": sorted(sub.env.unique().tolist()),
                                    "example_rows": sub.i.head(3).tolist()}
        out["share_of_rows_whose_target_occurs_8plus_times"] = round(float(df.target.map(tg["count"]).ge(8).mean()), 4)
        if name == "sft_train":
            tg.reset_index().to_csv(f"{OUT}/targets_train.csv", index=False)
        # ---- oracle: grade each decision target with the GRPO reward / evaluation grader --------------------------------------
        grades = []
        for r in (views[name][i] for i in df[~df.is_lookup].i):
            s = scn[r["scenario_id"]]
            sc, comp = grade(target_of(r), s, r["env_id"])
            grades.append({"i": views[name].index(r), "scenario_id": r["scenario_id"], "level": r["level"], "twin": r["twin"][:1],
                           "expected": s["expected"]["decision"], "decision": comp["decision"], "reward": sc})
        gd = pd.DataFrame(grades)
        out["oracle_decision_classes"] = {f"{a}|{b}|{c}": int(n) for (a, b, c), n in
                                          gd.groupby(["twin", "expected", "decision"]).size().items()}
        out["oracle_reward_dist"] = dict(sorted(C(gd.reward.round(3)).items()))
        low = gd[gd.reward < 1.0]
        out["oracle_targets_scoring_below_1"] = {"n_rows": int(len(low)), "n_scenarios": int(low.scenario_id.nunique()),
                                                 "examples": low.head(8).to_dict("records")}
        # ---- hint balance: is the P2 hint (a random 30 % of renderings) correlated with the label? ---------------------------
        dec = df[~df.is_lookup]
        out["P2_share_by_twin_decision_rows"] = {tw: round(float((g.ps == "P2").mean()), 3) for tw, g in dec.groupby("twin")}
        out["P2_share_by_target_kind_decision_rows"] = {k: round(float((g.ps == "P2").mean()), 3) for k, g in dec.groupby("tkind_coarse")}
        # ---- rendering duplicates: per scenario, how many decision rows share the same request + look-up + result + target --
        dup = []
        for sid, g in dec.groupby("scenario_id"):
            convs = {json.dumps(views[name][i]["prompt"][1:], ensure_ascii=False) for i in g.i}
            dup.append({"scenario_id": sid, "decision_rows": int(len(g)), "distinct_conversation_after_system": len(convs),
                        "distinct_targets": int(g.target.nunique())})
        dd = pd.DataFrame(dup)
        out["rendering_duplicates"] = {
            "scenarios": int(len(dd)), "decision_rows": int(dd.decision_rows.sum()),
            "distinct_request_lookup_result_triples": int(dd.distinct_conversation_after_system.sum()),
            "scenarios_with_one_target_for_all_levels": int((dd.distinct_targets == 1).sum()),
            "note": "rows of one scenario differ only in the system prompt (level) except L3's noisy request",
            "examples": dd.sort_values("decision_rows", ascending=False).head(3).to_dict("records")}
        # ---- look-up tools: which tools are look-up targets (val vs train) ------------------------------------------------------
        out["lookup_tools"] = dict(C(df[df.is_lookup].tkind).most_common())
        # ---- prompt shape: number of messages in the prompt vs the turn it leads to ---------------------------------------------
        shape = C((views[name][i]["turn"], len(views[name][i]["prompt"])) for i in df.i)
        out["prompt_shape_turn_counts"] = {f"{t}|{n}msgs": int(c) for (t, n), c in sorted(shape.items())}
        two = [(views[name][i]["turn"]) for i in df.i if len(views[name][i]["prompt"]) == 2]
        out["system_user_prompts"] = {"n": len(two), "lead_to_lookup": sum(t != "decision" for t in two),
                                      "lead_to_decision": sum(t == "decision" for t in two),
                                      "decision_scenarios": sorted({views[name][i]["scenario_id"] for i in df.i
                                                                    if len(views[name][i]["prompt"]) == 2
                                                                    and views[name][i]["turn"] == "decision"})}
        res[name] = out
    tr_tools = set(res["sft_train"]["lookup_tools"])
    res["val_lookup_tools_unseen_in_train_lookups"] = sorted(set(res["sft_val"]["lookup_tools"]) - tr_tools)
    res["val_lookup_rows_with_untrained_tool"] = int(sum(n for k, n in res["sft_val"]["lookup_tools"].items() if k not in tr_tools))
    res["val_lookup_rows"] = int(sum(res["sft_val"]["lookup_tools"].values()))
    # cross-view: SFT decision prompts that are GRPO prompts
    keyset = {(r["scenario_id"], r["level"]) for r in views["sft_train"] if r["turn"] == "decision"}
    gk = [(r["scenario_id"], r["level"]) for r in views["grpo_train"]]
    sft_by_key = {(r["scenario_id"], r["level"]): r for r in views["sft_train"] if r["turn"] == "decision"}
    same_prompt = sum(1 for r in views["grpo_train"] if (r["scenario_id"], r["level"]) in sft_by_key
                      and sft_by_key[(r["scenario_id"], r["level"])]["prompt"] == r["prompt"])
    res["grpo_prompts_in_sft_train_decision_rows"] = {"grpo_rows": len(gk), "key_in_sft": sum(k in keyset for k in gk),
                                                      "identical_prompt": same_prompt}
    dump(res, "composition.json")
    t = res["sft_train"]
    print({k: t[k] for k in ("n_rows", "n_distinct_targets", "n_distinct_decision_targets", "n_distinct_lookup_targets",
                             "lookup_share_rows", "repeats_per_decision_target", "repeats_per_lookup_target")})
    print(t["max_target_detail"])
    print(t["oracle_reward_dist"], t["oracle_targets_scoring_below_1"]["n_rows"])


if __name__ == "__main__":
    main()
