"""Is part of the measured "realism pressure" a simulator artefact? At L1/L2 every environment tool is offered (at L0r only the
scenario's l0_tools), so models look up more things; a look-up that no scenario tool rule covers falls through to the generic
handler (wtm/sim.py) and returns {"result": "No matching records found."} (or an empty view / 'Unknown tool' error).
A model that then asks the member "I can't find X, which store?" is graded ask_generic = a needless ask on twin B.

Per episode: empty_before = at least one pre-decision tool result that is empty/unscripted. Per cell: share of episodes with
empty_before, needless/ask rates split by empty_before, and the L0r -> L1/L2 change in B_needless and A_ask decomposed into
the part carried by episodes with empty_before vs without (paired by scenario, scenario bootstrap).
Also checks the training views (sft/grpo/dpo/sft_val prompts) for empty tool results.
Outputs: results/unscripted_cells.csv, results/unscripted.json
Run: python3 -B experiments/dataset_audit/realism-pressure/unscripted.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import RES, ROOT, boot_mean_ci, dump, iter_episode_files, twin_letter, short_model

FAMILY = {"grid_open_P0": "seed_v11_open", "grid_open_P1": "seed_v11_open", "grid_open_controls": "seed_v11_open",
          "grid_local_tinyaya": "seed_v11_local", "grid_spark": "seed_v11_spark", "grid_claude": "seed_v11_claude",
          "grid_open_variants": "variants_v1", "eval_base_v12": "eval_v12", "panel_hard_claude": "panel_hard_v12",
          "panel_hard_open": "panel_hard_v12", "panel_mined_claude": "panel_mined_v12",
          "panel_mined_open": "panel_mined_v12", "panel_mined_fable": "panel_mined_v12"}
ASK = {"ask_informed", "ask_generic"}
NEEDLESS = {"ask_informed", "ask_generic", "tell_informed"}


def is_empty(res) -> bool:
    if isinstance(res, dict):
        if res.get("result") == "No matching records found.":
            return True
        if "error" in res and str(res["error"]).startswith("Unknown tool"):
            return True
        if res.get("total") == 0 and not res.get("items"):
            return True
    if isinstance(res, str) and "No matching records found" in res:
        return True
    return False


def load():
    recs = []
    for src, path in iter_episode_files():
        fam = FAMILY.get(src)
        if fam is None:
            continue
        with open(path) as fh:
            for i, l in enumerate(fh):
                r = json.loads(l)
                tw = twin_letter(r["twin"])
                dec = r["decision_grade"]["decision"]
                if tw is None or dec == "error" or r.get("rep", 0) != 0:
                    continue
                evs = r.get("events", [])
                pre = evs[:-1] if evs else []          # decision mode: the last event is the decision (or the failure)
                n_calls = sum(1 for e in pre if e.get("kind") == "tool_call")
                n_empty = sum(1 for e in pre if e.get("kind") == "tool_call" and is_empty(e.get("result")))
                recs.append({"family": fam, "model": short_model(r["model"]), "level": r["level"], "prompt": r["prompt"],
                             "twin": tw, "id": r["id"], "decision": dec, "n_lookups": n_calls, "n_empty": n_empty,
                             "empty_before": int(n_empty > 0), "ask": int(dec in ASK), "needless": int(dec in NEEDLESS),
                             "ref": f"{os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}:{i}"})
    return pd.DataFrame(recs)


def training_views():
    out = {}
    for fn in ("sft_train.jsonl", "sft_val.jsonl", "grpo_train.jsonl", "dpo_train.jsonl"):
        n = n_empty = 0
        with open(os.path.join(ROOT, "dataset", "curated_v121", fn)) as fh:
            for l in fh:
                r = json.loads(l)
                n += 1
                trs = [m["content"] for m in r["prompt"] if m["role"] == "user" and "<tool_result" in m["content"]]
                n_empty += int(any("No matching records found" in t or '"total": 0' in t or "Unknown tool" in t for t in trs))
        out[fn] = {"rows": n, "rows_with_empty_tool_result": n_empty}
    return out


def main():
    df = load()
    cells = []
    for (fam, model, lv, pr, tw), g in df.groupby(["family", "model", "level", "prompt", "twin"]):
        e, ne = g[g.empty_before == 1], g[g.empty_before == 0]
        rec = {"family": fam, "model": model, "level": lv, "prompt": pr, "twin": tw, "n": int(len(g)),
               "share_empty_before": round(g.empty_before.mean(), 4), "mean_lookups": round(g.n_lookups.mean(), 3),
               "metric": "needless" if tw == "B" else "ask"}
        col = rec["metric"]
        rec["rate"] = round(g[col].mean(), 4)
        rec["rate_if_empty"] = round(e[col].mean(), 4) if len(e) else None
        rec["rate_if_not_empty"] = round(ne[col].mean(), 4) if len(ne) else None
        rec["share_of_metric_with_empty_before"] = round(float((g[col] * g.empty_before).sum() / max(1, g[col].sum())), 4)
        cells.append(rec)
    cells = pd.DataFrame(cells)
    cells.to_csv(os.path.join(RES, "unscripted_cells.csv"), index=False)
    # decomposition of the L0r -> L1/L2 change (paired by scenario)
    dec = []
    for (fam, model, pr, tw), g in df.groupby(["family", "model", "prompt", "twin"]):
        col = "needless" if tw == "B" else "ask"
        base = g[g.level == "L0r"].set_index("id")
        for lv in ("L1", "L2"):
            oth = g[g.level == lv].set_index("id")
            ids = sorted(set(base.index) & set(oth.index))
            if len(ids) < 10:
                continue
            tot = oth.loc[ids, col].to_numpy(float) - base.loc[ids, col].to_numpy(float)
            via_e = (oth.loc[ids, col] * oth.loc[ids, "empty_before"]).to_numpy(float) - \
                (base.loc[ids, col] * base.loc[ids, "empty_before"]).to_numpy(float)
            mu, lo, hi = boot_mean_ci(tot, None, seed=41)
            mue, loe, hie = boot_mean_ci(via_e, None, seed=42)
            dec.append({"family": fam, "model": model, "prompt": pr, "twin": tw, "metric": col, "from": "L0r", "to": lv,
                        "n": len(ids), "share_empty_from": round(base.loc[ids, "empty_before"].mean(), 4),
                        "share_empty_to": round(oth.loc[ids, "empty_before"].mean(), 4),
                        "diff_total": round(mu, 4), "ci_total": [round(lo, 4), round(hi, 4)],
                        "diff_via_empty": round(mue, 4), "ci_via_empty": [round(loe, 4), round(hie, 4)],
                        "diff_without_empty": round(mu - mue, 4)})
    dec = pd.DataFrame(dec)
    dec.to_csv(os.path.join(RES, "unscripted_decomposition.csv"), index=False)
    # pooled share of empty-before by level (all models, all families)
    pooled = df.groupby(["level", "twin"]).agg(n=("id", "size"), share_empty=("empty_before", "mean"),
                                                mean_lookups=("n_lookups", "mean")).reset_index()
    b = df[df.twin == "B"]
    cond = b.groupby(["level", "empty_before"]).needless.agg(["mean", "size"]).reset_index()
    examples = df[(df.twin == "B") & (df.level.isin(["L1", "L2"])) & (df.needless == 1) & (df.empty_before == 1)]
    # share of all B needless asks (and of all A asks) that came right after an empty / unscripted look-up, by level
    share_after_empty = []
    for (lv, tw), g in df.groupby(["level", "twin"]):
        col = "needless" if tw == "B" else "ask"
        k = int(g[col].sum())
        share_after_empty.append({"level": lv, "twin": tw, "metric": col, "n_events": k,
                                  "n_after_empty": int((g[col] * g.empty_before).sum()),
                                  "share_after_empty": round(float((g[col] * g.empty_before).sum() / max(1, k)), 4)})
    by_model = []
    for (model, lv), g in df[(df.twin == "B") & df.level.isin(["L0r", "L1", "L2"])].groupby(["model", "level"]):
        k = int(g.needless.sum())
        by_model.append({"model": model, "level": lv, "n_B": int(len(g)), "share_empty_before": round(g.empty_before.mean(), 4),
                         "B_needless": round(g.needless.mean(), 4), "n_needless": k,
                         "share_of_needless_after_empty": round(float((g.needless * g.empty_before).sum() / max(1, k)), 4)})
    dump({"training_views": training_views(),
          "pooled_share_empty_by_level": pooled.round(4).to_dict(orient="records"),
          "B_needless_by_level_and_empty": cond.round(4).to_dict(orient="records"),
          "decomposition": dec.to_dict(orient="records"),
          "share_of_asks_after_empty_by_level": share_after_empty,
          "B_needless_after_empty_by_model_level": by_model,
          "examples_B_needless_after_empty_L1L2": examples.head(6)[["family", "model", "level", "prompt", "id", "decision",
                                                                     "n_empty", "ref"]].to_dict(orient="records")},
         "unscripted.json")
    print(pooled.round(3).to_string(index=False))
    print(cond.round(3).to_string(index=False))
    show = dec[dec.model.str.contains("tiny-aya-global|claude|DeepSeek|Qwen3-235|aya-expanse")]
    print(show[["family", "model", "prompt", "twin", "metric", "to", "n", "share_empty_from", "share_empty_to",
                "diff_total", "diff_via_empty", "diff_without_empty"]].to_string(index=False))


if __name__ == "__main__":
    main()
