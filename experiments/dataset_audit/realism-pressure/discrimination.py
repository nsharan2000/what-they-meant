"""Twin discrimination by level: does the realistic prompt reduce a model's ability to tell the should-ask twin from the
should-act twin? (the objective is a CONDITIONAL behaviour, so a level that shifts both twins the same way is a threshold shift,
while a level that shrinks the A-B gap is a discrimination loss.)

Per pair (A and B twin of one pair, same model, level and prompt): D_ask = ask(A) - ask(B) (any chat question),
D_act = act(B) - act(A) (approve or commit), PP = both twins strictly correct (paired pass). D = 1 is perfect discrimination,
D = 0 means the model does the same thing on both twins. Means over pairs; scenario bootstrap over pairs (10,000) and paired level contrasts (L0r -> L1 / L2,
L2s -> L2, P0 -> P2/P1) of D_ask.
M1: per-prompt rates (8 samples) for the pairs whose A and B decision prompts were both sampled at the same level and setting.
Outputs: results/discrimination.csv, results/discrimination_contrasts.csv, results/discrimination.json
Run: python3 -B experiments/dataset_audit/realism-pressure/discrimination.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import RES, boot_mean_ci, dump

LEVEL_CONTRASTS = [(("L0r", "P0"), ("L1", "P0")), (("L0r", "P0"), ("L2", "P0")), (("L1", "P0"), ("L2", "P0")),
                   (("L2s", "P0"), ("L2", "P0")), (("L0r", "P1"), ("L1", "P1")), (("L0r", "P1"), ("L2", "P1")),
                   (("L2s", "P1"), ("L2", "P1")), (("L0r", "P2"), ("L1", "P2")), (("L0r", "P2"), ("L2", "P2")),
                   (("L2s", "P2"), ("L2", "P2")), (("L0r", "P0"), ("L0r", "P2")), (("L2", "P0"), ("L2", "P2")),
                   (("L0r", "P0"), ("L0r", "P1")), (("L2", "P0"), ("L2", "P1"))]


def pair_table(ep: pd.DataFrame) -> pd.DataFrame:
    ep = ep.copy()
    ep["pair"] = ep.id.str[:-1]
    a = ep[ep.twin == "A"].set_index(["family", "model", "level", "prompt", "pair"])
    b = ep[ep.twin == "B"].set_index(["family", "model", "level", "prompt", "pair"])
    idx = a.index.intersection(b.index)
    t = pd.DataFrame(index=idx)
    t["cluster"] = a.loc[idx, "cluster"].to_numpy()
    ask_b = b.loc[idx, "decision"].isin(["ask_informed", "ask_generic"]).to_numpy().astype(float)
    t["D_ask"] = a.loc[idx, "A_ask"].to_numpy() - ask_b
    t["D_act"] = b.loc[idx, "B_act"].to_numpy() - a.loc[idx, "decision"].str.startswith(("approve", "commit")).to_numpy()
    t["PP"] = a.loc[idx, "A_success"].to_numpy() * b.loc[idx, "B_success"].to_numpy()
    return t.reset_index()


def main():
    ep = pd.read_csv(os.path.join(RES, "episode_indicators.csv"))
    pt = pair_table(ep)
    cells = []
    for (fam, model, lv, pr), g in pt.groupby(["family", "model", "level", "prompt"]):
        rec = {"family": fam, "model": model, "level": lv, "prompt": pr, "n_pairs": int(len(g))}
        for m in ("D_ask", "D_act", "PP"):
            mu, lo, hi = boot_mean_ci(g[m].to_numpy(float), None, seed=31)
            rec[m], rec[m + "_ci"] = round(mu, 4), [round(lo, 4), round(hi, 4)]
        cells.append(rec)
    cells = pd.DataFrame(cells)
    cells.to_csv(os.path.join(RES, "discrimination.csv"), index=False)
    con = []
    for (fam, model), g in pt.groupby(["family", "model"]):
        cc = {(lv, pr): x.set_index("pair") for (lv, pr), x in g.groupby(["level", "prompt"])}
        for a, b in LEVEL_CONTRASTS:
            if a not in cc or b not in cc:
                continue
            ids = sorted(set(cc[a].index) & set(cc[b].index))
            if len(ids) < 10:
                continue
            for m in ("D_ask", "PP"):
                d = cc[b].loc[ids, m].to_numpy(float) - cc[a].loc[ids, m].to_numpy(float)
                mu, lo, hi = boot_mean_ci(d, None, seed=32)
                con.append({"family": fam, "model": model, "from": f"{a[0]}/{a[1]}", "to": f"{b[0]}/{b[1]}", "metric": m,
                            "n_pairs": len(ids), "from_val": round(float(cc[a].loc[ids, m].mean()), 4),
                            "to_val": round(float(cc[b].loc[ids, m].mean()), 4), "diff": round(mu, 4),
                            "ci": [round(lo, 4), round(hi, 4)], "sig": bool(lo > 0 or hi < 0)})
    con = pd.DataFrame(con)
    con.to_csv(os.path.join(RES, "discrimination_contrasts.csv"), index=False)

    # M1: pairs with both twins sampled at the same level and setting
    m1 = pd.read_csv(os.path.join(RES, "m1_samples.csv"))
    m1["pair"] = m1.scenario_id.str[:-1]
    pp = m1.groupby(["pair", "twin", "level", "prompt_setting"]).agg(
        ask=("ask", "mean"), act=("act", "mean"), approve=("approve", "mean"), commit=("commit", "mean"),
        success=("success", "mean")).reset_index()
    pp["act"] = pp.approve + pp.commit
    a = pp[pp.twin == "A"].set_index(["pair", "level", "prompt_setting"])
    b = pp[pp.twin == "B"].set_index(["pair", "level", "prompt_setting"])
    idx = a.index.intersection(b.index)
    m1pairs = pd.DataFrame({"D_ask": a.loc[idx, "ask"].to_numpy() - b.loc[idx, "ask"].to_numpy(),
                            "D_act": b.loc[idx, "act"].to_numpy() - a.loc[idx, "act"].to_numpy(),
                            "P_both": a.loc[idx, "success"].to_numpy() * b.loc[idx, "success"].to_numpy()},
                           index=idx).reset_index()
    m1_out = []
    for lv, g in list(m1pairs.groupby("level")) + [("all", m1pairs)]:
        rec = {"level": lv, "n_pairs": int(len(g))}
        for m in ("D_ask", "D_act", "P_both"):
            mu, lo, hi = boot_mean_ci(g[m].to_numpy(float), None, seed=33)
            rec[m], rec[m + "_ci"] = round(mu, 4), [round(lo, 4), round(hi, 4)]
        m1_out.append(rec)
    # all-sample twin gap (unpaired): P(ask | A prompt) - P(ask | B prompt), by level
    m1_gap = []
    per_prompt = m1.groupby(["grpo_line", "twin", "level"]).agg(ask=("ask", "mean"), approve=("approve", "mean")).reset_index()
    for lv, g in list(per_prompt.groupby("level")) + [("all", per_prompt)]:
        pa, pb = g[g.twin == "A"].ask.mean(), g[g.twin == "B"].ask.mean()
        m1_gap.append({"level": lv, "P_ask_given_A": round(float(pa), 4), "P_ask_given_B": round(float(pb), 4),
                       "gap": round(float(pa - pb), 4), "n_A_prompts": int((g.twin == "A").sum()),
                       "n_B_prompts": int((g.twin == "B").sum())})
    dump({"m1_paired": m1_out, "m1_unpaired_gap": m1_gap,
          "base_and_others_contrasts_sig": con[con.sig].to_dict(orient="records")}, "discrimination.json")
    show = cells[cells.model.str.contains("tiny-aya-global|claude|DeepSeek|Qwen3-235|aya-expanse")]
    print(show[["family", "model", "level", "prompt", "n_pairs", "D_ask", "D_act", "PP"]].to_string(index=False))
    print("M1 paired:", m1_out)
    print("M1 unpaired gap:", m1_gap)


if __name__ == "__main__":
    main()
