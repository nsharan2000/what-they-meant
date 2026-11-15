"""RP-2 sensitivity: how much of the toy -> realistic shift on BOTH twins survives when episodes that hit an empty / unscripted
look-up (at either level) are dropped?  (A sensitivity analysis only: dropping conditions on the model's own look-up choice.)
Input: results/v1_episode_flags.csv (written by v1_unscripted.py). Arms as in the analyst's RP-2 (grid_open_seedtools_P0 out).
Outputs: results/v15_rp2_sensitivity.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from vcommon import OUT, boot, dump


def main():
    df = pd.read_csv(os.path.join(OUT, "v1_episode_flags.csv"))
    df = df[df.src != "grid_open_seedtools_P0"]
    out = {}
    for frm, to in ((("L0r", "P0"), ("L1", "P0")), (("L0r", "P0"), ("L2", "P0")), (("L0r", "P1"), ("L1", "P1"))):
        key = f"{frm[0]}/{frm[1]}->{to[0]}/{to[1]}"
        res = {}
        for tw, col in (("A", "literal"), ("A", "ask"), ("B", "needless"), ("B", "act")):
            stats = {"all": [], "no_empty": []}
            for (fam, model), g in df[df.twin == tw].groupby(["fam", "model"]):
                a = g[(g.level == frm[0]) & (g.prompt == frm[1])].set_index("id")
                b = g[(g.level == to[0]) & (g.prompt == to[1])].set_index("id")
                ids = sorted(set(a.index) & set(b.index))
                if len(ids) < 10:
                    continue
                for sub in ("all", "no_empty"):
                    use = ids if sub == "all" else [i for i in ids if a.loc[i, "any_empty"] == 0 and b.loc[i, "any_empty"] == 0]
                    if len(use) < 10:
                        continue
                    d = b.loc[use, col].to_numpy(float) - a.loc[use, col].to_numpy(float)
                    mu, lo, hi = boot(d, None, seed=151)
                    stats[sub].append({"arm": f"{fam}:{model}", "n": len(use), "diff": mu, "sig_down": hi < 0, "sig_up": lo > 0})
            res[f"{tw}_{col}"] = {sub: {"n_arms": len(v), "sig_down": int(sum(x["sig_down"] for x in v)),
                                        "sig_up": int(sum(x["sig_up"] for x in v)),
                                        "median_diff": round(float(np.median([x["diff"] for x in v])), 4) if v else None}
                                  for sub, v in stats.items()}
        out[key] = res
    dump(out, "v15_rp2_sensitivity.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
