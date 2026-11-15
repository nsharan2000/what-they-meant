"""VERIFY RP-8: realistic member blocks repeat records (scenario_records merged on top of a member file that already has them).

Counts are taken from the rendered TRAINING prompts themselves (what was actually trained), not re-rendered:
member block = system prompt text between '## Member' and '## Tools'; a duplicated line = an identical '- ...' bullet line
occurring 2+ times. Attribution: for each duplicated history line, check whether the scenario spec's scenario_records hold a
transaction with that date/amount whose id is ALSO in the member file (render-merge duplicate) -- versus a duplicate that the
scenario itself plants (e.g. a double-charge clash).
Also: twin-B rows whose duplicated line is an OUTGOING payment (reads like 'already paid twice' = a duplicate-payment cue).
M1 success on duplicated vs clean GRPO prompts (own bootstrap).
Outputs: results/v5_dups.json, results/v5_dup_rows.csv
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from vcommon import CUR, OUT, ROOT, boot, dump, load_jsonl

HIST = re.compile(r"^- (\d{4}-\d{2}-\d{2}) \| ([+-])")


def member_block(sys_txt):
    a, b = sys_txt.find("## Member"), sys_txt.find("## Tools")
    return sys_txt[a:b] if 0 <= a < b else ""


def all_specs():
    specs = {}
    for p in (glob.glob(os.path.join(ROOT, "dataset", "scenarios", "*.jsonl")) +
              glob.glob(os.path.join(ROOT, "dataset", "scenarios_synth", "*.jsonl")) +
              glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        for s in load_jsonl(p):
            specs.setdefault(s["id"], s)
    for r in load_jsonl(os.path.join(CUR, "grpo_train.jsonl")):     # the as-trained spec wins
        specs[r["scenario_id"]] = json.loads(r["scenario"])
    return specs


def main():
    specs = all_specs()
    members = {}

    def member(env, mid):
        k = (env, mid)
        if k not in members:
            members[k] = json.load(open(os.path.join(ROOT, "envs", env, "members", f"{mid}.json")))
        return members[k]

    # spec level: scenario_records already present in the member file
    spec_dup = {}
    for sid, s in specs.items():
        recs = s.get("scenario_records") or []
        if not recs or "member_id" not in s:
            spec_dup[sid] = []
            continue
        m = member(s["env_id"], s["member_id"])
        d = []
        for x in recs:
            ex = m.get(x["field"], [])
            rec = x["record"]
            if (isinstance(rec, dict) and rec.get("id") and any(isinstance(e, dict) and e.get("id") == rec["id"] for e in ex)) \
                    or rec in ex:
                d.append(x["field"])
        spec_dup[sid] = d
    out = {"specs_checked": len(spec_dup), "specs_with_dup_scenario_records": sum(1 for v in spec_dup.values() if v),
           "dup_fields": dict(Counter(f for v in spec_dup.values() for f in set(v)))}
    rows = []
    for fn in ("sft_train", "sft_val", "grpo_train", "dpo_train"):
        for i, r in enumerate(load_jsonl(os.path.join(CUR, fn + ".jsonl"))):
            mb = member_block(r["prompt"][0]["content"])
            lines = [l for l in mb.splitlines() if l.startswith("- ")]
            c = Counter(lines)
            dups = [l for l, k in c.items() if k > 1]
            hist_dups = [l for l in dups if HIST.match(l)]
            out_dups = [l for l in hist_dups if HIST.match(l).group(2) == "-"]
            rows.append({"file": fn, "line": i, "sid": r["scenario_id"], "twin": r["twin"][0], "level": r["level"],
                         "ps": r["prompt_setting"], "turn": r.get("turn", "decision"), "n_dup_lines": len(dups),
                         "n_dup_hist": len(hist_dups), "n_dup_outgoing": len(out_dups),
                         "spec_merge_dup": bool(spec_dup.get(r["scenario_id"])),
                         "example": (dups[0][:140] if dups else "")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v5_dup_rows.csv"), index=False)
    real = df[df.level.isin(["L1", "L2", "L3"])]
    out["realistic_rows"] = {fn: {"rows": int(len(g)), "with_dup_line": int((g.n_dup_lines > 0).sum()),
                                  "with_dup_history_line": int((g.n_dup_hist > 0).sum()),
                                  "dup_rows_explained_by_spec_merge": int(((g.n_dup_lines > 0) & g.spec_merge_dup).sum()),
                                  "B_rows_with_dup_outgoing_payment": int(((g.n_dup_outgoing > 0) & (g.twin == "B")).sum()),
                                  "A_rows_with_dup_outgoing_payment": int(((g.n_dup_outgoing > 0) & (g.twin == "A")).sum())}
                             for fn, g in real.groupby("file")}
    out["L0r_rows_with_dup_line"] = int((df[df.level == "L0r"].n_dup_lines > 0).sum())
    out["by_level_sft_train"] = {lv: {"rows": int(len(g)), "dup": int((g.n_dup_lines > 0).sum())}
                                 for lv, g in df[df.file == "sft_train"].groupby("level")}
    # analyst examples
    out["sft_train_line2"] = df[(df.file == "sft_train") & (df.line.isin([2, 4]))][["line", "sid", "level", "n_dup_lines", "example"]]\
        .to_dict(orient="records")
    # B rows with a duplicated outgoing payment line: examples
    bdo = real[(real.n_dup_outgoing > 0) & (real.twin == "B") & (real.file == "sft_train")]
    out["examples_B_dup_outgoing"] = bdo.head(5)[["file", "line", "sid", "level", "ps", "turn", "example"]].to_dict(orient="records")
    # M1 success on dup vs clean prompts
    pp = pd.read_csv(os.path.join(OUT, "v3_m1_prompts.csv"))
    g = df[df.file == "grpo_train"][["line", "n_dup_lines"]].rename(columns={"line": "gi"})
    pp = pp.merge(g, on="gi", how="left")
    res = {}
    for tw in ("A", "B"):
        x = pp[(pp.twin == tw) & pp.level.isin(["L1", "L2", "L3"])]
        a, b = x[x.n_dup_lines == 0].succ.to_numpy(), x[x.n_dup_lines > 0].succ.to_numpy()
        rng = np.random.default_rng(9)
        diffs = b[rng.integers(0, len(b), (10000, len(b)))].mean(1) - a[rng.integers(0, len(a), (10000, len(a)))].mean(1)
        res[tw] = {"n_clean": int(len(a)), "n_dup": int(len(b)), "succ_clean": round(float(a.mean()), 4),
                   "succ_dup": round(float(b.mean()), 4), "diff": round(float(b.mean() - a.mean()), 4),
                   "ci": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}
    out["M1_success_dup_vs_clean_L1to3"] = res
    dump(out, "v5_dups.json")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
