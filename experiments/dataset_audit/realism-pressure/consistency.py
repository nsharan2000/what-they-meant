"""World-state consistency of the realistic prompts: do the transactions that the scripted look-up results show also appear in
the member history rendered into the L1/L2/L3 system prompt (and is any history line duplicated)?

Found in a transcript (panel_hard_claude.jsonl line 113, H051B L2/P2, Claude Opus): the decisive $300 payment exists only in the
scenario's scripted get_transactions result; the member records (rendered into the L1/L2 prompt and read by unscripted view tools)
lack it and list the $650 Trek payment twice (member file + scenario_records). A model that reads the realistic context sees two
contradicting worlds.

For every training row (sft_train / sft_val / grpo_train) at L1/L2/L3: parse transaction-like records (date + amount +
counterparty) from the <tool_result> blocks in the prompt; a record is 'missing' when its rendered history line head
("<date> | <sign><amount> | <counterparty>[ (<handle>)]", wtm/render.py _txn_line) is absent from the system prompt although its
date is not older than the oldest history line shown (so the cap cannot explain it). Duplicated history lines are counted too.
Joins M1 per-prompt success (results/m1_samples.csv) to test whether inconsistent prompts are harder for the trained policy.
Outputs: results/consistency_rows.csv, results/consistency.json
Run: python3 -B experiments/dataset_audit/realism-pressure/consistency.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

import pandas as pd

from common import RES, ROOT, boot_diff_unpaired, dump

sys.path.insert(0, ROOT)
from wtm.render import _fmt_money  # noqa: E402

TR_RE = re.compile(r"<tool_result name=\"([^\"]+)\">\s*(.*?)\s*</tool_result>", re.S)
LINE_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) \| [^\n]*$", re.M)


def records(obj):
    if isinstance(obj, dict):
        if "date" in obj and "amount" in obj and ("counterparty" in obj or "merchant" in obj):
            yield obj
        for v in obj.values():
            yield from records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from records(v)


def head(t: dict) -> str:
    sign = "-" if t.get("direction") == "out" else "+"
    who = t.get("counterparty", t.get("merchant", ""))
    if t.get("handle"):
        who += f" ({t['handle']})"
    return f"{t.get('date', '')} | {sign}{_fmt_money(t.get('amount'), t.get('currency', 'USD'))} | {who}"


def check_row(r: dict) -> dict:
    system = r["prompt"][0]["content"]
    hist_dates = LINE_RE.findall(system)
    oldest = min(hist_dates) if hist_dates else None
    lines = [l for l in system.splitlines() if LINE_RE.match(l)]
    dup = sum(c - 1 for c in Counter(lines).values() if c > 1)
    n_rec = n_missing = 0
    missing = []
    for m in r["prompt"][1:]:
        if m["role"] != "user":
            continue
        for name, body in TR_RE.findall(m["content"]):
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            for t in records(obj):
                if not isinstance(t.get("date"), str) or not re.match(r"\d{4}-\d{2}-\d{2}$", t["date"]):
                    continue
                n_rec += 1
                if oldest is None or t["date"] < oldest:
                    continue
                if head(t) not in system:
                    n_missing += 1
                    missing.append(f"{name}: {head(t)}")
    return {"hist_lines": len(lines), "dup_lines": dup, "tool_txn_records": n_rec, "missing_in_history": n_missing,
            "missing_examples": "; ".join(missing[:2])}


def main():
    recs = []
    for fn in ("sft_train.jsonl", "sft_val.jsonl", "grpo_train.jsonl"):
        with open(os.path.join(ROOT, "dataset", "curated_v121", fn)) as fh:
            for i, l in enumerate(fh):
                r = json.loads(l)
                if r["level"] == "L0r":
                    continue
                rec = {"file": fn, "line": i, "scenario_id": r["scenario_id"], "twin": r["twin"][0], "level": r["level"],
                       "prompt_setting": r["prompt_setting"], "turn": r.get("turn", "decision")}
                rec.update(check_row(r))
                rec["inconsistent"] = int(rec["missing_in_history"] > 0)
                rec["dup_flag"] = int(rec["dup_lines"] > 0)
                recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RES, "consistency_rows.csv"), index=False)
    summ = {}
    for (fn, lv, tw), g in df.groupby(["file", "level", "twin"]):
        summ[f"{fn}|{lv}|{tw}"] = {"rows": int(len(g)), "rows_with_tool_txn": int((g.tool_txn_records > 0).sum()),
                                    "rows_missing_txn_in_history": int(g.inconsistent.sum()),
                                    "rows_with_duplicated_history_line": int(g.dup_flag.sum())}
    # M1: success on inconsistent vs consistent grpo prompts (realistic levels)
    m1 = pd.read_csv(os.path.join(RES, "m1_samples.csv"))
    pp = m1.groupby(["grpo_line", "twin"]).success.mean().reset_index()
    gr = df[df.file == "grpo_train.jsonl"].set_index("line")
    pp = pp[pp.grpo_line.isin(gr.index)].copy()
    pp["inconsistent"] = pp.grpo_line.map(gr.inconsistent)
    pp["dup_flag"] = pp.grpo_line.map(gr.dup_flag)
    m1eff = []
    for col in ("inconsistent", "dup_flag"):
        for tw in ("A", "B"):
            a = pp[(pp.twin == tw) & (pp[col] == 0)].success.to_numpy()
            b = pp[(pp.twin == tw) & (pp[col] == 1)].success.to_numpy()
            if len(a) >= 3 and len(b) >= 3:
                d, lo, hi = boot_diff_unpaired(a, b, seed=51)
                m1eff.append({"flag": col, "twin": tw, "n_clean": int(len(a)), "n_flagged": int(len(b)),
                              "success_clean": round(float(a.mean()), 4), "success_flagged": round(float(b.mean()), 4),
                              "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]})
            else:
                m1eff.append({"flag": col, "twin": tw, "n_clean": int(len(a)), "n_flagged": int(len(b)),
                              "note": "too few flagged prompts among the 120 sampled"})
    ex = df[(df.inconsistent == 1)].head(4)[["file", "line", "scenario_id", "level", "prompt_setting", "turn",
                                             "missing_examples"]].to_dict(orient="records")
    exd = df[(df.dup_flag == 1)].head(3)[["file", "line", "scenario_id", "level", "dup_lines"]].to_dict(orient="records")
    tot = {"rows_checked": int(len(df)), "rows_with_tool_txn": int((df.tool_txn_records > 0).sum()),
           "rows_missing_txn_in_history": int(df.inconsistent.sum()),
           "scenarios_missing": sorted(set(df[df.inconsistent == 1].scenario_id)),
           "rows_with_duplicated_history_line": int(df.dup_flag.sum()),
           "scenarios_duplicated": sorted(set(df[df.dup_flag == 1].scenario_id))}
    dump({"totals": tot, "by_file_level_twin": summ, "m1_success_by_flag": m1eff, "examples_missing": ex,
          "examples_duplicated": exd}, "consistency.json")
    print(json.dumps(tot, indent=1)[:3000])
    print(json.dumps(m1eff, indent=1))
    print(json.dumps(ex, indent=1)[:2000])


if __name__ == "__main__":
    main()
