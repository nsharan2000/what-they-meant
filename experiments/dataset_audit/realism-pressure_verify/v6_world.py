"""Missed-issue check (extends RP-1): is the realistic member block consistent with the scripted tool results?

At L1/L2/L3 the system prompt shows the member's transactions (L1: last 15; L2: as many as fit) rendered from the member file
+ scenario_records. Decisive records that exist ONLY inside a scenario's tool_rules never reach that history, so the realistic
prompt can contradict the look-up result (e.g. H051B: the member asks to return a $300 payment that the history does not show).
L0r is immune (its member card is the seed's own text).
(a) spec level: transaction-like records in tool_rules results (dict with a date and an amount* key) that are missing from the
    merged member transactions (by id, else by date+|amount|) although their date is inside the L1 window (>= the 15th most
    recent merged transaction).
(b) training-row level (exact, on the as-trained prompts): a tool result in an L1/L2/L3 prompt shows a dated transaction whose
    date lies inside the history window printed in the member block, but no history line has that date and amount.
Outputs: results/v6_world.json, results/v6_world_specs.csv, results/v6_world_rows.csv
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

import pandas as pd

from vcommon import CUR, OUT, ROOT, dump, load_jsonl

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def txn_like(obj, out):
    if isinstance(obj, dict):
        d = obj.get("date") or obj.get("created") or obj.get("paid_on")
        amt_keys = [k for k in obj if k.startswith("amount") and isinstance(obj[k], (int, float))]
        if isinstance(d, str) and DATE.match(d) and amt_keys:
            out.append({"id": obj.get("id"), "date": d[:10], "amount": abs(float(obj[amt_keys[0]])),
                        "who": obj.get("counterparty") or obj.get("handle") or obj.get("merchant") or obj.get("vendor") or ""})
        for v in obj.values():
            txn_like(v, out)
    elif isinstance(obj, list):
        for v in obj:
            txn_like(v, out)


def merged_member(spec, cache):
    k = (spec["env_id"], spec["member_id"])
    if k not in cache:
        cache[k] = json.load(open(os.path.join(ROOT, "envs", spec["env_id"], "members", f"{spec['member_id']}.json")))
    m = json.loads(json.dumps(cache[k]))
    for x in (spec.get("scenario_records") or []) + (spec.get("l2_extra_records") or []):
        m.setdefault(x["field"], []).append(x["record"])
    return m


def main():
    specs = {}
    groups = {}
    for r in load_jsonl(os.path.join(CUR, "grpo_train.jsonl")):
        specs[r["scenario_id"]] = json.loads(r["scenario"])
        groups[r["scenario_id"]] = "train"
    for p in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        g = os.path.basename(p).rsplit("_", 1)[0]
        for s in load_jsonl(p):
            if s["id"] not in specs:
                specs[s["id"]] = s
                groups[s["id"]] = g
    cache = {}
    rows = []
    for sid, s in specs.items():
        if "member_id" not in s:
            continue
        m = merged_member(s, cache)
        tx = sorted(m.get("transactions", []), key=lambda t: t.get("date", ""), reverse=True)
        l1_start = tx[14]["date"][:10] if len(tx) >= 15 else (tx[-1]["date"][:10] if tx else "9999")
        oldest = tx[-1]["date"][:10] if tx else "9999"
        ids = {t.get("id") for t in tx if t.get("id")}
        keys = {(t.get("date", "")[:10], round(abs(float(t.get("amount", 0) or 0)), 2)) for t in tx}
        recs = []
        for rule in s.get("tool_rules", []):
            res = rule.get("result_raw", rule.get("result"))
            out = []
            txn_like(res, out)
            for o in out:
                o["tool"] = rule["tool"]
            recs.extend(out)
        missing_l1 = [o for o in recs if o["date"] >= l1_start and not ((o["id"] and o["id"] in ids) or
                                                                       (o["date"], round(o["amount"], 2)) in keys)]
        missing_span = [o for o in recs if o["date"] >= oldest and not ((o["id"] and o["id"] in ids) or
                                                                      (o["date"], round(o["amount"], 2)) in keys)]
        rows.append({"sid": sid, "group": groups[sid], "twin": sid[-1], "n_rule_txns": len(recs),
                     "missing_in_L1_window": len(missing_l1), "missing_in_history_span": len(missing_span),
                     "example": json.dumps(missing_l1[0])[:200] if missing_l1 else ""})
    sp = pd.DataFrame(rows)
    sp.to_csv(os.path.join(OUT, "v6_world_specs.csv"), index=False)
    out = {"spec_level": {g: {"specs": int(len(x)), "with_rule_txns": int((x.n_rule_txns > 0).sum()),
                              "missing_in_L1_window": int((x.missing_in_L1_window > 0).sum()),
                              "missing_in_history_span": int((x.missing_in_history_span > 0).sum())}
                          for g, x in sp.groupby("group")}}
    out["spec_examples"] = sp[sp.missing_in_L1_window > 0].head(8)[["sid", "group", "example"]].to_dict(orient="records")

    # (b) exact check on the as-trained realistic prompts
    HL = re.compile(r"^- (\d{4}-\d{2}-\d{2}) \| [+-]?[^\d]*([\d,]+\.\d{2})", re.M)
    rr = []
    for fn in ("sft_train", "sft_val", "grpo_train"):
        for i, r in enumerate(load_jsonl(os.path.join(CUR, fn + ".jsonl"))):
            if r["level"] not in ("L1", "L2", "L3"):
                continue
            sys_txt = r["prompt"][0]["content"]
            a, b = sys_txt.find("## Member"), sys_txt.find("## Tools")
            mb = sys_txt[a:b]
            hist = [(d, float(x.replace(",", ""))) for d, x in HL.findall(mb)]
            if not hist:
                continue
            start = min(d for d, _ in hist)
            hk = {(d, round(x, 2)) for d, x in hist}
            missing = []
            for msg in r["prompt"][1:]:
                if msg["role"] != "user" or "<tool_result" not in msg["content"]:
                    continue
                body = re.sub(r"^<tool_result[^>]*>\s*|\s*</tool_result>\s*$", "", msg["content"].strip())
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                found = []
                txn_like(obj, found)
                for o in found:
                    if o["date"] >= start and (o["date"], round(o["amount"], 2)) not in hk:
                        missing.append(o)
            rr.append({"file": fn, "line": i, "sid": r["scenario_id"], "twin": r["twin"][0], "level": r["level"],
                       "turn": r.get("turn", "decision"), "n_missing": len(missing),
                       "example": json.dumps(missing[0])[:200] if missing else ""})
    rw = pd.DataFrame(rr)
    rw.to_csv(os.path.join(OUT, "v6_world_rows.csv"), index=False)
    out["training_rows_L1to3"] = {fn: {"rows_with_history": int(len(g)), "rows_tool_txn_missing_from_history": int((g.n_missing > 0).sum()),
                                       "A": int(((g.n_missing > 0) & (g.twin == "A")).sum()),
                                       "B": int(((g.n_missing > 0) & (g.twin == "B")).sum()),
                                       "scenarios": sorted(g[g.n_missing > 0].sid.unique().tolist())}
                                  for fn, g in rw.groupby("file")}
    out["training_row_examples"] = rw[rw.n_missing > 0].head(6).to_dict(orient="records")
    dump(out, "v6_world.json")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
