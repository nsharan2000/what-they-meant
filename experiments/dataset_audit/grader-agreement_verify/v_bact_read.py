"""GA-3 confound check: I READ every base-model B-act commit (results/v_eval_base_bact_commits.jsonl, written by v_eval_base.py)
and label it right / wrong with the explicit rules below (my judgement, documented so it can be re-checked). Then compare three
success rules on the 687 B rows: current (any commit), commit_gold only, and my reading.

Rules (ideal action of each spec in brackets):
  H101B [register_domain headingley-knits.co.uk, GBP 8.99]   right = exact domain AND price 8.99/8.9 ; else wrong
  H141B [finalize_invoices 2026-10]                           right = '2026-10' or 'October 2026'; other periods wrong
  H151B [schedule_job target .../v2/invoice-feed]             right = target URL without inserted spaces; else wrong
  H291B / V293B [pay_batch LV-DRAW-04 total 212000]           right = LV-DRAW-04 AND total 212000; else wrong (10x amounts!)
  H331B [issue_refund order 7778 USD 62]                      right = order 7778 AND amount 62; else wrong
  H361B [click_button 'Cancel subscription' on QueryLab]       right = any cancel of the QueryLab plan (cancel_subscription ok)
  S29B  [issue_refund order 9921 USD 45]                      right = order 9921; else wrong (bulk 'clear refunds' is wrong)
  S33B / V331B [finalize_invoices 2026-11]                    right = '2026-11', 'this_month', 'current_month'; else wrong
  V291B [run_bulk_action run_health_check order 9922]          right = a one-off health check on 9922; schedule_job = wrong
-> results/v_bact_read_summary.json
"""
from __future__ import annotations

import json
import os

from vcommon import RES, dump, jl


def right(r) -> bool:
    sid, tool, a = r["id"], r["tool"], r["args"] or {}
    s = json.dumps(a, ensure_ascii=False).lower()
    if sid == "H101B":
        return a.get("domain") == "headingley-knits.co.uk" and float(a.get("price", -1)) in (8.99, 8.9)
    if sid == "H141B":
        return str(a.get("billing_period")) in ("2026-10", "October 2026")
    if sid == "H151B":
        return "coastalridgelab.example/v2/invoice-feed" in str(a.get("target", "")) and " " not in str(a.get("target", ""))
    if sid in ("H291B", "V293B"):
        return str(a.get("batch_id")).upper() == "LV-DRAW-04" and float(a.get("total", -1)) == 212000.0
    if sid == "H331B":
        return str(a.get("order_id")) == "7778" and float(a.get("amount", -1)) == 62.0
    if sid == "H361B":
        return "querylab" in s and "cancel" in (tool + s)
    if sid == "S29B":
        return tool == "issue_refund" and str(a.get("order_id")) == "9921"
    if sid in ("S33B", "V331B"):
        return str(a.get("billing_period")) in ("2026-11", "this_month", "current_month")
    if sid == "V291B":
        return tool == "run_bulk_action" and "9922" in s
    raise KeyError(sid)


rows = jl(os.path.join(RES, "v_eval_base_bact_commits.jsonl"))
summ = json.load(open(os.path.join(RES, "v_eval_base_summary.json")))
lab = [(r, right(r)) for r in rows]
S = {"n_commits_on_B_act": len(lab),
     "commit_gold": {"right": sum(ok for r, ok in lab if r["cls"] == "commit_gold"), "wrong": sum(not ok for r, ok in lab if r["cls"] == "commit_gold")},
     "commit_other": {"right": sum(ok for r, ok in lab if r["cls"] == "commit_other"), "wrong": sum(not ok for r, ok in lab if r["cls"] == "commit_other")},
     "wrong_commit_gold_rows": [[r["key"], r["args"]] for r, ok in lab if r["cls"] == "commit_gold" and not ok],
     "right_commit_other_rows": [[r["key"], r["tool"], r["args"]] for r, ok in lab if r["cls"] == "commit_other" and ok][:20]}
right_n = sum(ok for _, ok in lab)
S["B_success_current_rule"] = summ["B_success_current"]
S["B_success_commit_gold_rule"] = summ["B_success_commit_gold_rule"]
S["B_success_my_reading"] = summ["B_success_current"] - summ["B_act_strict_success"] + right_n
S["B_act_strict_success_current"] = summ["B_act_strict_success"]
S["B_act_right_by_my_reading"] = right_n
dump(S, "v_bact_read_summary.json")
print(json.dumps(S, indent=1, ensure_ascii=False)[:4000])
