"""Rendering defect check: scenario_records that already exist in the member file get added a second time (wtm/render.py
render() -> _with_extra(member, scenario_records)), so the realistic prompts (L1/L2/L2s/L3: member block rendered from records)
and the view tools show the decisive records twice. L0r is unaffected (it uses the seed's own l0_member_block).

Per scenario spec (training pool dataset/scenarios + dataset/scenarios_synth, eval files dataset/eval/*): count scenario_records
entries whose record is already in the member file (same 'id' for dict records with an id; exact equality otherwise).
Per training row (sft_train/sft_val/grpo_train/dpo_train, levels L1/L2/L3): count rows with a duplicated history line, and rows
with any duplicated bullet line inside the member block (history, notes, contacts, accounts ...) of the system prompt.
Also: the M1 per-prompt success on duplicated vs clean prompts (from results/consistency_rows.csv + results/m1_samples.csv) is in
results/consistency.json (written by consistency.py).
Outputs: results/dup_records.json
Run: python3 -B experiments/dataset_audit/realism-pressure/dup_records.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

from common import ROOT, dump

LINE_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) \| [^\n]*$", re.M)


def is_dup(rec, existing):
    if isinstance(rec, dict) and rec.get("id"):
        return any(isinstance(e, dict) and e.get("id") == rec["id"] for e in existing)
    return rec in existing


def main():
    members = {}
    out = {"spec_level": {}, "training_rows": {}, "examples": []}
    groups = {"train_pool": ["dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"],
              "eval_test": ["dataset/eval/test_*.jsonl"], "eval_hard_test": ["dataset/eval/hard_test_*.jsonl"],
              "eval_val": ["dataset/eval/val_*.jsonl"], "eval_hard_train": ["dataset/eval/hard_train_*.jsonl"]}
    for gname, pats in groups.items():
        n = n_dup = 0
        by_field = Counter()
        by_twin = Counter()
        ids = []
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(ROOT, pat))):
                for l in open(f):
                    s = json.loads(l)
                    key = (s["env_id"], s["member_id"])
                    if key not in members:
                        members[key] = json.load(open(os.path.join(ROOT, "envs", s["env_id"], "members", f"{s['member_id']}.json")))
                    m = members[key]
                    n += 1
                    dups = [x["field"] for x in (s.get("scenario_records") or [])
                            if is_dup(x["record"], m.get(x["field"], []))]
                    if dups:
                        n_dup += 1
                        by_field.update(dups)
                        by_twin[s["twin"][0]] += 1
                        ids.append(s["id"])
                        if len(out["examples"]) < 6 and gname in ("train_pool", "eval_test", "eval_hard_test"):
                            out["examples"].append({"group": gname, "id": s["id"], "file": os.path.relpath(f, ROOT),
                                                    "member": f"envs/{s['env_id']}/members/{s['member_id']}.json",
                                                    "duplicated_fields": dups})
        out["spec_level"][gname] = {"scenarios": n, "with_duplicated_scenario_records": n_dup,
                                    "by_field": dict(by_field), "by_twin": dict(by_twin), "ids": ids}
    for fn in ("sft_train.jsonl", "sft_val.jsonl", "grpo_train.jsonl", "dpo_train.jsonl"):
        c = defaultdict(lambda: [0, 0, 0])
        with open(os.path.join(ROOT, "dataset", "curated_v121", fn)) as fh:
            for l in fh:
                r = json.loads(l)
                if r["level"] == "L0r":
                    continue
                sysm = r["prompt"][0]["content"]
                lines = [x for x in sysm.splitlines() if LINE_RE.match(x)]
                dup = any(v > 1 for v in Counter(lines).values())
                mb = sysm[sysm.find("## Member"): sysm.find("## Tools")] if "## Member" in sysm else ""
                bullets = [x for x in mb.splitlines() if x.startswith("- ")]
                dup_any = any(v > 1 for v in Counter(bullets).values())
                k = f"{r['level']}|{r['twin'][0]}"
                c[k][0] += 1
                c[k][1] += int(dup)
                c[k][2] += int(dup_any)
        out["training_rows"][fn] = {k: {"rows": v[0], "rows_with_duplicated_history_line": v[1],
                                        "rows_with_any_duplicated_member_block_line": v[2]} for k, v in sorted(c.items())}
        out["training_rows"][fn]["total"] = {"rows": sum(v[0] for v in c.values()),
                                             "rows_with_duplicated_history_line": sum(v[1] for v in c.values()),
                                             "rows_with_any_duplicated_member_block_line": sum(v[2] for v in c.values())}
    # M1 per-prompt success on grpo prompts with / without any duplicated member-block line (realistic levels only)
    import pandas as pd
    from common import RES, boot_diff_unpaired
    flags = {}
    with open(os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl")) as fh:
        for i, l in enumerate(fh):
            r = json.loads(l)
            if r["level"] == "L0r":
                continue
            sysm = r["prompt"][0]["content"]
            mb = sysm[sysm.find("## Member"): sysm.find("## Tools")] if "## Member" in sysm else ""
            bullets = [x for x in mb.splitlines() if x.startswith("- ")]
            flags[i] = int(any(v > 1 for v in Counter(bullets).values()))
    m1 = pd.read_csv(os.path.join(RES, "m1_samples.csv"))
    pp = m1.groupby(["grpo_line", "twin"]).success.mean().reset_index()
    pp = pp[pp.grpo_line.isin(flags)].copy()
    pp["dup_any"] = pp.grpo_line.map(flags)
    out["m1_success_by_dup_any"] = []
    for tw in ("A", "B"):
        a = pp[(pp.twin == tw) & (pp.dup_any == 0)].success.to_numpy()
        b = pp[(pp.twin == tw) & (pp.dup_any == 1)].success.to_numpy()
        rec = {"twin": tw, "n_clean": int(len(a)), "n_dup": int(len(b))}
        if len(a) >= 3 and len(b) >= 3:
            d, lo, hi = boot_diff_unpaired(a, b, seed=61)
            rec.update({"success_clean": round(float(a.mean()), 4), "success_dup": round(float(b.mean()), 4),
                        "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]})
        out["m1_success_by_dup_any"].append(rec)
    dump(out, "dup_records.json")
    print(json.dumps(out["m1_success_by_dup_any"], indent=1))
    print(json.dumps({g: {k: v for k, v in d.items() if k != "ids"} for g, d in out["spec_level"].items()}, indent=1))
    print(json.dumps({fn: d["total"] for fn, d in out["training_rows"].items()}, indent=1))


if __name__ == "__main__":
    main()
