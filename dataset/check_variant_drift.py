"""Clash-type drift check for synthetic variants (split hygiene), with an OPEN model (training-data provenance policy Q5: no Claude).

Why: a variant must keep its seed's kind of clash, and it inherits the seed's split. A variant that drifted to ANOTHER seed's clash
type (e.g. a refund-then-dispute seed whose variant became a duplicate-invoice case) would carry that incident pattern into the
seed's split — if the matching seed sits in a different split, the pattern leaks from train to test.

For every variant twin-A row, the model sees the 45 seed clash patterns (id, request, what to check, clash, tags — origin NOT
revealed) and the variant (request, decisive facts, clash, ideal question) and names the best and second-best matching pattern.
Decision per pair: keep if best == origin; keep if the best-matching seed lives in the variant's split ("drift_same_split");
keep if best and origin share a clash FAMILY (seed tags, FAMILY below — e.g. near-miss payee cases already span all three splits in
the seed set itself, so a near-miss variant resembling another near-miss seed is not a leak); keep if the origin is the second
choice and the model marks both as the same kind. Otherwise the variant drifted into ANOTHER family that lives in another split →
drop ("leak_drop"). Writes dataset/variant_drift.json; --apply moves dropped pairs to dataset/scenarios_synth_dropped.jsonl.

python3 dataset/check_variant_drift.py [--model deepseek-ai/DeepSeek-V3.2:deepinfra] [--apply]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.drivers.openai_compat import OpenAICompat  # noqa: E402
from wtm.spec import ROOT, load_jsonl  # noqa: E402

FAMILY = {"near_miss_name": "payee", "typo": "payee", "new:forgotten_subscription": "subscription",
          "new:duplicate_payment": "duplicate", "new:refund_then_reinvoice": "duplicate", "new:refund_then_dispute": "dispute",
          "new:scam_pressure": "scam", "new:runaway_loop": "loop", "new:unit_mismatch": "item", "new:wrong_payment_method": "method",
          "new:wrong_payment_type": "method", "new:wrong_asset": "method", "new:institution_error": "statement"}


def families(seed_row: dict) -> set:
    f = {FAMILY[t] for t in seed_row.get("tags", []) if t in FAMILY}
    return f or {seed_row["pair_id"]}          # seeds without a family tag form their own family

SYSTEM = """You classify incidents from payment and finance apps by the KIND of mismatch between what a customer asked for and what
the records show (wrong payee, duplicate payment, auto-renewing trial, open dispute, changed bank details, scam pressure, …).
You compare the kind of mismatch, not the product, amounts or names. You answer with JSON only."""

PROMPT = """## Known incident patterns
{patterns}

## New scenario
Request: "{request}"
What the assistant finds when it looks things up: {facts}
The mismatch: {clash}
Ideal question to the customer: "{ideal}"

Which known pattern has the SAME KIND of mismatch as the new scenario? Return ONLY:
{{"best": "<pattern id>", "second": "<pattern id or null>", "best_and_second_same_kind": true|false, "reason": "<= 25 words"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2:deepinfra")
    ap.add_argument("--cache", default=os.environ.get("WTM_CACHE", "/tmp/wtm_cache/variant_drift"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "variant_drift.json"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ns = ap.parse_args()
    seed = [r for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios", "*.jsonl"))) for r in load_jsonl(f)]
    seed_a = [r for r in seed if r["twin"].startswith("A") and "push" not in r["twin"]]
    synth_files = sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios_synth", "*.jsonl")))
    synth = [r for f in synth_files for r in load_jsonl(f)]
    splits = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    split_of = {pid: k for k in ("train", "val", "test") for pid in splits.get(k, [])}
    patterns = "\n".join(
        f'{r["pair_id"]}: request "{r["user_message"][:110]}" · check: {r["expected"].get("must_check", "")} · clash: '
        f'{(r["expected"].get("discrepancy") or "")[:140]} · tags: {", ".join(r.get("tags", []))}' for r in seed_a)
    drv = OpenAICompat(model=ns.model, temperature=0.0, max_tokens=300, cache_dir=ns.cache)

    def classify(v):
        e = v["expected"]
        prompt = PROMPT.format(patterns=patterns, request=v["user_message"], facts="; ".join(v.get("decisive_facts", []))[:700],
                               clash=(e.get("discrepancy") or e.get("must_check") or "")[:300], ideal=(e.get("ideal_message") or "")[:300])
        res = drv.chat(SYSTEM, [{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", res.text or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        return v, d

    fam = {r["pair_id"]: families(r) for r in seed_a}
    var_a = [v for v in synth if v["twin"].startswith("A")]
    out, drop = [], set()
    with cf.ThreadPoolExecutor(max_workers=ns.workers) as ex:
        for v, d in ex.map(classify, var_a):
            origin, best, second = v["variant_of"], (d.get("best") or "").strip(), (d.get("second") or "") or ""
            vs = split_of.get(v["pair_id"])
            if not best:
                verdict = "unparsed_keep"
            elif best == origin:
                verdict = "same_as_origin"
            elif split_of.get(best) == vs:
                verdict = "drift_same_split"
            elif best in fam and origin in fam and fam[best] & fam[origin]:
                verdict = "drift_same_family_tag"
            elif second == origin and d.get("best_and_second_same_kind"):
                verdict = "drift_same_family"
            else:
                verdict = "leak_drop"
                drop.add(v["pair_id"])
            out.append({"variant": v["pair_id"], "origin": origin, "split": vs, "best": best, "best_split": split_of.get(best),
                        "origin_family": sorted(fam.get(origin, [])), "best_family": sorted(fam.get(best, [])),
                        "second": second, "same_kind": d.get("best_and_second_same_kind"), "reason": d.get("reason"),
                        "verdict": verdict})
    out.sort(key=lambda x: x["variant"])
    counts = {}
    for o in out:
        counts[o["verdict"]] = counts.get(o["verdict"], 0) + 1
    report = {"model": ns.model, "n_variant_pairs": len(out), "counts": counts, "dropped_pairs": sorted(drop),
              "tokens": [drv.total_prompt_tokens, drv.total_completion_tokens], "rows": out}
    json.dump(report, open(ns.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    if ns.apply and drop:
        dropped_path = os.path.join(ROOT, "dataset", "scenarios_synth_dropped.jsonl")
        kept_total = 0
        with open(dropped_path, "a") as dfh:
            for f in synth_files:
                rows = load_jsonl(f)
                keep = [r for r in rows if r["pair_id"] not in drop]
                for r in rows:
                    if r["pair_id"] in drop:
                        dfh.write(json.dumps(r | {"dropped_reason": "clash-type drift into another split (check_variant_drift.py)"},
                                             ensure_ascii=False) + "\n")
                with open(f, "w") as fh:
                    for r in keep:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept_total += len(keep)
        print(f"applied: {len(drop)} pairs moved to {dropped_path}; {kept_total} variant rows remain")


if __name__ == "__main__":
    main()
