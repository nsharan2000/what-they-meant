"""Map CuratorKIT's exports back to our original training rows (CuratorKIT decides WHICH rows survive; the text is never touched).

Why: CuratorKIT has no tool-call / trajectory support, so dataset/curate.py hands it each row's prompt message list as a JSON
string. Its "messages" exporter writes that string back as ONE user message (VERIFIED 2026-11-14 on the v1.2 export:
sft/sft_messages.jsonl rows are [user: json.dumps(prompt), assistant: target]). Training on the export directly would teach a
wrong format (a JSON blob instead of system/user/assistant turns), so the trainers read the rows this script writes: the original
rows (all metadata kept) whose content survived CuratorKIT's exact dedup + SecretsGate.

  sft_train.jsonl  original SFT rows matched to sft/sft_messages.jsonl (first occurrence of an exact duplicate)
  sft_val.jsonl    copied unchanged (validation loss only; not curated)
  dpo_train.jsonl  original DPO rows matched to dpo/dpo.jsonl
  grpo_train.jsonl GRPO prompts whose prompt survived the DPO curation (deduplicated by prompt)

python3 training/from_curated.py --data /workspace/training/data_v12 --curated /workspace/dataset/curated \
    --out /workspace/training/data_v12_curated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def dump(rows, p):
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def h(*parts):
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def norm(s):
    """Whitespace/Unicode-insensitive form, used only when CuratorKIT's text cleaning changed a string."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def j(x):
    return json.dumps(x, ensure_ascii=False)


def maybe_list(x):
    return json.loads(x) if isinstance(x, str) and x.startswith("[") else x


def sft_keys(prompt, target):
    p = j(prompt) if not isinstance(prompt, str) else prompt
    return h(p, target), h(norm(p), norm(target))


def dpo_keys(prompt, chosen, rejected):
    p, c, r = (x if isinstance(x, str) else j(x) for x in (prompt, chosen, rejected))
    return h(p, c, r), h(norm(p), norm(c), norm(r))


def match(orig, cur, okeys, ckeys):
    """Return (kept original rows in curated order, stats). Exact key first, normalised key as fallback.

    Two originals that differ only in whitespace/Unicode form share a normalised key: once CuratorKIT has cleaned the text, a
    curated row could map back to either of them, so the mapping would be ambiguous. That fails the run instead of guessing."""
    exact, loose, exact_of_loose = {}, {}, {}
    for r in orig:
        ke, kn = okeys(r)
        exact.setdefault(ke, r)
        loose.setdefault(kn, r)
        if exact_of_loose.setdefault(kn, ke) != ke:
            first = exact[exact_of_loose[kn]]
            raise SystemExit(f"FROM_CURATED_FAIL: two different original rows have the same normalised content and cannot be "
                             f"told apart after CuratorKIT: {first.get('scenario_id')}/{first.get('level')}/{first.get('turn')} "
                             f"and {r.get('scenario_id')}/{r.get('level')}/{r.get('turn')}")
    kept, seen, st = [], set(), Counter()
    for c in cur:
        ke, kn = ckeys(c)
        r = exact.get(ke)
        if r is not None:
            st["exact"] += 1
        else:
            r = loose.get(kn)
            st["normalised_only" if r is not None else "unmatched"] += 1
        if r is None:
            continue
        rid = id(r)
        if rid in seen:
            st["duplicate_in_export"] += 1
            continue
        seen.add(rid)
        kept.append(r)
    st["original"], st["curated_export"], st["kept"] = len(orig), len(cur), len(kept)
    st["original_distinct_exact"] = len(exact)
    return kept, dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training views (build_datasets.py output)")
    ap.add_argument("--curated", required=True, help="dataset/curate.py output dir (sft/, dpo/)")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    os.makedirs(ns.out, exist_ok=True)
    report = {}

    sft = load(os.path.join(ns.data, "sft_train.jsonl"))
    sft_cur = load(os.path.join(ns.curated, "sft", "sft_messages.jsonl"))
    for c in sft_cur:   # the format problem this script exists for: prompt list flattened into one user message
        assert [m["role"] for m in c["messages"]] == ["user", "assistant"], c["messages"][0]["role"]
    kept_sft, report["sft"] = match(
        sft, sft_cur,
        lambda r: sft_keys(r["prompt"], r["completion"][0]["content"]),
        lambda c: sft_keys(c["messages"][0]["content"], c["messages"][1]["content"]))
    dump(kept_sft, os.path.join(ns.out, "sft_train.jsonl"))
    if os.path.exists(os.path.join(ns.data, "sft_val.jsonl")):
        shutil.copy(os.path.join(ns.data, "sft_val.jsonl"), os.path.join(ns.out, "sft_val.jsonl"))
        report["sft_val_copied"] = len(load(os.path.join(ns.out, "sft_val.jsonl")))

    dpo = load(os.path.join(ns.data, "dpo_train.jsonl"))
    dpo_cur = load(os.path.join(ns.curated, "dpo", "dpo.jsonl"))
    kept_dpo, report["dpo"] = match(
        dpo, dpo_cur,
        lambda r: dpo_keys(r["prompt"], r["chosen"], r["rejected"]),
        lambda c: dpo_keys(maybe_list(c["prompt"]), maybe_list(c["chosen"]), maybe_list(c["rejected"])))
    dump(kept_dpo, os.path.join(ns.out, "dpo_train.jsonl"))

    grpo = load(os.path.join(ns.data, "grpo_train.jsonl"))
    ok_prompts = {h(j(r["prompt"])) for r in kept_dpo}
    kept_grpo, seen = [], set()
    for r in grpo:
        k = h(j(r["prompt"]))
        if k in ok_prompts and k not in seen:
            seen.add(k)
            kept_grpo.append(r)
    report["grpo"] = {"original": len(grpo), "kept": len(kept_grpo), "rule": "prompt survived the DPO curation; one row per prompt"}
    dump(kept_grpo, os.path.join(ns.out, "grpo_train.jsonl"))

    # composition of what survives (twins must stay balanced; no test/hard pair may appear)
    def comp(rows):
        return {"twin": dict(Counter(r.get("twin", "?")[:1] for r in rows)),
                "level": dict(sorted(Counter(r.get("level", "?") for r in rows).items())),
                "prompt_setting": dict(Counter(r.get("prompt_setting", "?") for r in rows)),
                "turn": dict(Counter(str(r.get("turn", "-")) for r in rows))}
    report["composition"] = {"sft_train": comp(kept_sft), "dpo_train": comp(kept_dpo), "grpo_train": comp(kept_grpo)}
    json.dump(report, open(os.path.join(ns.out, "from_curated_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    bad = report["sft"].get("unmatched", 0) + report["dpo"].get("unmatched", 0)
    if bad:
        raise SystemExit(f"FROM_CURATED_FAIL: {bad} curated rows could not be mapped back to an original row")
    print("FROM_CURATED_OK")


if __name__ == "__main__":
    main()
