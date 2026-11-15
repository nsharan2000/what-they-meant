"""GRPO v2 prompt order: every optimizer step sees both twins of two minimal pairs at one realism level.

WHY (experiments/dataset_audit/REPORT.md, P2 / GD-3 / GD-7, 2026-11-14): the first GRPO run (lr 5e-6, 30 steps, shuffled prompts)
moved its LoRA 1.8 % of SFT's ||dW||. v2 raises the learning rate to 5e-5, and this file makes each update contrastive: with
train_grpo.py --no-shuffle (TRL shuffle_dataset=False; RepeatSampler then takes prompts in file order, 4 per generation round =
one optimizer step at batch 4 x 8 generations x grad-accum 8), step k trains on rows 4k..4k+3 = [A1, B1, A2, B2]: two CELLS,
a cell being both twins of one pair at one level, rendered with the same prompt setting (P0/P2 was drawn per twin in v1.2.1, so
only 134 of 247 cells qualify). Both cells of a step share a level and have similar prompt lengths (length bucketing: the
padded generation batch wastes less time; the audit predicts ~526 s/step vs ~735 s shuffled). Steps are then shuffled with
--seed so levels interleave. Cells are spread over as many distinct pairs as possible (one cell per pair first), and the level
mix follows the cells available per level (largest-remainder quota).

Only the ORDER and the SUBSET of the approved grpo_train.jsonl rows change; every row is copied byte for byte.
--exclude <experiments/dataset_audit/lead/results/exclusion_list.json> drops every cell that holds one of the 44 flagged rows
(grpo_train 0-based line numbers) — a curation change that needs the user's approval (questions-for-user Q12).

  python3 training/build_grpo_v2.py --data /workspace/training/data_v121_curated/grpo_train.jsonl \
      --out /workspace/training/data_v121_grpo_v2 --steps 36 [--exclude .../exclusion_list.json] [--seed 0]
Writes <out>/grpo_train.jsonl (exactly steps x 4 rows) and <out>/manifest.json; prints GRPO_V2_DATA_OK <out>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

LEVEL_ORDER = ("L0r", "L1", "L2", "L3")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_chars(row: dict) -> int:
    return sum(len(m.get("content") or "") for m in row["prompt"])


def build_cells(lines: list[str], excluded: set[int]) -> tuple[list[dict], dict]:
    """Cells = (pair_id, level) with exactly one A and one B row in the same prompt_setting; returns cells + counts."""
    by_key: dict[tuple, dict] = defaultdict(dict)
    for i, line in enumerate(lines):
        r = json.loads(line)
        twin = r["twin"][0]
        key = (r["pair_id"], r["level"])
        if twin in by_key[key]:
            raise SystemExit(f"GRPO_V2_DATA_FAIL duplicate twin {twin} for {key} (lines {by_key[key][twin][0]} and {i})")
        by_key[key][twin] = (i, r)
    both = {k: v for k, v in by_key.items() if "A" in v and "B" in v}
    same = {k: v for k, v in both.items() if v["A"][1]["prompt_setting"] == v["B"][1]["prompt_setting"]}
    cells, n_flagged = [], 0
    for (pair, level), v in sorted(same.items()):
        (ia, ra), (ib, rb) = v["A"], v["B"]
        if ia in excluded or ib in excluded:
            n_flagged += 1
            continue
        cells.append({"pair_id": pair, "level": level, "setting": ra["prompt_setting"], "lines": [ia, ib],
                      "scenario_ids": [ra["scenario_id"], rb["scenario_id"]],
                      "chars": (prompt_chars(ra) + prompt_chars(rb)) // 2})
    counts = {"cells_with_both_twins": len(both), "same_setting_cells": len(same), "flagged_cells_dropped": n_flagged,
              "eligible_cells": len(cells),
              "eligible_by_level": dict(sorted(Counter(c["level"] for c in cells).items()))}
    return cells, counts


def level_quota(avail: dict[str, int], steps: int) -> dict[str, int]:
    """Steps per level proportional to the cells available (largest remainder), capped at floor(avail / 2)."""
    total = sum(avail.values())
    cap = {lv: n // 2 for lv, n in avail.items()}
    if sum(cap.values()) < steps:
        raise SystemExit(f"GRPO_V2_DATA_FAIL only {sum(cap.values())} two-cell steps available, {steps} requested")
    raw = {lv: steps * n / total for lv, n in avail.items()}
    q = {lv: min(cap[lv], int(raw[lv])) for lv in avail}
    while sum(q.values()) < steps:   # hand out the remaining steps by largest remainder among levels with spare cells
        lv = max((lv for lv in avail if q[lv] < cap[lv]), key=lambda lv: (raw[lv] - q[lv], -LEVEL_ORDER.index(lv)
                                                                          if lv in LEVEL_ORDER else 0))
        q[lv] += 1
    return q


def choose(cells: list[dict], quota: dict[str, int], rng: random.Random) -> dict[str, list[dict]]:
    """2 x quota cells per level, preferring pairs not used yet (the 4 renderings of one pair are near-duplicates)."""
    by_level = defaultdict(list)
    for c in cells:
        by_level[c["level"]].append(c)
    for lv in by_level:
        rng.shuffle(by_level[lv])
    picked: dict[str, list[dict]] = {lv: [] for lv in quota}
    used_pairs: Counter = Counter()
    # round-robin over levels so no level grabs all the fresh pairs first
    for rnd in range(2):   # round 0: only unused pairs; round 1: anything left
        progress = True
        while progress:
            progress = False
            for lv in sorted(quota, key=lambda x: LEVEL_ORDER.index(x) if x in LEVEL_ORDER else 99):
                need = 2 * quota[lv] - len(picked[lv])
                if need <= 0:
                    continue
                for c in by_level[lv]:
                    if c.get("_taken") or (rnd == 0 and used_pairs[c["pair_id"]]):
                        continue
                    c["_taken"] = True
                    picked[lv].append(c)
                    used_pairs[c["pair_id"]] += 1
                    progress = True
                    break
    for lv, n in quota.items():
        if len(picked[lv]) != 2 * n:
            raise SystemExit(f"GRPO_V2_DATA_FAIL level {lv}: picked {len(picked[lv])} cells, need {2 * n}")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="grpo_train.jsonl (training/from_curated.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--exclude", default=None, help="exclusion_list.json (its grpo_train rows, 0-based lines)")
    ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args()

    lines = [l for l in open(ns.data, encoding="utf-8").read().split("\n") if l.strip()]
    excluded: set[int] = set()
    if ns.exclude:
        ex = json.load(open(ns.exclude))["grpo_train"]["rows"]
        excluded = {int(x["line"]) for x in ex}
        for x in ex:   # the list must describe THIS file
            r = json.loads(lines[int(x["line"])])
            if (r["scenario_id"], r["level"], r["prompt_setting"]) != (x["scenario_id"], x["level"], x["prompt_setting"]):
                raise SystemExit(f"GRPO_V2_DATA_FAIL exclusion line {x['line']} does not match {ns.data}")
    cells, counts = build_cells(lines, excluded)
    avail = Counter(c["level"] for c in cells)
    quota = level_quota(dict(avail), ns.steps)
    rng = random.Random(ns.seed)
    picked = choose(cells, quota, rng)

    steps = []
    for lv, cs in picked.items():   # length bucketing: neighbours by prompt length share a step
        cs = sorted(cs, key=lambda c: (c["chars"], c["pair_id"]))
        for j in range(0, len(cs), 2):
            steps.append({"level": lv, "cells": [cs[j], cs[j + 1]]})
    steps.sort(key=lambda s: (LEVEL_ORDER.index(s["level"]) if s["level"] in LEVEL_ORDER else 99, s["cells"][0]["chars"]))
    rng.shuffle(steps)

    os.makedirs(ns.out, exist_ok=True)
    out_rows, manifest_steps = [], []
    for k, s in enumerate(steps):
        order = []
        for c in s["cells"]:
            order += c["lines"]   # [A, B] per cell -> [A1, B1, A2, B2]
        out_rows += [lines[i] for i in order]
        manifest_steps.append({"step": k + 1, "level": s["level"], "settings": [c["setting"] for c in s["cells"]],
                               "pairs": [c["pair_id"] for c in s["cells"]],
                               "scenario_ids": [sid for c in s["cells"] for sid in c["scenario_ids"]],
                               "src_lines": order, "mean_chars": [c["chars"] for c in s["cells"]]})
    assert len(out_rows) == 4 * ns.steps
    for k in range(0, len(out_rows), 4):   # invariants the trainer relies on
        rs = [json.loads(x) for x in out_rows[k:k + 4]]
        assert [r["twin"][0] for r in rs] == ["A", "B", "A", "B"], k
        assert len({r["level"] for r in rs}) == 1, k
        assert rs[0]["pair_id"] == rs[1]["pair_id"] and rs[2]["pair_id"] == rs[3]["pair_id"], k
        assert rs[0]["prompt_setting"] == rs[1]["prompt_setting"] and rs[2]["prompt_setting"] == rs[3]["prompt_setting"], k
    used = [i for s in manifest_steps for i in s["src_lines"]]
    assert len(used) == len(set(used)), "a row is used twice"
    out_path = os.path.join(ns.out, "grpo_train.jsonl")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_rows) + "\n")
    os.replace(tmp, out_path)
    rows_out = [json.loads(x) for x in out_rows]
    manifest = {
        "script": "training/build_grpo_v2.py", "args": vars(ns), "input": ns.data, "input_sha256": sha256_file(ns.data),
        "output": out_path, "output_sha256": sha256_file(out_path), "n_rows": len(out_rows), "steps": ns.steps,
        "exclusion_applied": bool(ns.exclude), "n_excluded_rows": len(excluded), **counts,
        "steps_per_level": dict(sorted(quota.items())),
        "distinct_pairs": len({r["pair_id"] for r in rows_out}),
        "settings": dict(Counter(r["prompt_setting"] for r in rows_out)),
        "b_expected": dict(Counter(json.loads(r["scenario"])["expected"]["decision"] for r in rows_out if r["twin"][0] == "B")),
        "step_composition": manifest_steps,
    }
    tmp = os.path.join(ns.out, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, os.path.join(ns.out, "manifest.json"))
    print(json.dumps({k: manifest[k] for k in ("n_rows", "eligible_cells", "same_setting_cells", "flagged_cells_dropped",
                                                 "steps_per_level", "distinct_pairs", "settings", "b_expected")}))
    print(f"GRPO_V2_DATA_OK {ns.out}", flush=True)


if __name__ == "__main__":
    main()
