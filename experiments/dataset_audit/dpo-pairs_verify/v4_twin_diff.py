"""Verifier step 4: where do the A and B twin prompts differ, measured from the end of the prompt (the generation point)?

For each pair_id x level with both twins in dpo_train.jsonl: the P2 hint is removed from both system prompts (the hint is drawn
independently per rendering), every message is split into lines, and difflib marks the lines that differ. Distances from the
end are in characters, converted to tokens with each row's own ratio (exact chat-template prompt tokens from
experiments/dataset_audit/results/token_stats.jsonl / total prompt characters), so they are approximate (+-10 %).
Reported per twin-level, using the A twin's prompt (the B twin's gives the same picture; both are stored):
  nearest_diff_tok  distance from the end to the closest differing line (where the decisive contrast ends)
  farthest_diff_tok distance from the end to the earliest differing line
  where             which messages differ: system / request / lookup-call / lookup-result
Output: results/v4_twin_diff.json, results/v4_twin_diff_rows.csv
"""
from __future__ import annotations

import csv
import difflib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.render import golden_text  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HINT = golden_text("P2")
dpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))]
tok = [t for t in (json.loads(l) for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))
       if t["file"] == "dpo_train.jsonl"]


def msgs(d):
    out = []
    for j, m in enumerate(d["prompt"]):
        c = m["content"]
        if j == 0 and d["prompt_setting"] == "P2":
            c = c.replace("\n\n" + HINT, "")
        role = "system" if j == 0 else "request" if j == 1 else ("lookup-call" if m["role"] == "assistant" else "lookup-result")
        out.append((role, c))
    return out


def diff_positions(ma, mb, total_chars):
    """for prompt A: list of (role, chars_from_end) of every line that differs from B (same message index)"""
    res = []
    offs = 0
    starts = []
    for role, c in ma:
        starts.append(offs)
        offs += len(c) + 1
    for j, (role, c) in enumerate(ma):
        other = mb[j][1] if j < len(mb) else ""
        la, lb = c.split("\n"), other.split("\n")
        sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
        pos = starts[j]
        line_starts = []
        p = pos
        for ln in la:
            line_starts.append(p)
            p += len(ln) + 1
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if i2 > i1:
                for i in range(i1, i2):
                    res.append((role, total_chars - line_starts[i]))
            else:   # insertion in B only: mark the position in A
                at = line_starts[i1] if i1 < len(line_starts) else p
                res.append((role, total_chars - at))
    if len(mb) > len(ma):
        res.append(("extra-msg-in-B", 0))
    return res


by = defaultdict(dict)
for i, d in enumerate(dpo):
    by[(d["pair_id"], d["level"])][d["twin"][:1]] = i
rows = []
for (pid, lv), v in sorted(by.items()):
    if "A" not in v or "B" not in v:
        continue
    ia, ib = v["A"], v["B"]
    ma, mb = msgs(dpo[ia]), msgs(dpo[ib])
    ca = sum(len(c) + 1 for _, c in ma)
    ratio = tok[ia]["prompt_tokens"] / ca
    pos = diff_positions(ma, mb, ca)
    if not pos:
        rows.append({"pair_id": pid, "level": lv, "idx_A": ia, "idx_B": ib, "no_difference": True})
        continue
    roles = sorted({r for r, _ in pos})
    near = min(p for _, p in pos) * ratio
    far = max(p for _, p in pos) * ratio
    rows.append({"pair_id": pid, "level": lv, "idx_A": ia, "idx_B": ib, "no_difference": False, "where": "+".join(roles),
                 "nearest_diff_tok": round(near), "farthest_diff_tok": round(far), "prompt_tokens_A": tok[ia]["prompt_tokens"],
                 "n_diff_lines": len(pos), "any_diff_beyond_4096": far > 4096})
ok = [r for r in rows if not r["no_difference"]]
near = np.array([r["nearest_diff_tok"] for r in ok])
summary = {
    "twin_levels": len(rows), "with_no_difference": sum(r["no_difference"] for r in rows),
    "nearest_diff_tok_median": float(np.median(near)), "nearest_diff_tok_p90": float(np.percentile(near, 90)),
    "where": dict(Counter(r["where"] for r in ok).most_common()),
    "only_lookup_result_differs": sum(1 for r in ok if r["where"] == "lookup-result"),
    "any_diff_beyond_4096_tokens_from_end": sum(r["any_diff_beyond_4096"] for r in ok),
    "beyond_4096_rows": [[r["pair_id"], r["level"], r["farthest_diff_tok"], r["where"]] for r in ok if r["any_diff_beyond_4096"]],
    "nearest_beyond_4096": sum(1 for r in ok if r["nearest_diff_tok"] > 4096),
    "method_note": "token distances approximated from characters with each row's exact tokens/characters ratio",
}
json.dump(summary, open(os.path.join(OUT, "v4_twin_diff.json"), "w"), indent=1)
with open(os.path.join(OUT, "v4_twin_diff_rows.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(json.dumps(summary, indent=1))
