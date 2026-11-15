"""VERIFY checked_ok 'labels do not depend on level': per scenario, the coarse class of the SFT decision target
(request_approval card / other tool call = commit / chat message) at every level x prompt setting; and whether the exact target
text differs across levels (it may legitimately, e.g. L3 noisy wording -- reported, not judged).
Outputs: results/v12_target_invariance.json
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from vcommon import CUR, dump, load_jsonl


def coarse(t: str) -> str:
    m = re.search(r'<tool_call>\s*\{\s*"name"\s*:\s*"([a-z_]+)"', t)
    if m:
        return "approve" if m.group(1) == "request_approval" else "tool:" + m.group(1)
    return "message:" + ("question" if "?" in t else "statement")


def main():
    by = defaultdict(set)
    texts = defaultdict(set)
    for fn in ("sft_train", "sft_val"):
        for r in load_jsonl(os.path.join(CUR, fn + ".jsonl")):
            if r["turn"] != "decision":
                continue
            t = r["completion"][0]["content"]
            by[r["scenario_id"]].add(coarse(t))
            texts[r["scenario_id"]].add(t.strip())
    varying = {k: sorted(v) for k, v in by.items() if len(v) > 1}
    out = {"n_scenarios": len(by), "n_with_varying_coarse_class": len(varying), "varying": varying,
           "n_with_varying_exact_text": sum(1 for v in texts.values() if len(v) > 1)}
    dump(out, "v12_target_invariance.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
