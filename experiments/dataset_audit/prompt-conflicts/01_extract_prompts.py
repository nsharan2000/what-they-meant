"""Step 1 — extract the system prompts actually used in training (from the data rows), per env x level x prompt setting.

Checks the rendering contract on every row of every file: system = env policy prefix (L0 template at L0r, L1 template at
L1/L2/L3) + member block + '## Tools' docs + the protocol block (wtm.protocol.PROTOCOL_INSTRUCTIONS) + tail, where the tail must
be exactly prompts/golden_short.md on P2 rows and empty on P0 rows.

Outputs: results/prompt_inventory.json, results/prompt_texts/<env>_<L0r|L1>_policy.txt, protocol.txt, p2_hint.txt
    python3 -B experiments/dataset_audit/prompt-conflicts/01_extract_prompts.py
"""
from __future__ import annotations

import collections
import os
import statistics

from pc_common import (FILES, HINT, PROTOCOL_INSTRUCTIONS, RES, dump, load_envs, load_rows, n_lookups, rel, split_system,
                       tools_in_docs)


def st(v):
    if not v:
        return None
    s = sorted(v)
    return {"n": len(v), "mean": round(statistics.mean(v), 1), "min": s[0], "p50": s[len(s) // 2], "max": s[-1]}


def main():
    envs = load_envs()
    os.makedirs(os.path.join(RES, "prompt_texts"), exist_ok=True)
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    violations = []
    policy_texts = {}
    tail_after_hint_chars = []
    for f in FILES:
        for i, r in enumerate(load_rows(f)):
            env = envs[r["env_id"]]
            sysm = r["prompt"][0]["content"]
            try:
                sec = split_system(sysm, env, r["level"])
            except ValueError as e:
                violations.append({"file": f, "i": i, "error": str(e)})
                continue
            key = (f, r["env_id"], r["level"], r["prompt_setting"])
            c = cells[key]
            c["system_chars"].append(len(sysm))
            c["policy_chars"].append(len(sec["policy"]))
            c["member_chars"].append(len(sec["member"]))
            c["tooldoc_chars"].append(len(sec["tools"]))
            c["n_tools"].append(len(tools_in_docs(sec["tools"])))
            c["conv_chars"].append(sum(len(m["content"]) for m in r["prompt"][1:]))
            c["n_lookups_in_prompt"].append(n_lookups(r["prompt"]))
            tail_ok = (sec["tail"] == HINT) if r["prompt_setting"] == "P2" else (sec["tail"] == "")
            if not tail_ok:
                violations.append({"file": f, "i": i, "error": f"tail mismatch for {r['prompt_setting']}: {sec['tail'][:80]!r}"})
            if r["prompt_setting"] == "P2":
                # characters between the end of the hint and the generation point (user msg + look-ups + results)
                tail_after_hint_chars.append(sum(len(m["content"]) for m in r["prompt"][1:]))
            lv = "L0r" if r["level"] == "L0r" else "L1"
            policy_texts.setdefault((r["env_id"], lv), set()).add(sec["policy"])
    for (e, lv), texts in sorted(policy_texts.items()):
        if len(texts) != 1:
            violations.append({"env": e, "level": lv, "error": f"{len(texts)} distinct policy texts"})
        with open(os.path.join(RES, "prompt_texts", f"{e}_{lv}_policy.txt"), "w") as fh:
            fh.write(sorted(texts)[0])
    with open(os.path.join(RES, "prompt_texts", "protocol.txt"), "w") as fh:
        fh.write(PROTOCOL_INSTRUCTIONS)
    with open(os.path.join(RES, "prompt_texts", "p2_hint.txt"), "w") as fh:
        fh.write(HINT)

    inventory = []
    for key in sorted(cells):
        f, e, lv, ps = key
        c = cells[key]
        inventory.append({"file": f, "env": e, "level": lv, "prompt_setting": ps, "rows": len(c["system_chars"]),
                          **{k: st(v) for k, v in c.items()}})
    policy_summary = {f"{e}/{lv}": {"chars": len(sorted(t)[0]), "words": len(sorted(t)[0].split()),
                                    "distinct_texts": len(t)} for (e, lv), t in sorted(policy_texts.items())}
    out = {"rows_checked": sum(len(load_rows(f)) for f in FILES), "contract_violations": violations,
           "policy_summary": policy_summary,
           "protocol": {"chars": len(PROTOCOL_INSTRUCTIONS), "words": len(PROTOCOL_INSTRUCTIONS.split())},
           "p2_hint": {"text": HINT, "chars": len(HINT), "words": len(HINT.split()),
                       "position": "appended after the protocol block = last text of the system prompt",
                       "chars_after_hint_to_generation_point": st(tail_after_hint_chars)},
           "cells": inventory}
    p = dump("prompt_inventory.json", out)
    print("rows checked", out["rows_checked"], "violations", len(violations))
    for k, v in policy_summary.items():
        print(k, v)
    print("wrote", rel(p))


if __name__ == "__main__":
    main()
