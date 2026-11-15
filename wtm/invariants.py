"""Invariance checks for the realism ladder (research/realism-gap.md §3 "Invariants, checked by a program on every rendering").

For every scenario:
  I1 the request sentence is identical at L0/L0r/L0p/L1/L2/L2s (L3 may differ: noisy rewrite)
  I2 every decisive_strings entry appears in the context the model can see at every level (system prompt, or a tool-rule result)
  I3 budgets: L1 ≤ 5.5K and L2/L2s/L3 ≤ 7.2K estimated Aya tokens (chars / 3.0, conservative; JSON tokenises at ~2.7)
  I4 raw tool results (result_raw) contain no verdict words (they must state fields, not conclusions)
  I5 twin parity: A and B of a pair share env + member, so their L1 system prompts should be identical (reported, not fatal)
  I6 the seed's verdict-hint phrases are counted (L0-seed hint audit) so the "hint effect" can be reported
Usage: python3 -m wtm.invariants dataset/scenarios/*.jsonl  → prints a report, exit 1 on hard failures
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from .render import render
from .spec import load_env, load_jsonl, load_member, norm

VERDICT_RE = re.compile(r"\b(match(es|ed)?|mismatch|does(n't| not) match|suspicious|scam|fraud(ulent)?|warning|risky|risk|"
                        r"never paid|no photo|already paid|duplicate|legit(imate)?|verified|unverified|safe to|caution|"
                        r"red flag|impersonat\w*|fake|wrong|correct|ok to pay)\b", re.I)
EST = 3.0
BUDGET = {"L1": 5500, "L2": 7200, "L2s": 7200, "L3": 7200}


def check(paths: list[str]) -> dict:
    rows = [r for p in paths for r in load_jsonl(p)]
    envs, members = {}, {}
    hard, soft, hints = [], [], []
    by_pair = defaultdict(dict)
    sizes = defaultdict(list)
    for s in rows:
        env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
        mk = (s["env_id"], s["member_id"])
        mem = members.setdefault(mk, load_member(*mk))
        rendered = {}
        for lv in ["L0", "L0r", "L0p", "L1", "L2", "L2s", "L3"]:
            try:
                rendered[lv] = render(env, mem, s, lv, "P0")
            except Exception as e:  # noqa: BLE001
                hard.append(f"{s['id']} {lv}: render error {type(e).__name__}: {e}")
        if not rendered:
            continue
        # I1
        reqs = {lv: r[1] for lv, r in rendered.items() if lv != "L3"}
        if len(set(reqs.values())) > 1:
            hard.append(f"{s['id']}: request differs across levels {sorted(set(reqs.values()))[:2]}")
        # I2
        rule_text = " ".join(norm(json.dumps(r.get("result_raw", r.get("result")), ensure_ascii=False)) +
                             " " + norm(json.dumps(r.get("result"), ensure_ascii=False)) for r in s.get("tool_rules", []))
        for ds in s.get("decisive_strings", []):
            for lv, (sysp, user, _) in rendered.items():
                if norm(ds) not in norm(sysp) and norm(ds) not in norm(user) and norm(ds) not in rule_text:
                    hard.append(f"{s['id']} {lv}: decisive string not visible: {ds!r}")
        # I3
        for lv, (sysp, user, _) in rendered.items():
            est = (len(sysp) + len(user)) / EST
            sizes[lv].append(est)
            if lv in BUDGET and est > BUDGET[lv]:
                hard.append(f"{s['id']} {lv}: estimated {est:.0f} tokens > {BUDGET[lv]}")
        # I4 + I6
        for r in s.get("tool_rules", []):
            if "result_raw" in r:
                hit = VERDICT_RE.search(json.dumps(r["result_raw"], ensure_ascii=False))
                if hit:
                    soft.append(f"{s['id']}: verdict word {hit.group(0)!r} in result_raw of {r['tool']}")
            elif s["twin"] != "B_should_act" or True:
                soft.append(f"{s['id']}: rule for {r['tool']} has no result_raw")
            h = VERDICT_RE.search(json.dumps(r.get("result"), ensure_ascii=False))
            if h:
                hints.append(f"{s['id']}:{r['tool']}:{h.group(0)}")
        by_pair[s["pair_id"]][s["twin"][0]] = rendered.get("L1", ("", "", []))[0]
    # I5
    for pid, d in by_pair.items():
        if "A" in d and "B" in d and d["A"] != d["B"]:
            soft.append(f"{pid}: twin L1 system prompts differ (check that the difference is decision-relevant only)")
    return {"n_rows": len(rows), "hard_failures": hard, "soft_warnings": soft, "seed_hint_phrases": hints,
            "est_tokens_by_level": {lv: {"mean": round(sum(v) / len(v)), "max": round(max(v))} for lv, v in sizes.items()}}


if __name__ == "__main__":
    rep = check(sys.argv[1:])
    print(json.dumps({k: (v if not isinstance(v, list) else {"n": len(v), "first": v[:25]}) for k, v in rep.items()},
                     indent=1, ensure_ascii=False))
    sys.exit(1 if rep["hard_failures"] else 0)
