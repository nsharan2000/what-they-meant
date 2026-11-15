"""GA-10: DPO pairs whose rejected side differs from the chosen side only in format (same parsed action), and the classes of
the rows the analyst cites (dpo_train 138 / 61 / 84).
-> results/v_dpo_format_summary.json
"""
from __future__ import annotations

import json
import re

from vcommon import RW, dump, kinds, rows, spec_files
from wtm.protocol import parse_action

specs = spec_files()
d = rows("dpo_train")
same, ex, cited = 0, [], {}
for i, r in enumerate(d):
    s = specs[r["scenario_id"]]
    k = kinds(s["env_id"])
    c, j = r["chosen"][0]["content"], r["rejected"][0]["content"]
    pc, pj = parse_action(c, set(k)), parse_action(j, set(k))
    # rejected un-parseable: try to repair one brace and compare
    rep = None
    m = re.search(r"<tool_call>(.*)</tool_call>", j, re.S)
    if m:
        body = m.group(1).strip()
        for cand in (body[:-1], body + "}", body + "}}"):
            try:
                rep = json.loads(cand)
                break
            except Exception:  # noqa: BLE001
                pass
    same_action = (pc.kind == "tool_call" and rep is not None and rep.get("name") == pc.name and rep.get("arguments") == pc.arguments)
    if same_action:
        same += 1
        ex.append([i, r["scenario_id"], r["level"]])
    if i in (138, 61, 84):
        cited[i] = {"sid": r["scenario_id"], "chosen_kind": pc.kind, "rejected_kind": pj.kind, "rejected_flags": pj.flags,
                    "reward_chosen": RW.reward(c, s, k)[0], "reward_rejected": RW.reward(j, s, k),
                    "rejected_repaired_same_action_as_chosen": same_action,
                    "rejected_repaired_name": rep.get("name") if rep else None}
S = {"n": len(d), "format_only_pairs": same, "examples": ex[:10], "cited": cited}
dump(S, "v_dpo_format_summary.json")
print(json.dumps(S, indent=1))
