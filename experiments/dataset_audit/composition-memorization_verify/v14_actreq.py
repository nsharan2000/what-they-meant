"""For the act scenarios' pairs: is the twin-A request the same as the act (twin-B) request? (distinctive cue vs subtle cue)
Output: results/v14_actreq.json"""
import json
from vcommon import cur, dump, request
tr = cur("sft_train")
out = {}
for pid in ("S10", "S14", "S15", "V104", "V152"):
    reqs = {}
    for r in tr:
        if r["pair_id"] == pid and r["turn"] == "decision" and r["level"] in ("L1", "L0r"):
            reqs.setdefault(r["scenario_id"], request(r))
    out[pid] = reqs
    out[pid + "_twins_share_request"] = len(set(reqs.values())) == 1 and len(reqs) == 2
dump(out, "v14_actreq.json"); print(json.dumps(out, indent=1, ensure_ascii=False))
