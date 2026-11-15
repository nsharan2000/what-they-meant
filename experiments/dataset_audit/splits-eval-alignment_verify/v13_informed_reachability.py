"""Can a policy that only READS its decision context produce an 'informed' question on held-out twin-A rows?

The grader calls a message informed when it hits any expected.informed_any keyword (wtm.spec.any_hit). For each twin-A row x
level (P0) we check whether at least one informed_any keyword occurs in the decision context the eval shows after the canonical
look-ups (system + request + look-up results). Rows where none occurs can only be passed by paraphrase luck (e.g. 'different
email' typed without it being in context is still possible: generic phrases are listed separately).
Also the same check on TRAINING twin-A decision rows (sft_train) for comparison.
Output: results/v13_informed_reachability.json
"""
import json
import re
from collections import Counter

from vcommon import curated, dump, episode_view, eval_sets, specs
from wtm.spec import kw_hit

GENERIC = re.compile(r"^(different|another|not the same|new|changed|wrong|mismatch|doesn't match|does not match|isn't|not your)", re.I)

ev = eval_sets()
out = {}
for name in ("test", "hard_test", "val"):
    c = Counter()
    miss = []
    for s in ev[name]:
        if not s["twin"].startswith("A_should"):
            continue
        kws = s["expected"].get("informed_any") or []
        specific = [k for k in kws if not GENERIC.search(k)]
        for lv in ("L0r", "L1", "L2"):
            system, user, tools, msgs = episode_view(s, lv, "P0")
            ctx = system + "\n" + user + "\n" + "\n".join(m["content"] for m in msgs[1:])
            any_kw = any(kw_hit(ctx, k) for k in kws)
            any_spec = any(kw_hit(ctx, k) for k in specific)
            c[(lv, "rows")] += 1
            c[(lv, "some_informed_keyword_in_context")] += any_kw
            c[(lv, "some_specific_keyword_in_context")] += any_spec
            if not any_spec and len(miss) < 12:
                miss.append([s["id"], lv, kws[:6]])
    out[name] = {"counts": {f"{a}|{b}": v for (a, b), v in sorted(c.items())}, "rows_without_specific_keyword": miss}

by_id = {s["id"]: s for s in specs()}
c = Counter()
for r in curated("sft_train"):
    if r["turn"] != "decision" or by_id[r["scenario_id"]]["twin"] != "A_should_ask":
        continue
    kws = by_id[r["scenario_id"]]["expected"].get("informed_any") or []
    ctx = "\n".join(m["content"] for m in r["prompt"])
    c[(r["level"], "rows")] += 1
    c[(r["level"], "some_informed_keyword_in_context")] += any(kw_hit(ctx, k) for k in kws)
out["sft_train"] = {f"{a}|{b}": v for (a, b), v in sorted(c.items())}
dump("v13_informed_reachability.json", out)
for k, v in out.items():
    print(k, v if k == "sft_train" else v["counts"], "" if k == "sft_train" else v["rows_without_specific_keyword"][:6])
