"""V1b (verifier): what differs between twin system prompts at L0r (12 training pairs) and L1/L2 (S12, S54)?
Line-level diff of the hint-stripped system prompts of matched decision rows.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v1b_l0r_system_diff.py -> results/v1b_system_diffs.json
"""
import sys
sys.dont_write_bytecode = True
import difflib, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import T, curated, nohint, parts, save

sft = curated("sft_train")
m = defaultdict(dict)
for r in sft:
    if r["turn"] == "decision" and T(r) in "AB":
        m[(r["pair_id"], r["level"])][T(r)] = r
out = []
for (pid, lv), d in sorted(m.items()):
    if "A" not in d or "B" not in d:
        continue
    sa, sb = nohint(parts(d["A"]["prompt"])[0]), nohint(parts(d["B"]["prompt"])[0])
    if sa == sb:
        continue
    la, lb = sa.splitlines(), sb.splitlines()
    diff = [x for x in difflib.unified_diff(la, lb, lineterm="", n=0) if x[:1] in "+-" and not x.startswith(("+++", "---"))]
    out.append({"pair_id": pid, "level": lv, "A_line": d["A"]["_i"], "B_line": d["B"]["_i"], "n_diff_lines": len(diff),
                "chars_A": len(sa), "chars_B": len(sb), "diff": [x[:400] for x in diff[:12]]})
save(out, "v1b_system_diffs.json")
for o in out:
    print(o["pair_id"], o["level"], o["A_line"], o["B_line"], o["n_diff_lines"], o["chars_A"], o["chars_B"])
    for x in o["diff"][:8]:
        print("     ", x[:300])
