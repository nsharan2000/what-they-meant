"""V10 (verifier): TCS-5's val-specific structure claims (B has more nulls in 36 vs 0 rows and more true values in 26 vs 0),
counted per matched (pair, level) val decision row and per val pair / family; plus the val request-difference share.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v10_val_structure.py -> results/v10_val_structure.json
"""
import sys
sys.dont_write_bytecode = True
import os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_twins import req_kind
from vcommon import T, curated, parts, result_obj, save, splits, struct_counts, sign_test_p

pair_split, cluster = splits()
va = [r for r in curated("sft_val") if r["turn"] == "decision" and T(r) in "AB"]
m = defaultdict(dict)
for r in va:
    m[(r["pair_id"], r["level"])][T(r)] = r
out = {"matched_rows": 0}
rows = {k: Counter() for k in ("null", "true", "false", "zero", "leaves")}
pairs = {k: defaultdict(list) for k in rows}
req = Counter()
for (pid, lv), d in m.items():
    if "A" not in d or "B" not in d:
        continue
    out["matched_rows"] += 1
    ca = struct_counts(result_obj(parts(d["A"]["prompt"])[3][-1])[0])
    cb = struct_counts(result_obj(parts(d["B"]["prompt"])[3][-1])[0])
    for k in rows:
        rows[k]["A>B" if ca[k] > cb[k] else ("B>A" if cb[k] > ca[k] else "tie")] += 1
        pairs[k][pid].append(ca[k] - cb[k])
    req[req_kind(parts(d["A"]["prompt"])[1], parts(d["B"]["prompt"])[1]) != "identical"] += 1
out["rows"] = {k: dict(v) for k, v in rows.items()}
out["pairs"] = {k: dict(Counter("A>B" if sum(v) > 0 else ("B>A" if sum(v) < 0 else "tie") for v in pv.values())) for k, pv in pairs.items()}
fam = {}
for k, pv in pairs.items():
    byc = defaultdict(list)
    for pid, v in pv.items():
        byc[cluster[pid]].append(sum(v) / len(v))
    c = Counter("A>B" if sum(v) > 0 else ("B>A" if sum(v) < 0 else "tie") for v in byc.values())
    fam[k] = dict(c) | {"sign_p": sign_test_p(c["A>B"], c["A>B"] + c["B>A"])}
out["families"] = fam
out["request_differs_rows_share"] = round(req[True] / sum(req.values()), 3)
save(out, "v10_val_structure.json")
print(out)
