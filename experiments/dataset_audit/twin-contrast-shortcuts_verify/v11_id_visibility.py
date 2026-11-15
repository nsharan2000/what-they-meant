"""V11 (verifier): for identifier-only twin pairs (the request differs only in an identifier), is the resolving evidence visible
in the A twin's decision prompt OUTSIDE the request at every level? Checks whether B's identifier (the 'right' one, which the
history usually holds) and A's identifier occur in A's system prompt + tool results. If neither occurs, the A/B contrast at that
level reduces to an arbitrary identifier in the request (no grounding), i.e. the row is unlearnable as intended.
Also: the negative-literal count heuristic vs M1 ask rate (analyst: A 0.104, B 0.183, n.s.).
python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v11_id_visibility.py -> results/v11_id_visibility.json
"""
import sys
sys.dont_write_bytecode = True
import os, re
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_twins import ID_RX, req_kind
from v8_fields_context import spear, perm_p
from vcommon import ROOT, T, curated, jl, leaves, parts, result_obj, save

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def loose_hit(tok, ctx):
    """identifier visible in an exact OR near-identical form: amounts by their numeric value (3,200 ~ 3200.0), ids with
    punctuation removed (INV8831 ~ INV-8831)."""
    t = tok.lower().replace(",", "")
    c = ctx.lower().replace(",", "")
    num = re.fullmatch(r"[$€£₹]?(\d+)(?:\.(\d+))?", t)
    if num:
        return re.search(r"(?<![\d.])" + num.group(1) + r"(?:\.0+)?(?![\d])", c) is not None
    return re.sub(r"[^a-z0-9@.]", "", t).strip(".") in re.sub(r"[^a-z0-9@.]", "", c)

out = {"rows": []}
for name in ("sft_train", "sft_val"):
    rows = [r for r in curated(name) if r["turn"] == "decision" and T(r) in "AB"]
    m = defaultdict(dict)
    for r in rows:
        m[(r["pair_id"], r["level"])][T(r)] = r
    for (pid, lv), d in sorted(m.items()):
        if "A" not in d or "B" not in d:
            continue
        ra, rb = parts(d["A"]["prompt"])[1], parts(d["B"]["prompt"])[1]
        if req_kind(ra, rb) != "identifier_only":
            continue
        ia = [x.strip(".,") for x in ID_RX.findall(ra)]
        ib = [x.strip(".,") for x in ID_RX.findall(rb)]
        da = [x for x in ia if x not in ib]
        db = [x for x in ib if x not in ia]
        ctx = "\n".join([d["A"]["prompt"][0]["content"]] + [m_["content"] for m_ in d["A"]["prompt"][2:]])
        nctx = norm(ctx)
        out["rows"].append({"file": name, "pair": pid, "level": lv, "A_ids": da, "B_ids": db,
                            "B_id_in_A_context": any(norm(x) in nctx for x in db),
                            "A_id_in_A_context": any(norm(x) in nctx for x in da),
                            "B_id_loose": any(loose_hit(x, ctx) for x in db), "A_id_loose": any(loose_hit(x, ctx) for x in da)})
vis = [r for r in out["rows"]]
out["n"] = len(vis)
out["neither_visible"] = [r for r in vis if not r["B_id_in_A_context"] and not r["A_id_in_A_context"]]
out["B_id_visible"] = sum(r["B_id_in_A_context"] for r in vis)
out["A_id_visible_only"] = sum(r["A_id_in_A_context"] and not r["B_id_in_A_context"] for r in vis)
out["neither_visible_loose"] = [r for r in vis if not r["B_id_loose"] and not r["A_id_loose"]]

# negative-literal heuristic vs M1 ask rate
sc = jl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl"))
MSG = {"ask_informed", "ask_generic", "tell_informed", "statement"}
per = defaultdict(list)
for s in sc:
    per[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"] in MSG)
grpo = {(r["scenario_id"], r["level"]): r for r in curated("grpo_train")}
neg = {}
for t in "AB":
    xs, ys = [], []
    for k, v in per.items():
        g = grpo[k]
        if T(g) != t:
            continue
        res = parts(g["prompt"])[3]
        obj = result_obj(res[-1])[0] if res else None
        vals = [v_ for _, v_ in leaves(obj)] if isinstance(obj, (dict, list)) else []
        xs.append(sum(v_ is None or v_ is False or v_ in ([], {}) or (not isinstance(v_, bool) and isinstance(v_, (int, float)) and v_ == 0) for v_ in vals))
        ys.append(float(np.mean(v)))
    neg[t] = {"n": len(xs), "spearman": round(spear(xs, ys), 3), "perm_p": round(perm_p(xs, ys), 4)}
out["m1_ask_vs_negative_literals"] = neg
save(out, "v11_id_visibility.json")
print({k: v for k, v in out.items() if k != "rows"})
for r in out["rows"]:
    print(r)
