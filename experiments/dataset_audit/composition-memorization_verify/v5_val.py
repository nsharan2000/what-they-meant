"""CM-3 / CM-5: validation-set composition and wording novelty, re-derived independently.

- clusters (source posts) of val vs train pairs (dataset/splits.json clusters)
- val look-up tools vs tools trained as look-up targets
- word-4-gram coverage of decision targets: val vs train; train leave-one-cluster-out vs leave-one-pair-out
- seed vs variant target similarity (difflib on word lists) and request reuse
Output: results/v5_val.json, results/v5_variant_similarity.csv
"""
import difflib
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vcommon import ROOT, RES, cur, dump, target, request, is_call, call_name

splits = json.load(open(os.path.join(ROOT, "dataset/splits.json")))
pair2cl = {p: c for c, v in splits["clusters"].items() for p in v["pairs"]}
tr, va = cur("sft_train"), cur("sft_val")
out = {}
for name, rows in (("train", tr), ("val", va)):
    cl = Counter(pair2cl.get(r["pair_id"], "?") for r in rows)
    out[f"{name}_rows"] = len(rows)
    out[f"{name}_pairs"] = len({r["pair_id"] for r in rows})
    out[f"{name}_clusters"] = len(cl)
    out[f"{name}_cluster_rows"] = dict(cl.most_common())
    out[f"{name}_cluster_share_top2"] = [round(v / len(rows), 4) for _, v in cl.most_common(2)]
    out[f"{name}_rows_per_cluster_range"] = [min(cl.values()), max(cl.values())]
    pc = Counter(r["pair_id"] for r in rows)
    out[f"{name}_rows_per_pair_range"] = [min(pc.values()), max(pc.values())]
out["val_clusters_in_train"] = sorted(set(pair2cl[r["pair_id"]] for r in va) & set(pair2cl[r["pair_id"]] for r in tr))

# look-up tools
tr_look_tools = Counter(call_name(target(r)) for r in tr if r["turn"] != "decision")
va_look_tools = Counter(call_name(target(r)) for r in va if r["turn"] != "decision")
out["val_lookup_tools"] = dict(va_look_tools)
out["val_lookup_tools_not_trained_as_lookup_targets"] = {k: v for k, v in va_look_tools.items() if k not in tr_look_tools}
out["val_lookup_rows_untrained_tool"] = sum(v for k, v in va_look_tools.items() if k not in tr_look_tools)
out["val_lookup_rows"] = sum(va_look_tools.values())
# are those tools in the training system prompts' tool docs?
for k in out["val_lookup_tools_not_trained_as_lookup_targets"]:
    out[f"train_prompts_listing_{k}"] = sum(f"- {k}(" in r["prompt"][0]["content"] for r in tr)
    out[f"train_prompts_with_{k}_call_in_context"] = sum(any(f'"name": "{k}"' in m["content"] for m in r["prompt"][1:]) for r in tr)


def words(s):
    return re.findall(r"[a-z0-9$£€₹.@_'-]+", s.lower())


def grams(s, n=4):
    w = words(s)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def coverage(t, ref):
    g = grams(t)
    return (len(g & ref) / len(g)) if g else np.nan


dec_tr = [r for r in tr if r["turn"] == "decision"]
dec_va = [r for r in va if r["turn"] == "decision"]
ref_all = set().union(*[grams(target(r)) for r in dec_tr])
dt_va = {target(r): r for r in dec_va}
cov_val = [coverage(t, ref_all) for t in dt_va]
cov_val_msg = [coverage(t, ref_all) for t in dt_va if not is_call(t)]
out["val_4gram_cov_by_train_all"] = round(float(np.nanmean(cov_val)), 4)
out["val_4gram_cov_by_train_messages"] = round(float(np.nanmean(cov_val_msg)), 4)
# leave-one-cluster-out / leave-one-pair-out on training decision targets (distinct targets, each with its own pair/cluster)
dt_tr = {}
for r in dec_tr:
    dt_tr.setdefault(target(r), r)
loco, lopo, loco_m, lopo_m = [], [], [], []
for t, r in dt_tr.items():
    c, p = pair2cl[r["pair_id"]], r["pair_id"]
    ref_c = set().union(*[grams(x) for x, y in dt_tr.items() if pair2cl[y["pair_id"]] != c])
    ref_p = set().union(*[grams(x) for x, y in dt_tr.items() if y["pair_id"] != p])
    a, b = coverage(t, ref_c), coverage(t, ref_p)
    loco.append(a)
    lopo.append(b)
    if not is_call(t):
        loco_m.append(a)
        lopo_m.append(b)
out["train_4gram_cov_leave_one_cluster_out"] = round(float(np.nanmean(loco)), 4)
out["train_4gram_cov_leave_one_pair_out"] = round(float(np.nanmean(lopo)), 4)
out["train_4gram_cov_loco_messages"] = round(float(np.nanmean(loco_m)), 4)
out["train_4gram_cov_lopo_messages"] = round(float(np.nanmean(lopo_m)), 4)

# ---- seed vs variant similarity (CM-5) -------------------------------------------------------------------------------------
grpo = cur("grpo_train")
spec = {}
for g in grpo:
    s = json.loads(g["scenario"])
    spec[s["id"]] = s
tgt_by_scn = {}
for r in dec_tr:
    tgt_by_scn.setdefault(r["scenario_id"], target(r))
req_by_scn = {}
for r in dec_tr:
    if r["level"] != "L3":
        req_by_scn.setdefault(r["scenario_id"], request(r))


def sim(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


recs = []
for sid, t in tgt_by_scn.items():
    s = spec.get(sid)
    if not s or not s.get("variant_of"):
        continue
    seed = s["variant_of"]
    seed_id = seed if seed in tgt_by_scn else (seed + sid[-1] if seed + sid[-1] in tgt_by_scn else None)
    if seed_id is None:
        continue
    c = pair2cl[s["pair_id"]]
    others = [x for y, x in tgt_by_scn.items() if pair2cl[y[:-1]] != c and is_call(x) == is_call(t)]
    best_other = max((sim(t, x) for x in others), default=np.nan)
    recs.append({"scenario": sid, "seed": seed_id, "kind": "call" if is_call(t) else "message", "sim_to_seed": sim(t, tgt_by_scn[seed_id]),
                 "best_sim_other_cluster": best_other, "same_request_as_seed": req_by_scn.get(sid) == req_by_scn.get(seed_id),
                 "identical_target_to_seed": t == tgt_by_scn[seed_id]})
vs = pd.DataFrame(recs)
vs.to_csv(os.path.join(RES, "v5_variant_similarity.csv"), index=False)
out["variant_of_values_example"] = sorted({spec[s]["variant_of"] for s in spec if spec[s].get("variant_of")})[:5]
out["variants_with_seed_in_train"] = len(vs)
out["variant_sim_to_seed_mean"] = round(float(vs.sim_to_seed.mean()), 4)
out["variant_best_sim_other_cluster_mean"] = round(float(vs.best_sim_other_cluster.mean()), 4)
out["variant_sim_ge_0.6"] = round(float((vs.sim_to_seed >= 0.6).mean()), 4)
out["variant_sim_ge_0.8"] = round(float((vs.sim_to_seed >= 0.8).mean()), 4)
out["variant_sim_by_kind"] = {k: round(float(x.sim_to_seed.mean()), 4) for k, x in vs.groupby("kind")}
out["variants_identical_target_to_seed"] = vs[vs.identical_target_to_seed].scenario.tolist()
out["variants_same_request_as_seed"] = vs[vs.same_request_as_seed].scenario.tolist()
# cited examples
for a, b in (("V532A", "S53A"), ("V091A", "S09A"), ("V093A", "S09A"), ("V094A", "S09A"), ("V042A", "S04A")):
    if a in tgt_by_scn and b in tgt_by_scn:
        out[f"sim_{a}_{b}"] = round(sim(tgt_by_scn[a], tgt_by_scn[b]), 3)
out["req_S04A"] = req_by_scn.get("S04A")
out["req_V042A"] = req_by_scn.get("V042A")
out["tgt_S04A"] = tgt_by_scn.get("S04A")
out["tgt_V042A"] = tgt_by_scn.get("V042A")
# S11 vs V113: identical request and target -- how do their decision prompts differ?
def dec_rows(sid):
    return {r["level"]: r for r in dec_tr if r["scenario_id"] == sid}
for a, b in (("S11A", "V113A"), ("S11B", "V113B")):
    ra, rb = dec_rows(a), dec_rows(b)
    d = {}
    for lv in sorted(set(ra) & set(rb)):
        pa, pb = ra[lv]["prompt"], rb[lv]["prompt"]
        d[lv] = {"sys_equal": pa[0]["content"] == pb[0]["content"], "rest_equal": pa[1:] == pb[1:],
                 "ps": [ra[lv]["prompt_setting"], rb[lv]["prompt_setting"]],
                 "sys_sim": round(difflib.SequenceMatcher(None, pa[0]["content"], pb[0]["content"], autojunk=False).ratio(), 4)}
    out[f"{a}_vs_{b}"] = d
dump(out, "v5_val.json")
print(json.dumps(out, indent=1, ensure_ascii=False))
