"""2. Near-duplicate requests, targets and decision contexts across splits (normalised-token Jaccard).

Units = scenario specs (seed + synthetic variants + hard pairs), each labelled with its split: train / val / test (splits.json by
pair) and hard_train / hard_val / hard_test (held_split of the source cluster). Training-side units are restricted to the
scenario ids that actually occur in dataset/curated_v121/sft_train.jsonl (what M1 saw).
Fields: request = user_message (+ user_message_noisy as a second request string), target = ideal_action rendered exactly as
training/build_datasets.py does (message text or the <tool_call> JSON), context = request + the canonical look-up results
(wtm.sim.Simulator at L1, raw results), i.e. what sits in front of the decision apart from the system prompt.

python3 -B experiments/dataset_audit/splits-eval-alignment/s02_near_dups.py -> results/s02_near_dups.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from common import ROOT, dump, jaccard, scen_files, splits, toks, views
from wtm.render import render
from wtm.sim import Simulator
from wtm.spec import load_env, load_member

THR = 0.8
sp = splits()
pair_split = {pid: k for k in ("train", "val", "test") for pid in sp[k]}
cluster_split = {c: v["split"] for c, v in sp["clusters"].items()}
V = views()
trained_ids = {r["scenario_id"] for r in V["sft_train"]}
val_ids = {r["scenario_id"] for r in V["sft_val"]}


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def target_of(s):
    a = s.get("ideal_action") or {}
    if not a:
        return ""
    return a["text"] if a["type"] == "message" else call_text(a["name"], a.get("arguments", {}))


envs, members = {}, {}


def context_of(s):
    env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
    mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
    _, user, tools = render(env, mem, s, "L1", "P0")
    sim = Simulator(env, mem, s, tools, level="L1")
    parts = [user]
    for c in s.get("prefix_calls") or []:
        res = sim.execute(c["tool"], c["arguments"])
        parts.append(json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res)
    return "\n".join(parts)


units = []
for s in scen_files("seed") + scen_files("synth"):
    split = pair_split.get(s["pair_id"])
    if split == "train" and s["id"] not in trained_ids:
        continue     # not in what M1 trained on (e.g. a pushback duplicate or a dropped row)
    units.append((split, s))
for s in scen_files("hard"):
    units.append(("hard_" + cluster_split[s["source"]["post_id"]], s))

rows = []
for split, s in units:
    reqs = [s["user_message"]] + ([s["user_message_noisy"]] if s.get("user_message_noisy") else [])
    rows.append({"split": split, "id": s["id"], "pair_id": s["pair_id"], "twin": s["twin"], "env": s["env_id"],
                 "member": s["member_id"], "variant_of": s.get("variant_of"),
                 "req": [toks(x) for x in reqs], "req_text": reqs[0], "tgt": toks(target_of(s)), "tgt_text": target_of(s),
                 "ctx": toks(context_of(s))})

HELD = ("val", "test", "hard_val", "hard_test", "hard_train")
train_rows = [r for r in rows if r["split"] == "train"]
out = {"n_units": dict(Counter(r["split"] for r in rows)), "threshold": THR}
for field in ("req", "tgt", "ctx"):
    pairs, maxsim = [], defaultdict(list)
    for h in rows:
        if h["split"] not in HELD:
            continue
        best = (0.0, None)
        for t in train_rows:
            if field == "req":
                j = max(jaccard(a, b) for a in h["req"] for b in t["req"])
            else:
                j = jaccard(h[field], t[field])
            if j > best[0]:
                best = (j, t)
            if j >= THR:
                pairs.append({"held_split": h["split"], "held_id": h["id"], "train_id": t["id"], "jaccard": round(j, 3),
                              "same_member": h["member"] == t["member"], "same_twin_kind": h["twin"][0] == t["twin"][0],
                              "held_text": (h["req_text"] if field != "tgt" else h["tgt_text"])[:200],
                              "train_text": (t["req_text"] if field != "tgt" else t["tgt_text"])[:200]})
        maxsim[h["split"]].append(best[0])
    dist = {k: {"n": len(v), "mean_max_jaccard": round(sum(v) / len(v), 3), "n_ge_0.8": sum(x >= 0.8 for x in v),
                "n_ge_0.6": sum(x >= 0.6 for x in v), "n_ge_0.5": sum(x >= 0.5 for x in v)} for k, v in maxsim.items()}
    pairs.sort(key=lambda x: -x["jaccard"])
    out[field] = {"cross_split_pairs_ge_thr": len(pairs), "held_rows_with_a_train_near_dup": dict(Counter(
        p["held_split"] for p in {(p["held_split"], p["held_id"]): p for p in pairs}.values())),
        "max_similarity_to_train_by_split": dist, "pairs": pairs[:60],
        "pairs_label_flip": [p for p in pairs if not p["same_twin_kind"]][:30],
        "n_pairs_label_flip": sum(not p["same_twin_kind"] for p in pairs)}

# within-train: one request string, different twins and targets is by design (twins share the request). Report how often the
# SAME request text appears in train and a held-out split (exact, after token normalisation).
exact = []
for h in rows:
    if h["split"] in HELD:
        for t in train_rows:
            if any(a == b for a in h["req"] for b in t["req"]):
                exact.append([h["split"], h["id"], t["id"], h["req_text"][:120]])
out["exact_request_matches_heldout_vs_train"] = exact
# variants of held-out seeds that sit in training (by variant_of), and train seeds' variants in held-out files
out["variants_of_heldout_seeds_in_train"] = sorted({r["id"] for r in train_rows if r["variant_of"] and
                                                    pair_split.get(r["variant_of"]) != "train"})
print(json.dumps({k: (v if k in ("n_units", "threshold") else {kk: vv for kk, vv in v.items() if kk not in ("pairs", "pairs_label_flip")}
                      if isinstance(v, dict) else v) for k, v in out.items()}, indent=1, ensure_ascii=False)[:6000])
print(dump("s02_near_dups.json", out))
