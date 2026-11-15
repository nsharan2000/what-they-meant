"""1. Split integrity by source post, pair_id and scenario_id (splits.json, eval files, training rows) + identifier leakage.

python3 -B experiments/dataset_audit/splits-eval-alignment/s01_split_integrity.py
-> results/s01_split_integrity.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from common import ROOT, dump, eval_files, scen_files, splits, target_text, views

sp = splits()
pair_split = {}
dup_pairs = []
for k in ("train", "val", "test"):
    for pid in sp[k]:
        if pid in pair_split:
            dup_pairs.append((pid, pair_split[pid], k))
        pair_split[pid] = k
cluster_split = {c: v["split"] for c, v in sp["clusters"].items()}
cluster_of_pair_json = {p: c for c, v in sp["clusters"].items() for p in v["pairs"]}

seed, synth, hard = scen_files("seed"), scen_files("synth"), scen_files("hard")
by_id = {r["id"]: r for r in seed + synth + hard}
res = {"pairs_in_two_split_lists": dup_pairs, "n_pairs": {k: len(sp[k]) for k in ("train", "val", "test")},
       "n_clusters": dict(Counter(cluster_split.values()))}

# A. every seed/synth row: its post's cluster split == its pair's split; variant_of in the same split; twins share a post
viol_post, viol_variant, pair_posts = [], [], defaultdict(set)
for r in seed + synth:
    post = r["source"].get("post_id") or r["pair_id"]
    pair_posts[r["pair_id"]].add(post)
    ps, cs = pair_split.get(r["pair_id"]), cluster_split.get(post)
    if ps != cs:
        viol_post.append([r["id"], r["pair_id"], post, ps, cs])
    if r.get("variant_of") and pair_split.get(r["variant_of"]) != ps:
        viol_variant.append([r["id"], r["variant_of"], ps, pair_split.get(r["variant_of"])])
res["A_rows_checked"] = len(seed) + len(synth)
res["A_post_split_mismatch"] = viol_post
res["A_variant_of_split_mismatch"] = viol_variant
res["A_pairs_with_more_than_one_post"] = {p: sorted(v) for p, v in pair_posts.items() if len(v) > 1}
res["A_pairs_missing_from_splits"] = sorted({r["pair_id"] for r in seed + synth if r["pair_id"] not in pair_split})
res["A_cluster_pairs_vs_pair_split"] = [[p, c, cluster_split[c], pair_split.get(p)] for p, c in cluster_of_pair_json.items()
                                         if cluster_split[c] != pair_split.get(p)]

# B. training views: every row's pair in its view's split; scenario/post never in an eval file of another split
V = views()
ev = eval_files()
eval_ids = {name: {r["id"] for r in rows} for name, rows in ev.items()}
eval_pairs = {name: {r["pair_id"] for r in rows} for name, rows in ev.items()}
eval_posts = {name: {r["source"].get("post_id") for r in rows} for name, rows in ev.items()}
want = {"sft_train": "train", "sft_val": "val", "dpo_train": "train", "grpo_train": "train"}
view_checks = {}
for v, rows in V.items():
    bad_split = [[i, r["scenario_id"], r["pair_id"], pair_split.get(r["pair_id"])] for i, r in enumerate(rows)
                 if pair_split.get(r["pair_id"]) != want[v]]
    sid_mismatch = [[i, r["scenario_id"], r["pair_id"]] for i, r in enumerate(rows)
                    if by_id.get(r["scenario_id"], {}).get("pair_id") != r["pair_id"]]
    posts = Counter(by_id[r["scenario_id"]]["source"].get("post_id") for r in rows if r["scenario_id"] in by_id)
    overlaps = {}
    for name in ev:
        if v == "sft_val" and name.startswith("val_"):
            continue      # by design: sft_val = the val pairs (loss only)
        o_ids = sorted({r["scenario_id"] for r in rows} & eval_ids[name])
        o_pairs = sorted({r["pair_id"] for r in rows} & eval_pairs[name])
        o_posts = sorted(set(posts) & eval_posts[name])
        if o_ids or o_pairs or o_posts:
            overlaps[name] = {"scenario_ids": o_ids[:10], "pair_ids": o_pairs[:10], "posts": o_posts[:10], "n_posts": len(o_posts)}
    view_checks[v] = {"rows": len(rows), "rows_outside_split": bad_split[:10], "n_rows_outside_split": len(bad_split),
                      "scenario_pair_mismatch": sid_mismatch[:10], "distinct_scenarios": len({r["scenario_id"] for r in rows}),
                      "distinct_pairs": len({r["pair_id"] for r in rows}), "distinct_posts": len(posts),
                      "overlap_with_eval_files": overlaps}
res["B_training_views"] = view_checks

# C. eval files: pair split, held_split, eval_only, consistency with the current specs
eval_checks = {}
for name, rows in ev.items():
    kind = name.split("_")[0] if not name.startswith("hard_") else "hard_" + name.split("_")[1]
    split = kind.replace("hard_", "")
    bad = []
    for r in rows:
        if name.startswith("hard_"):
            post_split = cluster_split.get(r["source"].get("post_id"))
            vsplit = pair_split.get(r.get("variant_of"))
            if not (r.get("held_split") == split == post_split == vsplit) or not r.get("eval_only"):
                bad.append([r["id"], r.get("held_split"), post_split, vsplit, r.get("eval_only")])
        elif pair_split.get(r["pair_id"]) != split:
            bad.append([r["id"], pair_split.get(r["pair_id"]), split])
    spec_diff = []
    for r in rows:
        cur = by_id.get(r["id"])
        if cur is None:
            spec_diff.append([r["id"], "missing from current specs"])
            continue
        a = {k: v for k, v in r.items() if k != "held_split"}
        b = {k: v for k, v in cur.items() if k != "held_split"}
        if a != b:
            spec_diff.append([r["id"], sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))])
    eval_checks[name] = {"rows": len(rows), "split_violations": bad, "differs_from_current_spec": spec_diff}
res["C_eval_files"] = eval_checks
# every test/val spec row is in an eval file and vice versa
spec_test = {r["id"] for r in seed + synth if pair_split.get(r["pair_id"]) == "test"}
spec_val = {r["id"] for r in seed + synth if pair_split.get(r["pair_id"]) == "val"}
in_test_files = set().union(*[eval_ids[n] for n in ev if n.startswith("test_")])
in_val_files = set().union(*[eval_ids[n] for n in ev if n.startswith("val_")])
res["C_test_spec_rows_not_in_test_files"] = sorted(spec_test - in_test_files)
res["C_val_spec_rows_not_in_val_files"] = sorted(spec_val - in_val_files)
res["C_hard_rows_not_in_any_eval_file"] = sorted({r["id"] for r in hard} - set().union(*eval_ids.values()))
# hard-set rows never in training
train_sids = {r["scenario_id"] for v in V.values() for r in v}
res["C_hard_ids_in_training_views"] = sorted({r["id"] for r in hard} & train_sids)
res["C_hard_variant_of_counts"] = dict(Counter(pair_split.get(r.get("variant_of")) for r in hard if r["twin"].startswith("A")))

# D. data as trained: sha256 over the rows exactly as train_sft.py / train_grpo.py compute it
def rows_sha(rows):
    return hashlib.sha256(json.dumps([[r["prompt"], r["completion"]] for r in rows], sort_keys=True).encode()).hexdigest()


m1_args = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m1_sft/args.json")))
m2_args = json.load(open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/args.json")))
grpo_rows = V["grpo_train"]
res["D_as_trained"] = {
    "sft_train_sha256_matches_m1_args": rows_sha(V["sft_train"]) == m1_args["train_sha256"],
    "sft_val_sha256_matches_m1_args": rows_sha(V["sft_val"]) == m1_args["val_sha256"],
    "grpo_sha256_matches_m2_args": hashlib.sha256(json.dumps(grpo_rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    == m2_args["data_sha256"],
}

# E. identifier leakage: specific strings of held-out scenarios (test, hard_test, val) inside training prompts/targets
ID_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w.@])@\w[\w.-]*\w|\b[A-Z]{1,5}-?\d{3,}[A-Z0-9-]*\b|\b\d{3}-\d{4}\b")


def ident_strings(r):
    out = set()
    for s in (r.get("decisive_strings") or []):
        out.add(s)
    for key in ("gold_commits", "harmful_commits"):
        for c in r.get(key) or []:
            for m in c.get("match", []):
                if re.search(r"[@]|[a-z][0-9]|[0-9][a-z]|\.", m, re.I) and len(m) >= 6:
                    out.add(m)
    for m in ID_RE.finditer(json.dumps(r.get("tool_rules", []), ensure_ascii=False)):
        out.add(m.group(0))
    return {s for s in out if len(s) >= 6}


train_text_prompt = defaultdict(list)   # lower-cased text per (view, idx)
corpus_prompt, corpus_target = [], []
for v in ("sft_train", "dpo_train", "grpo_train"):
    for i, r in enumerate(V[v]):
        corpus_prompt.append((v, i, r["scenario_id"], "\n".join(m["content"] for m in r["prompt"]).lower()))
        if v != "grpo_train":
            corpus_target.append((v, i, r["scenario_id"], target_text(r).lower()))
held = [(n, r) for n, rows in ev.items() if not n.startswith("hard_train") for r in rows]
leak_target, leak_prompt = [], []
members_train = Counter((by_id[r["scenario_id"]]["env_id"], by_id[r["scenario_id"]]["member_id"]) for r in V["sft_train"])
for n, r in held:
    for s in sorted(ident_strings(r)):
        sl = s.lower()
        tgt_hits = [[v, i, sid] for v, i, sid, t in corpus_target if sl in t]
        pr_hits = [[v, i, sid] for v, i, sid, t in corpus_prompt if sl in t]
        if tgt_hits:
            leak_target.append({"eval_file": n, "id": r["id"], "string": s, "n_hits": len(tgt_hits), "examples": tgt_hits[:3]})
        if pr_hits:
            leak_prompt.append({"eval_file": n, "id": r["id"], "member": r["member_id"], "string": s, "n_hits": len(pr_hits),
                                "train_scenarios": sorted({h[2] for h in pr_hits})[:8], "examples": pr_hits[:3]})
res["E_identifier_strings_checked"] = sum(len(ident_strings(r)) for _, r in held)
res["E_heldout_identifiers_in_training_targets"] = leak_target
res["E_heldout_identifiers_in_training_prompts"] = leak_prompt
res["E_summary"] = {
    "n_target_leaks": len(leak_target), "n_prompt_leaks": len(leak_prompt),
    "prompt_leaks_by_eval_split": dict(Counter(x["eval_file"].rsplit("_", 1)[0] for x in leak_prompt)),
    "heldout_ids_with_prompt_leak": len({(x["eval_file"], x["id"]) for x in leak_prompt}),
}

# F. member profiles shared between training scenarios and held-out scenarios (the same base history appears in both)
mem_split = defaultdict(set)
for r in seed + synth:
    mem_split[(r["env_id"], r["member_id"])].add(pair_split.get(r["pair_id"]))
for r in hard:
    mem_split[(r["env_id"], r["member_id"])].add("hard_" + (r.get("held_split") or cluster_split.get(r["source"].get("post_id"), "?")))
shared = {f"{e}/{m}": sorted(s) for (e, m), s in mem_split.items() if "train" in s and len(s) > 1}
res["F_members_total"] = len(mem_split)
res["F_members_shared_train_and_heldout"] = shared
res["F_n_members_shared"] = len(shared)
res["F_heldout_rows_with_trained_member"] = {
    k: sum(1 for r in rows if (r["env_id"], r["member_id"]) in members_train) for k, rows in ev.items()}
res["F_heldout_rows_total"] = {k: len(rows) for k, rows in ev.items()}
print(json.dumps({k: (v if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 400 else f"<{len(v)} items>")
                  for k, v in res.items()}, indent=1, ensure_ascii=False, default=str))
print(dump("s01_split_integrity.json", res))
