"""SEA-7 + alignment re-checks + base-arm drift.

(a) eval loop vs training: every sft_train/sft_val DECISION row equals the eval's decision context (render + canonical look-ups
    served by the eval Simulator with scenario_member) for the same scenario/level/prompt setting;
(b) twins with byte-identical FIRST prompts per level (test, hard_test); pairs whose canonical FIRST look-up calls differ while
    their first prompts are identical (a deterministic policy cannot issue the canonical call for both twins);
(c) base arm (experiments/training_eval/results/eval_base_v12.jsonl): look-up behaviour (no look-up / first action = the
    canonical look-up), and DRIFT: does the recorded system_chars / first user message / n_tools still match today's rendering
    of the same scenario/level/prompt (the trained arms will be evaluated on today's dataset/eval files)?
(d) cluster structure of the held-out sets (source posts) -> effective sample size.
Output: results/v07_eval_alignment.json
"""
import json
import os
from collections import Counter, defaultdict

from vcommon import ROOT, curated, dump, env, episode_view, eval_sets, member, render, specs, splits, hard_specs

by_id = {s["id"]: s for s in specs()}
ev = eval_sets()
ev_by_id = {s["id"]: s for rows in ev.values() for s in rows}

# (a)
A = Counter()
mism = []
for view in ("sft_train", "sft_val"):
    for i, r in enumerate(curated(view)):
        if r["turn"] != "decision":
            continue
        s = by_id[r["scenario_id"]]
        system, user, tools, msgs = episode_view(s, r["level"], r["prompt_setting"])
        ok = r["prompt"][0]["content"] == system and r["prompt"][1:] == msgs
        A[view + "_decision_rows"] += 1
        A[view + "_identical_to_eval_context"] += ok
        if not ok and len(mism) < 5:
            mism.append([view, i, r["scenario_id"], r["level"]])

# (b)
B = {}
for name in ("test", "hard_test", "val"):
    pairs = defaultdict(dict)
    for s in ev[name]:
        if not s["twin"].startswith("A_pushback"):
            pairs[s["pair_id"]][s["twin"][0]] = s
    c = Counter()
    conflict = []
    for pid, t in sorted(pairs.items()):
        a, b = t["A"], t["B"]
        for lv in ("L0r", "L1", "L2"):
            ra = render(env(a["env_id"]), member(a["env_id"], a["member_id"]), a, lv, "P0")
            rb = render(env(b["env_id"]), member(b["env_id"], b["member_id"]), b, lv, "P0")
            same = ra[0] == rb[0] and ra[1] == rb[1]
            c[lv + "_identical_first_prompt"] += same
            ca = json.dumps((a.get("prefix_calls") or [{}])[0], sort_keys=True)
            cb = json.dumps((b.get("prefix_calls") or [{}])[0], sort_keys=True)
            if same and ca != cb:
                c[lv + "_identical_prompt_but_different_canonical_first_call"] += 1
                conflict.append([pid, lv, ca, cb])
        c["pairs"] += 1
    B[name] = {"counts": dict(c), "conflicts": conflict}

# (c)
base = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl"))]
C = Counter()
drift = []
canon_first = Counter()
for r in base:
    s = ev_by_id.get(r["id"])
    if s is None:
        C["base_rows_without_current_eval_row"] += 1
        continue
    system, user, tools = render(env(s["env_id"]), member(s["env_id"], s["member_id"]), s, r["level"], r["prompt"])
    first_user = r["messages"][0]["content"] if r.get("messages") else None
    same = (len(system) == r["system_chars"]) and (first_user == user) and (len(tools) == r["n_tools"])
    C["base_rows"] += 1
    C["base_rows_matching_current_render"] += same
    if not same:
        drift.append([r["id"], r["level"], r["prompt"], r["system_chars"], len(system), first_user == user])
    evs = r["events"]
    pc = (s.get("prefix_calls") or [{}])[0]
    if evs and evs[0].get("tool") == pc.get("tool") and evs[0].get("args") == pc.get("arguments"):
        canon_first[r["level"]] += 1
    if evs and evs[0].get("tool") == pc.get("tool"):
        canon_first[r["level"] + "_same_tool_any_args"] += 1
drift_ids = sorted({d[0] for d in drift})
drift_split = Counter(next((n for n, rows in ev.items() if any(x["id"] == i for x in rows)), "?") for i in drift_ids)

# (d)
sp = splits()
D = {}
for name in ("test", "hard_test", "val", "hard_val"):
    posts = Counter(s["source"]["post_id"] for s in ev[name])
    D[name] = {"rows": len(ev[name]), "pairs": len({s["pair_id"] for s in ev[name]}), "source_post_clusters": len(posts),
               "max_rows_per_cluster": max(posts.values())}
test_posts = {s["source"]["post_id"] for s in ev["test"]}
D["hard_test_rows_sharing_a_cluster_with_test"] = sum(s["source"]["post_id"] in test_posts for s in ev["hard_test"])

dump("v07_eval_alignment.json", {"a_train_vs_eval_context": dict(A), "a_mismatch_examples": mism, "b_first_prompts": B,
                                 "c_base_arm": dict(C), "c_base_first_action_canonical_lookup": dict(canon_first),
                                 "c_drift_rows(id,level,prompt,recorded_chars,current_chars,same_user)": drift[:40],
                                 "c_drift_ids": drift_ids, "c_drift_ids_by_split": dict(drift_split), "d_clusters": D})
print(dict(A), mism)
for k, v in B.items():
    print(k, v["counts"], v["conflicts"][:3])
print(dict(C), dict(canon_first))
print("drift ids", len(drift_ids), drift_ids[:30], dict(drift_split))
print(D)
