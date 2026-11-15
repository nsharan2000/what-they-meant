"""SEA-1 verification: split integrity, member-world sharing, and held-out decisive identifiers stated in TRAINING targets.

Independent re-derivation (own identifier regexes, own member accounting).
Output: results/v01_split_members.json
"""
from collections import Counter, defaultdict

from vcommon import ROOT, curated, dump, episode_view, eval_sets, hard_specs, identifiers, specs, splits, target_text

sp = splits()
pair_split = {}
dup_pairs = []
for k in ("train", "val", "test"):
    for p in sp[k]:
        if p in pair_split:
            dup_pairs.append(p)
        pair_split[p] = k
S = specs()
H = hard_specs()
by_id = {s["id"]: s for s in S + H}
ev = eval_sets()

# ---- A. split integrity -------------------------------------------------------------------------------------------
post_splits = defaultdict(set)
for s in S:
    post_splits[s["source"]["post_id"]].add(pair_split.get(s["pair_id"], "NONE"))
posts_multi = {p: sorted(v) for p, v in post_splits.items() if len(v) > 1}
missing_pairs = sorted({s["pair_id"] for s in S if s["pair_id"] not in pair_split})
cluster_split = {post: c["split"] for post, c in sp["clusters"].items()}
post_vs_cluster = sorted({s["id"] for s in S if cluster_split.get(s["source"]["post_id"]) != pair_split.get(s["pair_id"])})

train_views = {v: curated(v) for v in ("sft_train", "dpo_train", "grpo_train")}
val_view = curated("sft_val")
train_ids = sorted({r["scenario_id"] for rows in train_views.values() for r in rows})
val_ids = sorted({r["scenario_id"] for r in val_view})
heldout_ids = {name: sorted({r["id"] for r in rows}) for name, rows in ev.items()}
leak = {name: sorted(set(ids) & set(train_ids)) for name, ids in heldout_ids.items()}
train_split_of_ids = Counter(pair_split.get(by_id[i]["pair_id"], "NONE") for i in train_ids)
val_split_of_ids = Counter(pair_split.get(by_id[i]["pair_id"], "NONE") for i in val_ids)
# posts shared between training scenarios and held-out sets (hard_train shares posts with train by design)
train_posts = {by_id[i]["source"]["post_id"] for i in train_ids}
post_share = {name: sorted({r["id"] for r in rows if r["source"]["post_id"] in train_posts}) for name, rows in ev.items()}

A = {"dup_pairs_in_split_lists": dup_pairs, "spec_pairs_missing_from_splits": missing_pairs,
     "posts_spanning_splits": posts_multi, "rows_where_cluster_split_ne_pair_split": post_vs_cluster,
     "n_train_scenario_ids": len(train_ids), "train_ids_by_split": dict(train_split_of_ids),
     "n_val_scenario_ids": len(val_ids), "val_ids_by_split": dict(val_split_of_ids),
     "heldout_ids_in_training_views": leak,
     "heldout_rows_sharing_a_source_post_with_training": {k: len(v) for k, v in post_share.items()},
     "n_train_pairs_with_rows": len({by_id[i]["pair_id"] for i in train_ids})}

# ---- B. member worlds ---------------------------------------------------------------------------------------------
def mkey(s):
    return (s["env_id"], s["member_id"])


train_members = {mkey(by_id[i]) for i in train_ids}
B = {"n_members_used_by_training": len(train_members)}
all_used = set(train_members)
for name, rows in ev.items():
    mem = {mkey(r) for r in rows}
    all_used |= mem
    in_train = [r["id"] for r in rows if mkey(r) in train_members]
    B[name] = {"n_rows": len(rows), "n_members": len(mem), "members_also_in_training": len(mem & train_members),
               "rows_in_training_member": len(in_train), "share": round(len(in_train) / len(rows), 3)}
heldout_members = {mkey(r) for name in ("val", "test", "hard_val", "hard_test") for r in ev[name]}
test_members = {mkey(r) for name in ("test", "hard_test") for r in ev[name]}
B["n_members_used_anywhere(train+all eval files)"] = len(all_used)
B["n_members_used_by_train_or_val_test_hardval_hardtest"] = len(train_members | heldout_members)
B["members_in_training_AND_in_val/test/hard_val/hard_test"] = len(train_members & heldout_members)
B["members_in_training_AND_in_test/hard_test"] = len(train_members & test_members)

# ---- C. held-out decisive identifiers found in TRAINING TARGETS -----------------------------------------------------
sft = curated("sft_train")
tgt_index = defaultdict(list)          # identifier (lower) -> [(line, scenario_id, level, turn)]
for i, r in enumerate(sft):
    for kind, v in identifiers(target_text(r)):
        tgt_index[v.lower()].append((i, r["scenario_id"], r["level"], r["turn"], kind))


def decisive_ids(s):
    txt = []
    ia = s.get("ideal_action") or {}
    txt.append(ia.get("text", "") if ia.get("type") == "message" else str(ia.get("arguments", "")))
    for g in s.get("gold_commits") or []:
        txt.extend(g.get("match", []))
    txt.extend(s.get("decisive_strings") or [])
    txt.extend(s["expected"].get("informed_any") or [])
    return {v for _, v in identifiers(" \n ".join(map(str, txt)))}


C = {}
for name in ("test", "hard_test", "val", "hard_val"):
    hits = []
    for s in ev[name]:
        for ident in sorted(decisive_ids(s)):
            rows = [x for x in tgt_index.get(ident.lower(), []) if x[1] != s["id"]]
            if not rows:
                continue
            # is the identifier visible in the held-out row's own decision context?
            vis = {}
            for lv in ("L1", "L2"):
                system, user, tools, msgs = episode_view(s, lv, "P0")
                res = "\n".join(m["content"] for m in msgs[2::2])
                vis[lv] = {"in_system": ident.lower() in system.lower(), "in_request": ident.lower() in user.lower(),
                           "in_lookup_results": ident.lower() in res.lower()}
            hits.append({"heldout_id": s["id"], "member": s["member_id"], "identifier": ident,
                         "n_train_target_rows": len(rows), "train_rows(line,scenario,level,turn,kind)": rows[:6],
                         "train_scenarios": sorted({x[1] for x in rows}), "visible_in_own_context": vis})
    C[name] = {"n_hits": len(hits), "n_heldout_rows_hit": len({h["heldout_id"] for h in hits}),
               "rows_hit": sorted({h["heldout_id"] for h in hits}), "hits": hits}

# ---- D. the analyst's named examples, verbatim -------------------------------------------------------------------
def tgt_rows(sid, turn="decision"):
    return [(i, r["level"], r["prompt_setting"], target_text(r)) for i, r in enumerate(sft) if r["scenario_id"] == sid and r["turn"] == turn]


D = {}
for sid in ("H231A", "H231B", "V492A", "V492B", "H201A", "H201B", "S06A", "V232A", "V232B", "S49B", "S01A", "V474A", "V082A"):
    s = by_id.get(sid)
    D[sid] = {"split": s.get("held_split") or pair_split.get(s["pair_id"]), "member": s["member_id"],
              "user_message": s["user_message"], "ideal": (s.get("ideal_action") or {}).get("text") or (s.get("ideal_action") or {}).get("arguments"),
              "gold": s.get("gold_commits"), "sft_train_decision_rows": tgt_rows(sid)[:4]}

dump("v01_split_members.json", {"A_split_integrity": A, "B_member_worlds": B, "C_heldout_ids_in_training_targets": C,
                                "D_examples": D})
print({k: v for k, v in A.items() if k not in ("posts_spanning_splits",)})
print(B)
for name in C:
    print(name, C[name]["n_hits"], C[name]["rows_hit"])
