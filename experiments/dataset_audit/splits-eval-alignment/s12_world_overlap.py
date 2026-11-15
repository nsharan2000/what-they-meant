"""2 (continued). Semantic near-duplicates that token Jaccard on the request misses: same member world, same entities.

Variants and hard pairs are written for a 'target member' drawn from the environment's member pool, so a held-out item can be set in
a TRAINING member's world. For every held-out row (test, hard_test, val, hard_val) and every trained scenario with the SAME member,
compare the sets of salient entities: identifiers (emails, @handles, invoice/order ids, phone numbers, masked numbers), money amounts
and counterparty names found in the request, the canonical look-up results, scenario_records and the target. Report pairs with
entity-set Jaccard >= 0.35 or >= 3 shared identifiers, and whether the trained row's target already states the held-out decisive
identifier (gold / harmful commit signature).

python3 -B experiments/dataset_audit/splits-eval-alignment/s12_world_overlap.py -> results/s12_world_overlap.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from common import dump, eval_files, jaccard, scen_files, views
from wtm.render import render
from wtm.sim import Simulator
from wtm.spec import load_env, load_member

V = views()
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
trained = [by_id[x] for x in sorted({r["scenario_id"] for r in V["sft_train"]})]
held = []
for name, rows in eval_files().items():
    g = name.rsplit("_", 1)[0]
    if g in ("test", "hard_test", "val", "hard_val"):
        held += [(g, s) for s in rows]
IDENT = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w.@])@\w[\w.-]*\w|\b[A-Z]{2,5}-?\d{3,}[A-Z0-9-]*\b|\b\d{3}-\d{4}\b|"
                   r"\*{2,}\d{3,4}")
MONEY = re.compile(r"\d[\d,]*\.\d\d|\$\s?\d[\d,]*")
NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
envs, members = {}, {}


def target(s):
    a = s.get("ideal_action") or {}
    return a.get("text") or json.dumps(a.get("arguments", {}), ensure_ascii=False)


def entities(s):
    env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
    mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
    _, user, tools = render(env, mem, s, "L1", "P0")
    sim = Simulator(env, mem, s, tools, level="L1")
    res = " ".join(json.dumps(sim.execute(c["tool"], c["arguments"]), ensure_ascii=False) for c in s.get("prefix_calls") or [])
    blob = " ".join([user, res, json.dumps(s.get("scenario_records") or [], ensure_ascii=False), target(s)])
    ids = {m.group(0).lower() for m in IDENT.finditer(blob)}
    money = {m.group(0).replace(",", "").replace("$", "").strip() for m in MONEY.finditer(blob)}
    money = {x if "." in x else x + ".00" for x in money}
    names = {m.group(0) for m in NAME.finditer(blob)}
    return ids, money, names


def sigs(s, key):
    return [m.lower() for c in s.get(key) or [] for m in c.get("match", []) if re.search(r"[@\d]|\.", m) and len(m) >= 5]


tr_ent = {s["id"]: entities(s) for s in trained}
tr_by_member = defaultdict(list)
for s in trained:
    tr_by_member[(s["env_id"], s["member_id"])].append(s)
pairs = []
stat = Counter()
for g, h in held:
    hi, hm, hn = entities(h)
    hs = hi | hm | hn
    best = None
    for t in tr_by_member.get((h["env_id"], h["member_id"]), []):
        ti, tm, tn = tr_ent[t["id"]]
        ts = ti | tm | tn
        j = jaccard(hs, ts)
        shared_ids = sorted(hi & ti)
        if best is None or j > best["entity_jaccard"]:
            best = {"held_split": g, "held_id": h["id"], "train_id": t["id"], "entity_jaccard": round(j, 3),
                    "shared_identifiers": shared_ids, "shared_money": sorted(hm & tm)[:6], "shared_names": sorted(hn & tn)[:6],
                    "held_twin": h["twin"][0], "train_twin": t["twin"][0]}
    stat[f"{g}|rows"] += 1
    if best is None:
        continue
    stat[f"{g}|rows_with_same_member_trained_row"] += 1
    hit = best["entity_jaccard"] >= 0.35 or len(best["shared_identifiers"]) >= 3
    if hit:
        stat[f"{g}|rows_with_world_near_duplicate"] += 1
        decisive = set(sigs(h, "gold_commits") + sigs(h, "harmful_commits"))
        t_target = target(by_id[best["train_id"]]).lower()
        best["held_decisive_identifier_in_train_target"] = sorted(d for d in decisive if d in t_target)
        stat[f"{g}|near_dup_and_decisive_identifier_in_train_target"] += bool(best["held_decisive_identifier_in_train_target"])
        best["held_request"] = h["user_message"][:120]
        best["train_request"] = by_id[best["train_id"]]["user_message"][:120]
        pairs.append(best)
out = {"stats": dict(sorted(stat.items())), "pairs": sorted(pairs, key=lambda x: -x["entity_jaccard"])}
print(json.dumps(out["stats"], indent=1))
for p in out["pairs"][:30]:
    print(json.dumps({k: p[k] for k in ("held_split", "held_id", "train_id", "entity_jaccard", "shared_identifiers",
                                        "held_decisive_identifier_in_train_target", "held_request", "train_request")},
                     ensure_ascii=False))
print(dump("s12_world_overlap.json", out))
