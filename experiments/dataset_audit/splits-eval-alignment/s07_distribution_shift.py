"""6. Distribution shift between what is trained and what is evaluated, plus what separates the twins inside the look-up result.

(a) Composition: train (scenarios M1 trained on) vs val / test / hard_test / hard_train scenario files: environment, expected decision,
    target kind, sub_domain, tags, number of canonical look-ups, synthetic vs seed vs Claude-written hard pairs, member reuse.
(b) Levels / prompt settings: trained (rows per level x prompt setting) vs evaluated (run_eval.sh grid).
(c) Twin-separating keys: for pairs whose A/B decision prompts differ only in the look-up result, which JSON fields differ.
(d) M1 discrimination: for pair x level where the GRPO log sampled both twins, P(ask-class) on A vs on B.

python3 -B experiments/dataset_audit/splits-eval-alignment/s07_distribution_shift.py -> results/s07_distribution_shift.json
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from common import ROOT, dump, eval_files, jl, scen_files, splits, views

V = views()
sp = splits()
pair_split = {pid: k for k in ("train", "val", "test") for pid in sp[k]}
by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
trained = sorted({r["scenario_id"] for r in V["sft_train"]})
groups = {"train(trained)": [by_id[x] for x in trained]}
for name, rows in eval_files().items():
    groups.setdefault(name.rsplit("_", 1)[0], []).extend(rows)


def tkind(s):
    a = s.get("ideal_action") or {}
    if a.get("type") == "message":
        return "message"
    return "card" if a.get("name") == "request_approval" else "commit_tool"


def origin(s):
    return "hard(claude)" if s["id"].startswith("H") else ("variant(open-teacher)" if s.get("synthetic") else "seed")


comp = {}
for g, rows in groups.items():
    rows = [s for s in rows if not s["twin"].startswith("A_pushback")]
    n = len(rows)
    c = {"n": n,
         "env": dict(Counter(s["env_id"] for s in rows).most_common()),
         "expected": dict(Counter(f"{s['twin'][0]}:{s['expected']['decision']}" for s in rows).most_common()),
         "target_kind": dict(Counter(f"{s['twin'][0]}:{tkind(s)}" for s in rows).most_common()),
         "origin": dict(Counter(origin(s) for s in rows).most_common()),
         "n_prefix_calls": dict(Counter(len(s.get("prefix_calls") or []) for s in rows).most_common()),
         "has_noisy_request": sum(bool(s.get("user_message_noisy")) for s in rows),
         "has_l2_extra_records": sum(bool(s.get("l2_extra_records")) for s in rows),
         "has_scenario_records": sum(bool(s.get("scenario_records")) for s in rows),
         "sub_domain": dict(Counter(s.get("sub_domain") for s in rows).most_common()),
         "tags": dict(Counter(t for s in rows for t in s.get("tags", [])).most_common()),
         "hard_pattern": dict(Counter(s.get("hard_pattern") for s in rows if s.get("hard_pattern")).most_common())}
    comp[g] = c
train_tags = set(comp["train(trained)"]["tags"])
train_sub = set(comp["train(trained)"]["sub_domain"])
for g, c in comp.items():
    c["tags_never_trained"] = sorted(t for t in c["tags"] if t not in train_tags)
    c["rows_with_a_tag_never_trained"] = sum(any(t not in train_tags for t in s.get("tags", [])) for s in groups[g]
                                             if not s["twin"].startswith("A_pushback"))
    c["sub_domains_never_trained"] = sorted(t for t in c["sub_domain"] if t not in train_sub)
out = {"a_composition": comp}

# (b) trained vs evaluated grid
trained_grid = Counter(f"{r['level']}|{r['prompt_setting']}" for r in V["sft_train"] if r["turn"] == "decision")
out["b_trained_decision_rows_by_level_prompt"] = dict(sorted(trained_grid.items()))
out["b_evaluated_grid"] = {"levels": ["L0r", "L1", "L2", "L2s"], "prompts": ["P0", "P2"],
                           "source": "experiments/training_eval/run_eval.sh LEVELS/PROMPTS"}
out["b_levels_evaluated_never_trained"] = ["L2s"]
out["b_levels_trained_never_evaluated"] = ["L3"]

# (c) twin-separating keys inside look-up results
dec = defaultdict(dict)
for i, r in enumerate(V["sft_train"]):
    if r["turn"] == "decision" and not r["twin"].startswith("A_pushback"):
        dec[(r["pair_id"], r["level"])][r["twin"][0]] = (i, r)


def flat(x, pre=""):
    if isinstance(x, dict):
        o = {}
        for k, v in x.items():
            o.update(flat(v, f"{pre}.{k}" if pre else k))
        return o
    if isinstance(x, list):
        o = {}
        for j, v in enumerate(x):
            o.update(flat(v, f"{pre}[{j}]"))
        return o
    return {pre: x}


RES_RE = re.compile(r'<tool_result name="[^"]*">\n(.*)\n</tool_result>', re.S)
keys = Counter()
key_pairs = defaultdict(list)
n_pairs = 0
for (pid, lv), d in dec.items():
    if lv != "L1" or "A" not in d or "B" not in d:
        continue
    a, b = d["A"][1], d["B"][1]
    if len(a["prompt"]) != len(b["prompt"]) or len(a["prompt"]) < 4:
        continue
    n_pairs += 1
    for k in range(3, len(a["prompt"]), 2):
        ma, mb = RES_RE.search(a["prompt"][k]["content"]), RES_RE.search(b["prompt"][k]["content"])
        if not (ma and mb):
            continue
        try:
            ja, jb = json.loads(ma.group(1)), json.loads(mb.group(1))
        except json.JSONDecodeError:
            keys["<non-json result differs>"] += ma.group(1) != mb.group(1)
            continue
        fa, fb = flat(ja), flat(jb)
        diff = {re.sub(r"\[\d+\]", "[]", k_) for k_ in set(fa) | set(fb) if fa.get(k_) != fb.get(k_)}
        for k_ in diff:
            leaf = k_.split(".")[-1]
            keys[leaf] += 1
            if len(key_pairs[leaf]) < 3:
                key_pairs[leaf].append(pid)
out["c_L1_pairs_compared"] = n_pairs
out["c_differing_result_fields_top"] = dict(keys.most_common(30))
out["c_field_examples"] = {k: key_pairs[k] for k, _ in keys.most_common(30)}

# (d) does M1 separate the twins? P(ask-class) per prompt from the GRPO samples
ask = defaultdict(lambda: [0, 0])
for sc in jl(os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")):
    m = sc["meta"]
    d = sc["components"]["decision"]
    s = by_id[m["scenario_id"]]
    ask[(s["pair_id"], m["level"], m["twin"][0])][0] += d.startswith(("ask", "tell", "statement"))
    ask[(s["pair_id"], m["level"], m["twin"][0])][1] += 1
both = []
for (pid, lv, tw), (k, n) in ask.items():
    if tw == "A" and (pid, lv, "B") in ask:
        kb, nb = ask[(pid, lv, "B")]
        both.append({"pair_id": pid, "level": lv, "p_ask_A": round(k / n, 3), "p_ask_B": round(kb / nb, 3), "n_A": n, "n_B": nb})
out["d_pairs_with_both_twins_sampled"] = len(both)
if both:
    out["d_mean_p_ask_A"] = round(sum(x["p_ask_A"] for x in both) / len(both), 3)
    out["d_mean_p_ask_B"] = round(sum(x["p_ask_B"] for x in both) / len(both), 3)
    out["d_pairs_where_p_ask_B_ge_p_ask_A"] = sum(x["p_ask_B"] >= x["p_ask_A"] for x in both)
out["d_pairs"] = sorted(both, key=lambda x: x["p_ask_A"] - x["p_ask_B"])
# all sampled prompts: overall P(ask-class | A) and P(ask-class | B)
tot = {tw: [sum(v[0] for k_, v in ask.items() if k_[2] == tw), sum(v[1] for k_, v in ask.items() if k_[2] == tw)] for tw in "AB"}
out["d_overall_p_ask_class"] = {tw: round(a / n, 3) for tw, (a, n) in tot.items()}
print(json.dumps({g: {k: v for k, v in c.items() if k in ("n", "env", "expected", "target_kind", "origin", "tags_never_trained",
                                                          "rows_with_a_tag_never_trained", "sub_domains_never_trained",
                                                          "has_noisy_request", "has_l2_extra_records")}
                  for g, c in comp.items()}, indent=1)[:7000])
print(json.dumps({k: v for k, v in out.items() if k.startswith(("b_", "c_", "d_")) and k != "d_pairs"}, indent=1)[:4000])
print(dump("s07_distribution_shift.json", out))
