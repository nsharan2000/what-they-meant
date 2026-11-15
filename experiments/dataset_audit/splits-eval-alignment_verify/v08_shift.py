"""SEA-8 verification: patterns in the held-out sets that training never shows.

(a) number of canonical look-ups per row (prefix_calls) in training scenarios vs each eval set; pairs whose twins first
    differ at look-up >= 2 (identical first look-up result);
(b) sub_domains and tags present in eval sets but absent from TRAINED scenarios;
(c) 'act' share among twin-B rows: training scenarios vs val/test/hard_test;
(d) level / prompt-setting mix: trained decision rows vs the eval grid (run_eval.sh: L0r L1 L2 L2s x P0 P2);
(e) look-up robustness: does the canonical look-up return the same result when its string arguments are lower-cased, or
    reduced to the first or last word (simple variants a policy might produce)?
Output: results/v08_shift.json
"""
import json
from collections import Counter

from vcommon import curated, dump, env, eval_sets, member, render, scenario_member, specs, Simulator

by_id = {s["id"]: s for s in specs()}
sft = curated("sft_train")
train_ids = sorted({r["scenario_id"] for r in sft})
train = [by_id[i] for i in train_ids]
ev = eval_sets()
sets = {"train": train, **ev}

A = {}
for name, rows in sets.items():
    n = Counter(len(s.get("prefix_calls") or []) for s in rows)
    A[name] = {"rows": len(rows), "lookups_per_row": dict(sorted(n.items())), "rows_with_2plus": sum(v for k, v in n.items() if k >= 2)}
# twins first differing at look-up >= 2
late = {}
for name in ("train", "test", "hard_test"):
    pairs = {}
    for s in sets[name]:
        if not s["twin"].startswith("A_pushback"):
            pairs.setdefault(s["pair_id"], {})[s["twin"][0]] = s
    lst = []
    for pid, t in pairs.items():
        if len(t) < 2:
            continue
        res = {}
        for k, s in t.items():
            e, m = env(s["env_id"]), member(s["env_id"], s["member_id"])
            _, _, tools = render(e, m, s, "L1", "P0")
            sim = Simulator(e, scenario_member(m, s, "L1"), s, tools, level="L1")
            res[k] = [json.dumps(sim.execute(c["tool"], c["arguments"]), sort_keys=True) for c in s.get("prefix_calls") or []]
            res[k + "u"] = s["user_message"]
        if res["Au"] == res["Bu"] and res["A"][:1] == res["B"][:1] and res["A"] != res["B"]:
            lst.append(pid)
    late[name] = sorted(lst)

trained_sub = {s.get("sub_domain") for s in train}
trained_tags = {t for s in train for t in s.get("tags", [])}
B = {}
for name in ("val", "test", "hard_test", "hard_val"):
    rows = ev[name]
    B[name] = {"untrained_sub_domain_rows": dict(Counter(s.get("sub_domain") for s in rows if s.get("sub_domain") not in trained_sub)),
               "untrained_tag_rows": sum(any(t not in trained_tags for t in s.get("tags", [])) for s in rows),
               "untrained_tags": dict(Counter(t for s in rows for t in s.get("tags", []) if t not in trained_tags))}

C = {}
for name, rows in sets.items():
    b = [s for s in rows if s["twin"].startswith("B")]
    C[name] = [sum(s["expected"]["decision"] == "act" for s in b), len(b)]

dec_rows = [r for r in sft if r["turn"] == "decision"]
D = {"trained_decision_rows_by_level": dict(Counter(r["level"] for r in dec_rows)),
     "trained_decision_rows_by_prompt": dict(Counter(r["prompt_setting"] for r in dec_rows)),
     "trained_P2_share": round(sum(r["prompt_setting"] == "P2" for r in dec_rows) / len(dec_rows), 3),
     "eval_grid": {"levels": ["L0r", "L1", "L2", "L2s"], "prompts": ["P0", "P2"], "P2_share": 0.5},
     "L3_in_training_rows": sum(r["level"] == "L3" for r in sft), "L2s_in_training_rows": sum(r["level"] == "L2s" for r in sft)}


def variants(args):
    out = []
    for k, v in args.items():
        if isinstance(v, str) and v.strip():
            w = v.split()
            for nv in {v.lower(), w[0], w[-1]}:
                if nv != v:
                    out.append({**args, k: nv})
    return out


E = {}
for name in ("train", "test", "hard_test"):
    c = Counter()
    ex = []
    for s in sets[name]:
        e, m = env(s["env_id"]), member(s["env_id"], s["member_id"])
        _, _, tools = render(e, m, s, "L1", "P0")
        for call in s.get("prefix_calls") or []:
            sim = Simulator(e, scenario_member(m, s, "L1"), s, tools, level="L1")
            base = json.dumps(sim.execute(call["tool"], call["arguments"]), sort_keys=True)
            for va in variants(call["arguments"]):
                sim2 = Simulator(e, scenario_member(m, s, "L1"), s, tools, level="L1")
                r2 = json.dumps(sim2.execute(call["tool"], va), sort_keys=True)
                c["variant_calls"] += 1
                c["same_result"] += r2 == base
                if r2 != base and len(ex) < 4:
                    ex.append([s["id"], call["tool"], va, r2[:120]])
    E[name] = {**dict(c), "share_same": round(c["same_result"] / max(1, c["variant_calls"]), 3), "examples_changed": ex}

dump("v08_shift.json", {"a_lookups": A, "a_twins_first_differ_at_lookup2plus": late, "b_untrained_strata": B,
                        "c_act_share_twinB": C, "d_level_prompt_mix": D, "e_lookup_variant_robustness": E})
print(json.dumps(A))
print(late)
print(json.dumps(B))
print(C)
print(D)
print({k: {kk: vv for kk, vv in v.items() if kk != "examples_changed"} for k, v in E.items()})
