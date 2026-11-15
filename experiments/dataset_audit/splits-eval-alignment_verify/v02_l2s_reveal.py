"""SEA-2 verification: does L2s ('## Most relevant records' = scenario.decisive_facts) hand over look-up outcomes?

For every held-out row: the canonical look-up results as the EVAL loop serves them (scenario_member + Simulator at L2s),
the L2 first prompt (system + request, no L2s block), and each decisive_facts line. A line 'reveals a look-up result' when it
contains an informative token (number with >= 2 digits, identifier-like token, or a word >= 5 letters outside a stoplist) that
occurs in the canonical look-up results AND NOT in the L2 first prompt (i.e. information that otherwise arrives only through
the look-up).  Also: twin-identical first prompts at L2 vs L2s, the block in training rows, base-model look-up behaviour.
Output: results/v02_l2s_reveal.json
"""
import json
import os
import re
from collections import Counter

from vcommon import ROOT, curated, dump, episode_view, eval_sets, render, env, member

STOP = set("""about above after again against because before being below between could during every their there these those
through under until where which while would should shall cannot member members payment payments account accounts amount
result results found records record""".split())
TOK = re.compile(r"[a-z0-9][a-z0-9@._-]*[a-z0-9]|[a-z0-9]")


def norm(t):
    t = (t or "").lower().replace(",", "").replace("’", "'")
    return re.sub(r"[$€£₹]", " ", t)


def informative(tokens):
    out = set()
    for t in tokens:
        if re.fullmatch(r"\d+(\.\d+)?", t):
            if len(t.replace(".", "")) >= 2:
                out.add(t)
        elif re.search(r"\d", t) or "@" in t or "-" in t or "_" in t:
            if len(t) >= 3:
                out.add(t)
        elif len(t) >= 5 and t not in STOP:
            out.add(t)
    return out


ev = eval_sets()
res = {}
examples = {}
for name in ("test", "hard_test", "val", "hard_val", "hard_train"):
    rows = ev[name]
    n_rows_reveal, n_lines, n_lines_reveal = 0, 0, 0
    per_row = []
    for s in rows:
        system_l2, user, tools, msgs = episode_view(s, "L2", "P0")
        results_txt = norm("\n".join(m["content"] for m in msgs[2::2]))
        ctx = norm(system_l2 + "\n" + user)
        res_toks, ctx_toks = set(TOK.findall(results_txt)), set(TOK.findall(ctx))
        revealing = []
        for f in s.get("decisive_facts") or []:
            n_lines += 1
            ft = informative(TOK.findall(norm(f)))
            only_lookup = sorted(t for t in ft if t in res_toks and t not in ctx_toks)
            if only_lookup:
                n_lines_reveal += 1
                revealing.append({"fact": f, "lookup_only_tokens": only_lookup[:8]})
        n_rows_reveal += bool(revealing)
        per_row.append({"id": s["id"], "n_facts": len(s.get("decisive_facts") or []), "n_revealing": len(revealing),
                        "revealing": revealing[:3]})
    # twin identity of FIRST prompts (system + request) at L2 and L2s, P0
    pairs = {}
    for s in rows:
        pairs.setdefault(s["pair_id"], {})[s["twin"][0]] = s
    ident = Counter()
    n_pairs = 0
    for pid, tw in pairs.items():
        if "A" not in tw or "B" not in tw:
            continue
        n_pairs += 1
        for lv in ("L0r", "L1", "L2", "L2s"):
            fa = render(env(tw["A"]["env_id"]), member(tw["A"]["env_id"], tw["A"]["member_id"]), tw["A"], lv, "P0")
            fb = render(env(tw["B"]["env_id"]), member(tw["B"]["env_id"], tw["B"]["member_id"]), tw["B"], lv, "P0")
            ident[lv] += (fa[0] == fb[0] and fa[1] == fb[1])
    res[name] = {"n_rows": len(rows), "rows_with_a_lookup_revealing_fact": n_rows_reveal,
                 "share": round(n_rows_reveal / max(1, len(rows)), 3), "n_fact_lines": n_lines,
                 "fact_lines_revealing": n_lines_reveal, "n_pairs": n_pairs,
                 "pairs_with_identical_first_prompt_P0": dict(ident)}
    examples[name] = [r for r in per_row if r["n_revealing"]][:6] + [r for r in per_row if not r["n_revealing"]][:3]

# named examples
named = {}
by = {s["id"]: s for rows in ev.values() for s in rows}
for sid in ("S19A", "S19B", "S43A", "S43B", "H291B"):
    s = by[sid]
    system, user, tools, msgs = episode_view(s, "L2s", "P0")
    block = system[system.find("## Most relevant records"):]
    named[sid] = {"l2s_block": block[:700], "lookup_results": [m["content"][:400] for m in msgs[2::2]]}

# training rows containing the L2s block
train_block = {v: sum("## Most relevant records" in r["prompt"][0]["content"] for r in curated(v))
               for v in ("sft_train", "sft_val", "dpo_train", "grpo_train")}

# base model look-up behaviour per level (eval_base_v12: test + hard_* rows)
base = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/eval_base_v12.jsonl"))]
beh = {}
for lv in ("L0r", "L1", "L2", "L2s"):
    rr = [r for r in base if r["level"] == lv]
    no_lookup = 0
    first_kind = Counter()
    succ = 0
    for r in rr:
        evs = r["events"]
        n_look = sum(1 for e in evs if e.get("tool_kind") in ("lookup", "verify"))
        no_lookup += n_look == 0
        first_kind[(evs[0].get("tool_kind") or evs[0].get("kind")) if evs else "none"] += 1
        succ += bool(r["decision_grade"]["success"])
    beh[lv] = {"n": len(rr), "episodes_without_any_lookup": no_lookup, "first_event_kind": dict(first_kind),
               "strict_success": succ}
# test-only subset
beh_test = {}
test_ids = {s["id"] for s in ev["test"]}
for lv in ("L2", "L2s"):
    rr = [r for r in base if r["level"] == lv and r["id"] in test_ids]
    beh_test[lv] = {"n": len(rr), "no_lookup": sum(1 for r in rr if not any(e.get("tool_kind") in ("lookup", "verify") for e in r["events"])),
                    "strict_success": sum(bool(r["decision_grade"]["success"]) for r in rr)}

dump("v02_l2s_reveal.json", {"per_split": res, "examples": examples, "named": named,
                             "training_rows_with_most_relevant_records_block": train_block,
                             "base_model_v12_behaviour_by_level(all 172 eval scenarios)": beh,
                             "base_model_v12_test_only": beh_test})
print(json.dumps(res, indent=0))
print(train_block)
print(json.dumps(beh, indent=0))
print(beh_test)
