"""Brittleness of the 'informed' keyword rule on the training targets themselves, plus the DPO rejected texts it calls generic.

1. Meaning-preserving surface perturbations of every twin-A decision target (SFT train + val, deduplicated by scenario and text),
   re-scored with wtm.reward.reward (= the evaluation's classify_decision + reward table):
     md_bold_numbers   **bold** every token containing a digit (common chat-model markdown: '**£300**')
     md_bold_caps      **bold** every capitalised word that is not sentence-initial ('**USDC**', '**Visa**')
     decimals_added    whole amounts get '.00' ('£300' -> '£300.00')
     decimals_dropped  '.00' removed ('30.00' -> '30')
     hyphen_to_space   'collect-balance' -> 'collect balance'
     space_to_hyphen   first two-word span of each sentence hyphenated is too arbitrary; instead 'extra usage' style:
                       every 'X Y' where 'X-Y' occurs in the keyword list is hyphenated (tests the reverse direction)
     expand_contractions  "doesn't" -> 'does not', "won't" -> 'will not', "isn't" -> 'is not', "can't" -> 'cannot'
     drop_hit_sentence the sentence that carries the FIRST keyword hit is removed (is there a second, independent naming?)
2. DPO rejected texts on twin-A rows that the grader classes ask_generic / statement: listed in full with the scenario keywords,
   so a reader can see whether any of them actually names the discrepancy (DPO would then push down a correct answer).
-> results/perturb_summary.json, results/dpo_rejected_A_generic.jsonl
"""
from __future__ import annotations

import json
import re
from collections import Counter

from ga_common import OUT, dump_json, dump_jsonl, kw_hit, load_rows, load_specs, reward, tool_kinds_of, trained_specs

CONTR = {"doesn't": "does not", "won't": "will not", "isn't": "is not", "can't": "cannot", "don't": "do not",
         "didn't": "did not", "aren't": "are not", "wasn't": "was not", "hasn't": "has not", "haven't": "have not"}


def md_bold_numbers(t):
    return re.sub(r"(?<![\w*])([£$€₹]?\d[\d,.:/-]*\d?%?)", r"**\1**", t)


def md_bold_caps(t):
    return re.sub(r"(?<=[a-z,;:] )([A-Z][A-Za-z0-9&.'-]*[A-Za-z0-9])", r"**\1**", t)


def decimals_added(t):
    return re.sub(r"([£$€₹]\d[\d,]*)(?![\d.,])", r"\1.00", t)


def decimals_dropped(t):
    return re.sub(r"(\d)\.00(?!\d)", r"\1", t)


def hyphen_to_space(t):
    return re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", t)


def expand_contractions(t):
    for a, b in CONTR.items():
        t = re.sub(re.escape(a), b, t, flags=re.I)
        t = t.replace(a.replace("'", "’"), b)
    return t


def space_to_hyphen_factory(keywords):
    pairs = [k.replace("-", " ") for k in keywords if re.search(r"[a-z]-[a-z]", k.lower())]

    def f(t):
        for p in pairs:
            t = re.sub(re.escape(p), p.replace(" ", "-"), t, flags=re.I)
        return t
    return f


def drop_hit_sentence_factory(keywords):
    def f(t):
        sents = re.split(r"(?<=[.?!])\s+", t)
        for i, s in enumerate(sents):
            if any(kw_hit(s, k) for k in keywords):
                rest = sents[:i] + sents[i + 1:]
                out = " ".join(rest)
                return out if "?" in out else (out + " Is that right?").strip()
        return t
    return f


def main():
    specs = load_specs()
    tspecs, _ = trained_specs(specs)
    seen, items = set(), []
    for f in ("sft_train", "sft_val"):
        for i, r in enumerate(load_rows(f)):
            if r["turn"] != "decision" or not r["twin"].startswith("A"):
                continue
            t = r["completion"][0]["content"]
            k = (r["scenario_id"], t)
            if k in seen:
                continue
            seen.add(k)
            items.append((f, i, r["scenario_id"], t))
    perturbs = {"md_bold_numbers": lambda s: md_bold_numbers, "md_bold_caps": lambda s: md_bold_caps,
                "decimals_added": lambda s: decimals_added, "decimals_dropped": lambda s: decimals_dropped,
                "hyphen_to_space": lambda s: hyphen_to_space, "expand_contractions": lambda s: expand_contractions,
                "space_to_hyphen": lambda s: space_to_hyphen_factory(s["expected"]["informed_any"]),
                "drop_hit_sentence": lambda s: drop_hit_sentence_factory(s["expected"]["informed_any"])}
    res = {p: Counter() for p in perturbs}
    examples = {p: [] for p in perturbs}
    for f, i, sid, t in items:
        scn = tspecs[sid]
        kinds = tool_kinds_of(scn["env_id"])
        base_r, base_p = reward(t, scn, kinds)
        for p, mk in perturbs.items():
            t2 = mk(scn)(t)
            if t2 == t:
                res[p]["unchanged_text"] += 1
                continue
            r2, p2 = reward(t2, scn, kinds)
            res[p]["changed_text"] += 1
            res[p][f"{base_p['decision']}->{p2['decision']}"] += 1
            if p2["decision"] != base_p["decision"] and len(examples[p]) < 5:
                examples[p].append({"file": f, "line": i, "scenario_id": sid, "before": t, "after": t2,
                                    "class_after": p2["decision"], "reward_after": r2,
                                    "hits_before": [k for k in scn["expected"]["informed_any"] if kw_hit(t, k)]})
    summ = {"n_unique_A_targets": len(items), "by_perturbation": {}, "examples": examples}
    for p, c in res.items():
        ch = c["changed_text"]
        flips = sum(v for k, v in c.items() if "->" in k and not k.endswith("->ask_informed"))
        summ["by_perturbation"][p] = {"texts_changed": ch, "flipped_out_of_ask_informed": flips,
                                      "flip_rate_among_changed": round(flips / ch, 4) if ch else None, "detail": dict(c)}
    # DPO rejected twin-A texts the grader calls generic / statement
    out = []
    for i, r in enumerate(load_rows("dpo_train")):
        if not r["twin"].startswith("A"):
            continue
        scn = tspecs[r["scenario_id"]]
        rej = r["rejected"][0]["content"]
        rr, rp = reward(rej, scn, tool_kinds_of(scn["env_id"]))
        if rp["decision"] in ("ask_generic", "statement", "tell_informed", "approve_informed"):
            out.append({"line": i, "scenario_id": r["scenario_id"], "level": r["level"], "prompt_setting": r["prompt_setting"],
                        "rejected_class": rp["decision"], "rejected_reward": rr, "rejected": rej,
                        "chosen": r["chosen"][0]["content"], "informed_any": scn["expected"]["informed_any"],
                        "discrepancy": scn["expected"].get("discrepancy")})
    dump_jsonl(out, f"{OUT}/dpo_rejected_A_generic.jsonl")
    summ["dpo_rejected_A_generic_or_statement"] = dict(Counter(o["rejected_class"] for o in out))
    dump_json(summ, f"{OUT}/perturb_summary.json")
    print(json.dumps(summ["by_perturbation"], indent=1))
    for p, exs in examples.items():
        for e in exs[:2]:
            print(f"--- {p} {e['scenario_id']} ({e['file']}:{e['line']}) -> {e['class_after']}\n  before: {e['before'][:220]}\n  after : {e['after'][:220]}\n  hits_before: {e['hits_before']}")
    print(summ["dpo_rejected_A_generic_or_statement"])


if __name__ == "__main__":
    main()
