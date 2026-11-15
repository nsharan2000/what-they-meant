"""Step 2 (brief item 2): label-correlated cues OUTSIDE the decisive fact, on the trained decision rows.

  a) cue words ('flag', 'warning', 'unusual', 'mismatch', 'duplicate', 'verify', 'suspicious', 'note' + extras) counted per prompt
     component (member block of the system prompt / request / look-up calls / tool results), A vs B, per row and within matched
     twins (same pair, level; hint stripped);
  b) structural asymmetries of the last tool result: null leaves, empty lists, leaf count, characters, boolean values;
  c) single 'field=value' tokens of the tool results that are (almost) pure for one twin across the training set
     (e.g. payment_allowed=false) — verdict-like raw fields;
  d) prompt length (exact tokens) A vs B, within matched twins;
  e) member / profile and tool-list differences between twins (specs, every split);
  f) prompt setting (P2 hint) vs twin.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s2_cues.py  -> results/s2_cues.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402
import statistics as stt  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CUE_RX, RES, components, flatten, generic_path, jdump, load_rows, load_specs, load_split,  # noqa: E402
                    load_token_stats, member_block, parse_result, strip_hint, twin_letter)


def binom_two_sided(k, n):
    """exact two-sided sign-test p-value (p=0.5)."""
    from math import comb
    if n == 0:
        return 1.0
    pk = [comb(n, i) / 2 ** n for i in range(n + 1)]
    obs = pk[k]
    return min(1.0, sum(p for p in pk if p <= obs + 1e-12))


def leaf_tokens(obj) -> set:
    """field=value tokens for structural / short values (null, booleans, 0, empty containers, short strings)."""
    out = set()
    for p, v in flatten(obj).items():
        g = generic_path(p)
        if v is None:
            out.add(f"{g}=null")
        elif isinstance(v, bool):
            out.add(f"{g}={str(v).lower()}")
        elif v == [] or v == {}:
            out.add(f"{g}=empty")
        elif isinstance(v, (int, float)) and v == 0:
            out.add(f"{g}=0")
        elif isinstance(v, str) and len(v.split()) <= 3 and not any(ch.isdigit() for ch in v):
            out.add(f"{g}={v.lower()}")
        out.add(f"{g}=<present>")
    return out


def last_result_obj(prompt):
    comps = components(prompt)
    if not comps["results"]:
        return None, ""
    name, obj = parse_result(comps["results"][-1])
    return obj, comps["results"][-1]


def struct(obj):
    if not isinstance(obj, (dict, list)):
        return {"leaves": 0, "nulls": 0, "empties": 0, "falses": 0, "trues": 0, "zeros": 0}
    f = flatten(obj)
    return {"leaves": len(f), "nulls": sum(v is None for v in f.values()), "empties": sum(v in ([], {}) for v in f.values()),
            "falses": sum(v is False for v in f.values()), "trues": sum(v is True for v in f.values()),
            "zeros": sum((not isinstance(v, bool)) and isinstance(v, (int, float)) and v == 0 for v in f.values())}


def cue_counts(text):
    return {k: len(rx.findall(text)) for k, rx in CUE_RX.items()}


def main():
    specs, origin = load_specs()
    split = load_split()
    tstats = load_token_stats()
    out = {}
    for f in ("sft_train", "sft_val"):
        rows = [r for r in load_rows(f) if r["turn"] == "decision" and twin_letter(r["twin"]) != "P"]
        per = []
        for r in rows:
            comps = components(r["prompt"])
            obj, raw = last_result_obj(r["prompt"])
            rec = {"file": f, "i": r["_i"], "pair": r["pair_id"], "twin": twin_letter(r["twin"]), "level": r["level"],
                   "ps": r["prompt_setting"], "tokens": tstats[(f, r["_i"])]["prompt_tokens"],
                   "cue": {"member_block": cue_counts(member_block(comps["system"])), "request": cue_counts(comps["request"]),
                           "calls": cue_counts(" ".join(comps["calls"])), "results": cue_counts(" ".join(comps["results"]))},
                   "struct": struct(obj), "result_chars": len(raw), "n_lookups": len(comps["calls"]),
                   "leaf_tokens": sorted(leaf_tokens(obj)) if isinstance(obj, (dict, list)) else []}
            per.append(rec)
        # ---- a) cue words: row-level rates and within-pair differences
        cue_tab = {}
        by_key = defaultdict(dict)
        for p in per:
            by_key[(p["pair"], p["level"])][p["twin"]] = p
        matched = [(d["A"], d["B"]) for d in by_key.values() if "A" in d and "B" in d]
        for comp in ("member_block", "request", "calls", "results"):
            for cw in CUE_RX:
                a_rows = [p["cue"][comp][cw] for p in per if p["twin"] == "A"]
                b_rows = [p["cue"][comp][cw] for p in per if p["twin"] == "B"]
                if sum(a_rows) + sum(b_rows) == 0:
                    continue
                diffs = [a["cue"][comp][cw] - b["cue"][comp][cw] for a, b in matched]
                ag, bg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
                cue_tab[f"{comp}|{cw}"] = {
                    "rows_with_cue_A": sum(x > 0 for x in a_rows), "rows_with_cue_B": sum(x > 0 for x in b_rows),
                    "n_A": len(a_rows), "n_B": len(b_rows),
                    "pairs_A_more": ag, "pairs_B_more": bg, "pairs_equal": len(diffs) - ag - bg,
                    "sign_test_p": round(binom_two_sided(min(ag, bg), ag + bg), 5) if ag + bg else 1.0,
                    "examples_A_more": [[a["file"], a["i"], b["i"], a["pair"], a["level"]] for a, b in matched
                                        if a["cue"][comp][cw] > b["cue"][comp][cw]][:3],
                    "examples_B_more": [[a["file"], a["i"], b["i"], a["pair"], a["level"]] for a, b in matched
                                        if a["cue"][comp][cw] < b["cue"][comp][cw]][:3]}
        # ---- b) structure of the last tool result
        struct_tab = {}
        for k in ("leaves", "nulls", "empties", "falses", "trues", "zeros"):
            diffs = [a["struct"][k] - b["struct"][k] for a, b in matched]
            ag, bg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            struct_tab[k] = {"mean_A": round(stt.mean(p["struct"][k] for p in per if p["twin"] == "A"), 3),
                             "mean_B": round(stt.mean(p["struct"][k] for p in per if p["twin"] == "B"), 3),
                             "pairs_A_more": ag, "pairs_B_more": bg, "pairs_equal": len(diffs) - ag - bg,
                             "sign_test_p": round(binom_two_sided(min(ag, bg), ag + bg), 5) if ag + bg else 1.0}
        rc = [a["result_chars"] - b["result_chars"] for a, b in matched]
        struct_tab["result_chars"] = {"mean_A": round(stt.mean(p["result_chars"] for p in per if p["twin"] == "A"), 1),
                                      "mean_B": round(stt.mean(p["result_chars"] for p in per if p["twin"] == "B"), 1),
                                      "pairs_A_more": sum(d > 0 for d in rc), "pairs_B_more": sum(d < 0 for d in rc),
                                      "pairs_equal": sum(d == 0 for d in rc),
                                      "sign_test_p": round(binom_two_sided(min(sum(d > 0 for d in rc), sum(d < 0 for d in rc)),
                                                                           sum(d != 0 for d in rc)), 5)}
        # ---- c) near-pure field=value tokens (counted once per pair to avoid level repetition)
        tok_pairs = defaultdict(lambda: {"A": set(), "B": set()})
        tok_rows = defaultdict(lambda: {"A": [], "B": []})
        for p in per:
            for t in p["leaf_tokens"]:
                tok_pairs[t][p["twin"]].add(p["pair"])
                if len(tok_rows[t][p["twin"]]) < 3:
                    tok_rows[t][p["twin"]].append([p["file"], p["i"], p["pair"], p["level"]])
        pure = []
        for t, d in tok_pairs.items():
            na, nb = len(d["A"]), len(d["B"])
            if na + nb >= 3 and t.split("=", 1)[1] != "<present>" and max(na, nb) / (na + nb) >= 0.8:
                pure.append({"token": t, "pairs_A": na, "pairs_B": nb, "examples_A": tok_rows[t]["A"],
                             "examples_B": tok_rows[t]["B"]})
        pure.sort(key=lambda x: -(abs(x["pairs_A"] - x["pairs_B"])))
        # ---- d) prompt length (exact tokens) within matched twins, same prompt setting only (hint adds tokens)
        strict = [(a, b) for a, b in matched if a["ps"] == b["ps"]]
        td = [a["tokens"] - b["tokens"] for a, b in strict]
        len_tab = {"n_strict_pairs": len(td), "mean_A_minus_B": round(stt.mean(td), 2), "median": stt.median(td),
                   "A_longer": sum(x > 0 for x in td), "B_longer": sum(x < 0 for x in td), "equal": sum(x == 0 for x in td),
                   "sign_test_p": round(binom_two_sided(min(sum(x > 0 for x in td), sum(x < 0 for x in td)),
                                                        sum(x != 0 for x in td)), 5),
                   "row_mean_tokens_A": round(stt.mean(p["tokens"] for p in per if p["twin"] == "A"), 1),
                   "row_mean_tokens_B": round(stt.mean(p["tokens"] for p in per if p["twin"] == "B"), 1)}
        # ---- b2) verdict-like WORDS inside the raw tool results (English warnings in raw fields), any of the stems below
        verdict = ["flag", "warning", "unusual", "mismatch", "duplicate", "suspicious", "scam", "fraud", "risk", "alert", "caution",
                   "failed", "expired", "not", "unknown", "never"]
        def has_verdict(p):
            return any(p["cue"]["results"][k] > 0 for k in verdict)
        va = [p for p in per if p["twin"] == "A" and has_verdict(p)]
        vb = [p for p in per if p["twin"] == "B" and has_verdict(p)]
        vpairs_a = sorted({p["pair"] for p in va})
        vpairs_b = sorted({p["pair"] for p in vb})
        struct_tab["verdict_words_in_results"] = {
            "stems": verdict, "rows_A": len(va), "rows_B": len(vb), "pairs_A": len(vpairs_a), "pairs_B": len(vpairs_b),
            "pairs_A_only": sorted(set(vpairs_a) - set(vpairs_b)), "pairs_B_only": sorted(set(vpairs_b) - set(vpairs_a)),
            "examples_A": [[p["file"], p["i"], p["pair"], p["level"]] for p in va[:3]],
            "examples_B": [[p["file"], p["i"], p["pair"], p["level"]] for p in vb[:3]]}
        # ---- f) prompt setting vs twin
        ps = Counter((p["twin"], p["ps"]) for p in per)
        from math import comb
        a2, a0, b2, b0 = ps[("A", "P2")], ps[("A", "P0")], ps[("B", "P2")], ps[("B", "P0")]
        n_, k_, na_ = a2 + a0 + b2 + b0, a2 + b2, a2 + a0

        def hyp(x):
            return comb(na_, x) * comb(n_ - na_, k_ - x) / comb(n_, k_)
        obs = hyp(a2)
        fisher_p = sum(hyp(x) for x in range(max(0, k_ - (n_ - na_)), min(k_, na_) + 1) if hyp(x) <= obs * (1 + 1e-9))
        n_lookups = Counter((p["twin"], p["n_lookups"]) for p in per)
        out[f] = {"n_decision_rows": len(per), "n_matched_twins_same_level": len(matched),
                  "cue_words": dict(sorted(cue_tab.items(), key=lambda kv: -abs(kv[1]["pairs_A_more"] - kv[1]["pairs_B_more"]))),
                  "result_structure": struct_tab, "near_pure_field_value_tokens": pure[:40],
                  "prompt_length_exact_tokens": len_tab,
                  "prompt_setting_by_twin": {f"{k[0]}|{k[1]}": v for k, v in sorted(ps.items())},
                  "prompt_setting_vs_twin_fisher_two_sided_p": round(fisher_p, 4),
                  "n_lookups_by_twin": {f"{k[0]}|{k[1]}": v for k, v in sorted(n_lookups.items())}}

    # ---- e) member / tool list / request / prefix-call differences between twins, by split (specs)
    pairs = defaultdict(dict)
    for s in specs.values():
        pairs[s["pair_id"]][twin_letter(s["twin"])] = s
    spec_tab = defaultdict(Counter)
    spec_ex = defaultdict(list)
    for pid, d in sorted(pairs.items()):
        if "A" not in d or "B" not in d:
            continue
        a, b = d["A"], d["B"]
        sp = split.get(pid) or origin[a["id"]]
        spec_tab[sp]["pairs"] += 1
        checks = {"member_id_differs": a["member_id"] != b["member_id"],
                  "l0_member_block_differs": a.get("l0_member_block") != b.get("l0_member_block"),
                  "l0_tools_differ": set(a.get("l0_tools") or []) != set(b.get("l0_tools") or []),
                  "l0_tools_A_superset": set(a.get("l0_tools") or []) > set(b.get("l0_tools") or []),
                  "l0_tools_B_superset": set(b.get("l0_tools") or []) > set(a.get("l0_tools") or []),
                  "user_message_differs": a["user_message"] != b["user_message"],
                  "prefix_calls_differ": json.dumps(a.get("prefix_calls")) != json.dumps(b.get("prefix_calls")),
                  "n_prefix_calls_differ": len(a.get("prefix_calls") or []) != len(b.get("prefix_calls") or []),
                  "scenario_records_differ": json.dumps(a.get("scenario_records"), sort_keys=True)
                  != json.dumps(b.get("scenario_records"), sort_keys=True),
                  "l2_extra_records_differ": json.dumps(a.get("l2_extra_records"), sort_keys=True)
                  != json.dumps(b.get("l2_extra_records"), sort_keys=True)}
        for k, v in checks.items():
            if v:
                spec_tab[sp][k] += 1
                if len(spec_ex[(sp, k)]) < 6:
                    extra = ""
                    if k.startswith("l0_tools"):
                        extra = f" A-only={sorted(set(a['l0_tools']) - set(b['l0_tools']))} B-only={sorted(set(b['l0_tools']) - set(a['l0_tools']))}"
                    if k == "member_id_differs":
                        extra = f" {a['member_id']} vs {b['member_id']}"
                    spec_ex[(sp, k)].append(pid + extra)
    out["spec_twin_differences_by_split"] = {sp: dict(c) for sp, c in spec_tab.items()}
    out["spec_twin_difference_examples"] = {f"{k[0]}|{k[1]}": v for k, v in spec_ex.items()}
    jdump(out, os.path.join(RES, "s2_cues.json"))
    for f in ("sft_train", "sft_val"):
        o = out[f]
        print("=====", f, o["n_decision_rows"], o["n_matched_twins_same_level"])
        for k, v in list(o["cue_words"].items())[:14]:
            print(f"  {k:28s} A_rows={v['rows_with_cue_A']:3d} B_rows={v['rows_with_cue_B']:3d}  pairsA>B={v['pairs_A_more']:3d} "
                  f"B>A={v['pairs_B_more']:3d} eq={v['pairs_equal']:3d} p={v['sign_test_p']}")
        print("  structure", json.dumps(o["result_structure"]))
        print("  length", o["prompt_length_exact_tokens"])
        print("  ps", o["prompt_setting_by_twin"], "lookups", o["n_lookups_by_twin"])
        for t in o["near_pure_field_value_tokens"][:25]:
            print("   pure", t["token"], t["pairs_A"], t["pairs_B"])
    print(json.dumps(out["spec_twin_differences_by_split"], indent=0))
    print(json.dumps(out["spec_twin_difference_examples"], indent=0))


if __name__ == "__main__":
    main()
