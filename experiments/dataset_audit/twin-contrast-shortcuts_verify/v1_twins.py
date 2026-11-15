"""V1 (verifier): twin matching, request differences, prompt-setting draw, system-prompt identity, result structure and length
asymmetries in the TRAINING views, counted both per matched (pair, level) decision row pair (the analyst's unit) and per pair
(the independent unit: levels of one pair share the same request and, mostly, the same look-up result).

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v1_twins.py -> results/v1_twins.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import os  # noqa: E402
import re  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import (T, all_specs, curated, eval_specs, fisher_two_sided, nohint, parts, result_obj, save,  # noqa: E402
                     sign_test_p, splits, struct_counts, token_stats, words)

ID_RX = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|@[\w.-]+|[$€£₹]?\d[\d,.:/\-]*\w*|\b\w*\d\w*\b|\b[\w-]+\.(?:com|net|org|io|in|co|uk|app|dev|store|pro)\b")


def masked(t):
    return " ".join(words(ID_RX.sub(" IDTOKEN ", t)))


def req_kind(a, b):
    if a.strip() == b.strip():
        return "identical"
    if masked(a) == masked(b):
        return "identifier_only"
    return "wording"


def main():
    out = {}
    pair_split, cluster = splits()
    specs = all_specs()
    ts = token_stats()
    sft = curated("sft_train")
    dec = [r for r in sft if r["turn"] == "decision"]
    out["sft_train_rows"] = len(sft)
    out["sft_train_decision_rows"] = len(dec)
    out["decision_rows_by_twin"] = dict(Counter(T(r) for r in dec))
    m = defaultdict(dict)
    for r in dec:
        if T(r) == "P":
            continue
        m[(r["pair_id"], r["level"])].setdefault(T(r), []).append(r)
    dup = {k: {t: len(v) for t, v in d.items()} for k, d in m.items() if any(len(v) > 1 for v in d.values())}
    matched = {k: (d["A"][0], d["B"][0]) for k, d in m.items() if "A" in d and "B" in d}
    unmatched = sorted([(k, sorted(d)) for k, d in m.items() if not ("A" in d and "B" in d)])
    out["matched_decision_pairs"] = len(matched)
    out["duplicate_keys"] = dup
    out["unmatched"] = unmatched
    pairs_both = sorted({k[0] for k in matched})
    out["n_train_pairs_with_both_twins_matched"] = len(pairs_both)
    out["n_train_pairs_in_split"] = sum(1 for p, s in pair_split.items() if s == "train")

    # ------------------------------------------------------------------ request differences
    rk_rows = Counter()
    rk_rows_by_level = defaultdict(Counter)
    rk_pair = {}
    for (pid, lv), (a, b) in matched.items():
        ra, rb = parts(a["prompt"])[1], parts(b["prompt"])[1]
        k = req_kind(ra, rb)
        rk_rows[k] += 1
        rk_rows_by_level[lv][k] += 1
    # pair level from the specs' clean user_message (independent of L3 noisy rewrites)
    by_pair = defaultdict(dict)
    for s in specs.values():
        if T(s) in "AB":
            by_pair[s["pair_id"]][T(s)] = s
    for pid, d in by_pair.items():
        if "A" in d and "B" in d:
            rk_pair[pid] = req_kind(d["A"]["user_message"], d["B"]["user_message"])
    out["request_kind_matched_decision_rows"] = dict(rk_rows)
    out["request_kind_matched_decision_rows_by_level"] = {k: dict(v) for k, v in rk_rows_by_level.items()}
    out["request_kind_train_pairs"] = dict(Counter(rk_pair[p] for p in pairs_both))
    out["request_differs_train_pairs"] = sum(rk_pair[p] != "identical" for p in pairs_both)
    # by split for eval files (spec-level)
    for name in ("test", "val", "hard_test", "hard_val", "hard_train"):
        bp = defaultdict(dict)
        for s in eval_specs(name):
            if T(s) in "AB":
                bp[s["pair_id"]][T(s)] = s
        kinds = Counter(req_kind(d["A"]["user_message"], d["B"]["user_message"]) for d in bp.values() if "A" in d and "B" in d)
        out[f"request_kind_{name}_pairs"] = dict(kinds) | {"n_pairs": sum(kinds.values())}
    out["train_pairs_request_listing"] = [
        {"pair_id": p, "kind": rk_pair[p], "cluster": cluster.get(p), "A": by_pair[p]["A"]["user_message"],
         "B": by_pair[p]["B"]["user_message"]} for p in pairs_both if rk_pair[p] != "identical"]

    # ------------------------------------------------------------------ cue words in the request
    cues = {"new": r"\bnew\b", "first": r"\bfirst\b", "change": r"\bchang(?:e|ed|es)\b"}
    cue_tab = {}
    for cname, rx in cues.items():
        rx = re.compile(rx, re.I)
        rows_a_gt = rows_b_gt = 0
        pairs_a_gt, pairs_b_gt = set(), set()
        for (pid, lv), (a, b) in matched.items():
            na, nb = len(rx.findall(parts(a["prompt"])[1])), len(rx.findall(parts(b["prompt"])[1]))
            if na > nb:
                rows_a_gt += 1
                pairs_a_gt.add(pid)
            elif nb > na:
                rows_b_gt += 1
                pairs_b_gt.add(pid)
        cue_tab[cname] = {"rows_A_more": rows_a_gt, "rows_B_more": rows_b_gt, "pairs_A_more": sorted(pairs_a_gt),
                          "pairs_B_more": sorted(pairs_b_gt),
                          "clusters_A_more": sorted({cluster.get(p) for p in pairs_a_gt}),
                          "clusters_B_more": sorted({cluster.get(p) for p in pairs_b_gt})}
    out["cue_words_request"] = cue_tab

    # ------------------------------------------------------------------ prompt setting
    same = sum(a["prompt_setting"] == b["prompt_setting"] for a, b in matched.values())
    out["same_prompt_setting_share"] = round(same / len(matched), 3)
    out["same_prompt_setting_n"] = same
    p2a = sum(1 for r in dec if T(r) == "A" and r["prompt_setting"] == "P2")
    p2b = sum(1 for r in dec if T(r) == "B" and r["prompt_setting"] == "P2")
    na = sum(1 for r in dec if T(r) == "A")
    nb = sum(1 for r in dec if T(r) == "B")
    out["p2_decision_rows"] = {"A": p2a, "B": p2b, "A_total": na, "B_total": nb,
                               "fisher_two_sided_p": round(fisher_two_sided(p2a, na - p2a, p2b, nb - p2b), 4)}

    # ------------------------------------------------------------------ system prompt identity (hint stripped)
    sys_same = defaultdict(Counter)
    sys_diff_pairs = defaultdict(set)
    for (pid, lv), (a, b) in matched.items():
        same_sys = nohint(parts(a["prompt"])[0]) == nohint(parts(b["prompt"])[0])
        sys_same[lv][same_sys] += 1
        if not same_sys:
            sys_diff_pairs[lv].add(pid)
    out["system_identical_by_level"] = {lv: {"identical": c[True], "differs": c[False]} for lv, c in sys_same.items()}
    out["system_differs_pairs_by_level"] = {lv: sorted(v) for lv, v in sys_diff_pairs.items()}
    l12 = set(sys_diff_pairs.get("L1", set())) | set(sys_diff_pairs.get("L2", set()))
    pairs_l12 = {pid for (pid, lv) in matched if lv in ("L1", "L2")}
    out["system_differs_at_L1_or_L2_pairs"] = {"n_differs": len(l12), "n_pairs": len(pairs_l12), "pairs": sorted(l12)}

    # ------------------------------------------------------------------ last tool result structure
    keys = ("leaves", "null", "false", "true", "zero", "empty", "chars")
    row_cmp = {k: Counter() for k in keys}
    pair_vals = defaultdict(lambda: defaultdict(list))
    res_same_across_levels = defaultdict(set)
    for (pid, lv), (a, b) in matched.items():
        ra, rb = parts(a["prompt"])[3], parts(b["prompt"])[3]
        if not ra or not rb:
            continue
        oa, rawa = result_obj(ra[-1])
        ob, rawb = result_obj(rb[-1])
        ca, cb = struct_counts(oa), struct_counts(ob)
        ca["chars"], cb["chars"] = len(rawa), len(rawb)
        res_same_across_levels[pid].add((rawa, rawb))
        for k in keys:
            row_cmp[k]["A>B" if ca[k] > cb[k] else ("B>A" if cb[k] > ca[k] else "tie")] += 1
            pair_vals[pid][k].append(ca[k] - cb[k])
    out["last_result_A_vs_B_rows"] = {k: dict(v) | {"sign_p": sign_test_p(v["A>B"], v["A>B"] + v["B>A"])}
                                      for k, v in row_cmp.items()}
    pc = {}
    for k in keys:
        c = Counter()
        for pid, d in pair_vals.items():
            x = sum(d[k]) / len(d[k])
            c["A>B" if x > 0 else ("B>A" if x < 0 else "tie")] += 1
        pc[k] = dict(c) | {"sign_p": sign_test_p(c["A>B"], c["A>B"] + c["B>A"])}
    out["last_result_A_vs_B_pairs"] = pc
    # cluster level (sibling variants of one source post share structure)
    cc = {}
    for k in keys:
        byc = defaultdict(list)
        for pid, d in pair_vals.items():
            byc[cluster.get(pid)].append(sum(d[k]) / len(d[k]))
        c = Counter("A>B" if sum(v) / len(v) > 0 else ("B>A" if sum(v) / len(v) < 0 else "tie") for v in byc.values())
        cc[k] = dict(c) | {"sign_p": sign_test_p(c["A>B"], c["A>B"] + c["B>A"])}
    out["last_result_A_vs_B_clusters"] = cc
    out["pairs_whose_last_results_vary_by_level"] = sorted(p for p, v in res_same_across_levels.items() if len(v) > 1)

    # ------------------------------------------------------------------ prompt length (exact tokens) on same-setting twins
    lens = Counter()
    diffs = []
    pair_len = defaultdict(list)
    for (pid, lv), (a, b) in matched.items():
        if a["prompt_setting"] != b["prompt_setting"]:
            continue
        la, lb = ts[("sft_train", a["_i"])]["prompt_tokens"], ts[("sft_train", b["_i"])]["prompt_tokens"]
        lens["A>B" if la > lb else ("B>A" if lb > la else "tie")] += 1
        diffs.append(la - lb)
        pair_len[pid].append(la - lb)
    out["prompt_tokens_same_setting_rows"] = dict(lens) | {"n": sum(lens.values()), "mean_A_minus_B": round(sum(diffs) / len(diffs), 2),
                                                           "sign_p": sign_test_p(lens["A>B"], lens["A>B"] + lens["B>A"])}
    pl = Counter("A>B" if sum(v) > 0 else ("B>A" if sum(v) < 0 else "tie") for v in pair_len.values())
    out["prompt_tokens_same_setting_pairs"] = dict(pl) | {"sign_p": sign_test_p(pl["A>B"], pl["A>B"] + pl["B>A"])}
    # all matched rows (any setting): the hint adds ~45 tokens to whichever twin drew P2
    allc = Counter()
    for (pid, lv), (a, b) in matched.items():
        la, lb = ts[("sft_train", a["_i"])]["prompt_tokens"], ts[("sft_train", b["_i"])]["prompt_tokens"]
        allc["A>B" if la > lb else ("B>A" if lb > la else "tie")] += 1
    out["prompt_tokens_all_matched_rows"] = dict(allc)
    mean_tok = [ts[("sft_train", r["_i"])]["prompt_tokens"] for ab in matched.values() for r in ab]
    out["mean_prompt_tokens_matched_decision_rows"] = round(sum(mean_tok) / len(mean_tok), 1)
    save(out, "v1_twins.json")
    for k, v in out.items():
        if k != "train_pairs_request_listing":
            print(k, v)


if __name__ == "__main__":
    main()
