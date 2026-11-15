"""Step 1 (brief items 1 + 3): match twin A and twin B rows and diff their prompts message by message; check twin balance.

Matching, per file (sft_train, sft_val, dpo_train, grpo_train of dataset/curated_v121; pushback rows excluded from matching):
  strict = same pair_id, level, prompt_setting, turn (what the brief asks for)
  loose  = same pair_id, level, turn, any prompt_setting; the P2 hint is stripped from both system prompts before diffing
           (build_datasets.py draws P0/P2 independently for every scenario x level, so twins often get different settings)
Per matched pair (loose; strict is a subset) the script records which messages differ (system / request / look-up call / tool
result), changed characters per side, an ESTIMATED token count (changed chars / the row's own chars-per-token from the exact
token_stats.jsonl), the exact whole-prompt token difference, the system sections that differ, the JSON fields of the tool
results that differ, and the position of the differences measured in estimated tokens from the end of the prompt.
"Decisive-linked" (heuristic, validated by examples in the summary): a changed request token / result field counts as decisive
when its value (or for booleans/null its key) shares an informative token with the pair's decisive vocabulary = both twins'
decisive_facts, decisive_strings, expected.{discrepancy, must_check, informed_any, ideal_message}, ideal_action and clash_type.

python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s1_twin_diff.py
-> results/s1_twin_diff_rows.jsonl, results/s1_twin_diff_summary.json, results/s1_balance.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import difflib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import statistics as st  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, CUR, FILES, RES, components, flatten, generic_path, jdump, jl_dump, load_jsonl,  # noqa: E402
                    load_rows, load_specs, load_token_stats, parse_call, parse_result, section_at, strip_hint,
                    system_sections, twin_letter)

TOK = re.compile(r"\s+|\w+|[^\w\s]")
IDENT = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*|@[\w.-]+|[$€£₹]?\w*\d[\w,.\-/:]*|\b[\w-]+\.(?:com|net|org|pro|in|store|dev|de|"
                   r"ph|io|co|uk|app)\b")
STOP = set("""a an the to of for and or in on at by is it its be are was were this that with as from my me i you your he she his her
their they them we our us do does did not no so if but please send pay using use have has had will would can just now new
s t ll ve re m d""".split())
GENERIC_JSON = {"null", "true", "false", "none", "id", "name", "type", "status", "date", "amount", "currency", "usd", "gbp", "eur",
                "inr"}


def toks_with_spans(s):
    out, pos = [], 0
    for m in TOK.finditer(s):
        out.append((m.group(0), m.start(), m.end()))
    return out


def text_diff(a: str, b: str):
    """-> dict(changed chars a/b, list of (a_start, a_end, b_start, b_end, a_text, b_text))"""
    if a == b:
        return {"a_chars": 0, "b_chars": 0, "spans": []}
    la, lb = a.splitlines(keepends=True), b.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    oa = [0]
    for x in la:
        oa.append(oa[-1] + len(x))
    ob = [0]
    for x in lb:
        ob.append(ob[-1] + len(x))
    spans = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        sa, sb = a[oa[i1]:oa[i2]], b[ob[j1]:ob[j2]]
        ta, tb = toks_with_spans(sa), toks_with_spans(sb)
        sm2 = difflib.SequenceMatcher(None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False)
        for op2, k1, k2, l1, l2 in sm2.get_opcodes():
            if op2 == "equal":
                continue
            as_ = oa[i1] + (ta[k1][1] if k1 < len(ta) else len(sa))
            ae = oa[i1] + (ta[k2 - 1][2] if k2 > k1 else (ta[k1][1] if k1 < len(ta) else len(sa)))
            bs = ob[j1] + (tb[l1][1] if l1 < len(tb) else len(sb))
            be = ob[j1] + (tb[l2 - 1][2] if l2 > l1 else (tb[l1][1] if l1 < len(tb) else len(sb)))
            spans.append((as_, ae, bs, be, a[as_:ae], b[bs:be]))
    return {"a_chars": sum(s[1] - s[0] for s in spans), "b_chars": sum(s[3] - s[2] for s in spans), "spans": spans}


def info_tokens(text) -> set:
    t = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    t = t.lower().replace(",", "")
    out = set()
    for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?", t):
        if w in STOP or w in GENERIC_JSON:
            continue
        if w.isalpha() and len(w) < 3:
            continue
        out.add(w)
        if re.fullmatch(r"\d+\.0+", w):
            out.add(w.split(".")[0])
    # ISO dates -> also day/month tokens
    for y, mth, d in re.findall(r"(\d{4})-(\d{2})-(\d{2})", t):
        out |= {str(int(d)), ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][int(mth) - 1]}
    return out


def decisive_vocab(sa: dict, sb: dict) -> set:
    v = set()
    for s in (sa, sb):
        exp = s.get("expected", {})
        parts = list(s.get("decisive_facts") or []) + list(s.get("decisive_strings") or []) + [
            exp.get("discrepancy", ""), exp.get("must_check", ""), exp.get("ideal_message", ""), s.get("clash_type", "")]
        parts += list(exp.get("informed_any") or [])
        ia = s.get("ideal_action") or {}
        parts.append(ia.get("text") or json.dumps(ia.get("arguments", {}), ensure_ascii=False))
        for p in parts:
            v |= info_tokens(p or "")
    return v


def leaf_linked(path, va, vb, vocab) -> bool:
    """lenient on purpose (a field counts as decisive when its value OR its key name shares an informative token with the
    decisive vocabulary), so 'unlinked' fields are a conservative lower bound on non-decisive differences."""
    toks = set()
    for v in (va, vb):
        if v is None or isinstance(v, bool) or v in ([], {}):
            continue
        toks |= info_tokens(v)
    key_toks = {w for w in re.split(r"[^a-z]+", generic_path(path).lower()) if len(w) > 2 and w not in GENERIC_JSON}
    return bool(toks & vocab) or bool(key_toks & vocab)


def result_diff(ra: str, rb: str, vocab):
    na, oa = parse_result(ra)
    nb, ob = parse_result(rb)
    out = {"tool_a": na, "tool_b": nb, "json": isinstance(oa, (dict, list)) and isinstance(ob, (dict, list))}
    if not out["json"]:
        out["text_changed"] = oa != ob
        return out
    fa, fb = flatten(oa), flatten(ob)
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    changed = sorted(p for p in set(fa) & set(fb) if json.dumps(fa[p]) != json.dumps(fb[p]))
    leaves = [(p, fa.get(p, "<absent>"), fb.get(p, "<absent>")) for p in only_a + only_b + changed]
    linked = [p for p, va, vb in leaves if leaf_linked(p, None if va == "<absent>" else va, None if vb == "<absent>" else vb, vocab)]

    def nulls(f):
        return sum(1 for v in f.values() if v is None)

    def empties(f):
        return sum(1 for v in f.values() if v in ([], {}))
    out.update({"n_leaves_a": len(fa), "n_leaves_b": len(fb), "only_a": only_a, "only_b": only_b, "changed": changed,
                "n_diff_leaves": len(leaves), "n_linked": len(linked),
                "unlinked": [[p, fa.get(p, "<absent>"), fb.get(p, "<absent>")] for p, _, _ in leaves if p not in linked][:40],
                "linked": [[p, fa.get(p, "<absent>"), fb.get(p, "<absent>")] for p in linked][:40],
                "nulls_a": nulls(fa), "nulls_b": nulls(fb), "empties_a": empties(fa), "empties_b": empties(fb),
                "keys_a": len({generic_path(p) for p in fa}), "keys_b": len({generic_path(p) for p in fb}),
                "schema_diff": sorted({generic_path(p) for p in only_a} ^ {generic_path(p) for p in only_b}
                                      - ({generic_path(p) for p in only_a} & {generic_path(p) for p in only_b}))})
    return out


def request_kind(ra: str, rb: str, vocab):
    if ra == rb:
        return "identical", [], []
    ma, mb = IDENT.sub("<ID>", ra), IDENT.sub("<ID>", rb)
    wa, wb = re.findall(r"[a-z]+", ra.lower()), re.findall(r"[a-z]+", rb.lower())
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    rem, add = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            rem += wa[i1:i2]
            add += wb[j1:j2]
    if ma == mb:
        return "identifier_only", rem, add
    return "wording", rem, add


def main():
    specs, _ = load_specs()
    tstats = load_token_stats()
    out_rows, balance = [], {}
    build_report = json.load(open(os.path.join(CUR, "build_report.json")))
    s0 = json.load(open(os.path.join(RES, "s0_rebuild.json")))
    for f in FILES:
        rows = load_rows(f)
        idx = defaultdict(dict)
        n_push = 0
        for r in rows:
            t = twin_letter(r["twin"])
            if t == "P":
                n_push += 1
                continue
            idx[(r["pair_id"], r["level"], r["turn"])][t] = r
        # ---------------- balance (item 3)
        pre = load_rows(f, os.path.join(CACHE, "rebuild_v121"))
        pre_idx = defaultdict(dict)
        for r in pre:
            if twin_letter(r["twin"]) != "P":
                pre_idx[(r["pair_id"], r["level"], r["turn"])][twin_letter(r["twin"])] = r
        drops = {(d["scenario_id"], d["level"], d["turn"] or "decision"): d["reason"] for d in build_report["dropped"]
                 if d["view"] == f}
        cur_drops = {(d["scenario_id"], d["level"], d["turn"]): d["reason"] + (f" (kept copy: twin {d['kept_duplicate_twin']})"
                                                                              if d.get("kept_duplicate_twin") else "")
                     for d in s0["curation_attribution"][f]["dropped"]}
        bal = Counter()
        unmatched = []
        pair_rows = defaultdict(Counter)
        for (pid, lv, turn), d in idx.items():
            for t, r in d.items():
                pair_rows[pid][(t, turn == "decision")] += 1
                o = "B" if t == "A" else "A"
                other = d.get(o)
                tk = "decision" if turn == "decision" else "lookup"
                if other is None:
                    sid = pid + o
                    why = drops.get((sid, lv, turn)) or cur_drops.get((sid, lv, turn))
                    if why is None:
                        if sid not in specs:
                            why = "other twin absent from specs"
                        elif o not in pre_idx.get((pid, lv, turn), {}):
                            why = "other twin never rendered at this level/turn (different number of look-ups)"
                        else:
                            why = "?"
                    bal[(tk, t, "no_counterpart")] += 1
                    unmatched.append({"pair_id": pid, "level": lv, "turn": turn, "present": t, "missing": o,
                                      "present_row": [f, r["_i"]], "why": why or "?"})
                elif other["prompt_setting"] != r["prompt_setting"]:
                    bal[(tk, t, "counterpart_other_prompt_setting")] += 1
                else:
                    bal[(tk, t, "strict_counterpart")] += 1
        # pair-level label skew among decision rows
        skew = []
        for pid, c in pair_rows.items():
            a, b = c[("A", True)], c[("B", True)]
            if a + b:
                skew.append(max(a, b) / (a + b))
        dec = [r for r in rows if r["turn"] == "decision" and twin_letter(r["twin"]) != "P"]
        maj = sum(max(pair_rows[p][("A", True)], pair_rows[p][("B", True)]) for p in pair_rows) / max(1, len(dec))
        ps_twin = Counter((twin_letter(r["twin"]), r["prompt_setting"]) for r in rows if twin_letter(r["twin"]) != "P")
        ps_dec_twin = Counter((twin_letter(r["twin"]), r["prompt_setting"]) for r in dec)
        balance[f] = {"n_rows": len(rows), "n_pushback_rows": n_push,
                      "counts": {"|".join(k): v for k, v in sorted(bal.items())},
                      "unmatched_rows": unmatched,
                      "unmatched_by_why": dict(Counter(f"{u['turn'][:6]}|{u['why']}" for u in unmatched)),
                      "pair_majority_classifier_acc_decision_rows": round(maj, 4),
                      "pairs_with_unequal_A_B_decision_rows": sum(1 for p in pair_rows if pair_rows[p][("A", True)]
                                                                  != pair_rows[p][("B", True)]),
                      "n_pairs": len(pair_rows),
                      "prompt_setting_by_twin_all_rows": {f"{k[0]}|{k[1]}": v for k, v in sorted(ps_twin.items())},
                      "prompt_setting_by_twin_decision_rows": {f"{k[0]}|{k[1]}": v for k, v in sorted(ps_dec_twin.items())}}

        # ---------------- diffs (item 1)
        for (pid, lv, turn), d in sorted(idx.items()):
            if "A" not in d or "B" not in d:
                continue
            a, b = d["A"], d["B"]
            sa, sb = specs[a["scenario_id"]], specs[b["scenario_id"]]
            vocab = decisive_vocab(sa, sb)
            pa, pb = a["prompt"], b["prompt"]
            ta, tb = tstats[(f, a["_i"])], tstats[(f, b["_i"])]
            chars_a = sum(len(m["content"]) for m in pa)
            chars_b = sum(len(m["content"]) for m in pb)
            cpt_a, cpt_b = chars_a / ta["prompt_tokens"], chars_b / tb["prompt_tokens"]
            rec = {"file": f, "pair_id": pid, "level": lv, "turn": turn, "a_row": a["_i"], "b_row": b["_i"],
                   "ps_a": a["prompt_setting"], "ps_b": b["prompt_setting"], "strict": a["prompt_setting"] == b["prompt_setting"],
                   "env": a["env_id"], "n_msgs_a": len(pa), "n_msgs_b": len(pb),
                   "tokens_a": ta["prompt_tokens"], "tokens_b": tb["prompt_tokens"],
                   "tokens_a_minus_b": ta["prompt_tokens"] - tb["prompt_tokens"],
                   "tokens_a_minus_b_hint_adjusted": None, "member_same": sa["member_id"] == sb["member_id"],
                   "target_a": (a.get("completion") or a.get("chosen") or [{"content": ""}])[0]["content"][:200],
                   "target_b": (b.get("completion") or b.get("chosen") or [{"content": ""}])[0]["content"][:200],
                   "loci": [], "msgs": []}
            # distance-from-end bookkeeping (chars after each message start)
            after_a = [sum(len(m["content"]) for m in pa[j + 1:]) for j in range(len(pa))]
            after_b = [sum(len(m["content"]) for m in pb[j + 1:]) for j in range(len(pb))]
            dist_a, dist_b = [], []
            for j in range(max(len(pa), len(pb))):
                ma = pa[j] if j < len(pa) else None
                mb = pb[j] if j < len(pb) else None
                role = (ma or mb)["role"]
                kind = ("system" if j == 0 else "request" if j == 1 else "call" if role == "assistant" else "result")
                ca = ma["content"] if ma else ""
                cb = mb["content"] if mb else ""
                if j == 0:
                    ca, cb = strip_hint(ca), strip_hint(cb)
                if ca == cb:
                    continue
                td = text_diff(ca, cb)
                m = {"j": j, "kind": kind, "a_chars": td["a_chars"], "b_chars": td["b_chars"],
                     "a_tok_est": round(td["a_chars"] / cpt_a, 1), "b_tok_est": round(td["b_chars"] / cpt_b, 1),
                     "only_one_side": ma is None or mb is None}
                for s_ in td["spans"]:
                    if ma is not None:
                        dist_a.append((len(ca) - s_[0]) + after_a[j])
                    if mb is not None:
                        dist_b.append((len(cb) - s_[2]) + after_b[j])
                if kind == "system":
                    secs_a, secs_b = system_sections(ca), system_sections(cb)
                    secs = Counter()
                    for s_ in td["spans"]:
                        secs[section_at(secs_a, s_[0])] += s_[1] - s_[0]
                        secs[section_at(secs_b, s_[2])] += s_[3] - s_[2]
                    m["sections"] = dict(secs.most_common())
                elif kind == "request":
                    rk, rem, add = request_kind(ca, cb, vocab)
                    extra = sorted({w for w in rem + add if w not in STOP and w not in vocab and len(w) > 2})
                    m.update({"request_kind": rk, "a_only_words": rem, "b_only_words": add, "words_not_in_decisive_vocab": extra,
                              "text_a": ca, "text_b": cb})
                elif kind == "call":
                    na, aa = parse_call(ca)
                    nb, ab = parse_call(cb)
                    fa, fb = flatten(aa), flatten(ab)
                    m.update({"tool_a": na, "tool_b": nb, "same_tool": na == nb,
                              "arg_diff": sorted(p for p in set(fa) | set(fb) if json.dumps(fa.get(p)) != json.dumps(fb.get(p))),
                              "call_a": ca[:300], "call_b": cb[:300]})
                else:
                    if ma is not None and mb is not None:
                        m.update(result_diff(ca, cb, vocab))
                rec["msgs"].append(m)
                rec["loci"].append(kind)
            rec["loci"] = sorted(set(rec["loci"]))
            rec["diff_chars_a"] = sum(m["a_chars"] for m in rec["msgs"])
            rec["diff_chars_b"] = sum(m["b_chars"] for m in rec["msgs"])
            rec["diff_tok_est_a"] = round(rec["diff_chars_a"] / cpt_a, 1)
            rec["diff_tok_est_b"] = round(rec["diff_chars_b"] / cpt_b, 1)
            rec["farthest_diff_tok_from_end_est"] = round(max([x / cpt_a for x in dist_a] + [x / cpt_b for x in dist_b] or [0]))
            rec["nearest_diff_tok_from_end_est"] = round(min([x / cpt_a for x in dist_a] + [x / cpt_b for x in dist_b] or [0]))
            res = [m for m in rec["msgs"] if m["kind"] == "result" and m.get("json")]
            rec["result_diff_leaves"] = sum(m["n_diff_leaves"] for m in res)
            rec["result_unlinked_leaves"] = sum(m["n_diff_leaves"] - m["n_linked"] for m in res)
            req = [m for m in rec["msgs"] if m["kind"] == "request"]
            rec["request_kind"] = req[0]["request_kind"] if req else "identical"
            rec["request_extra_words"] = req[0]["words_not_in_decisive_vocab"] if req else []
            # single locus and decisive-only heuristics
            rec["single_locus"] = len(rec["loci"]) == 1
            rec["decisive_only_heuristic"] = (rec["loci"] in (["result"], ["call", "result"], ["request"],
                                                              ["call", "request", "result"], ["request", "result"])
                                              and rec["result_unlinked_leaves"] == 0
                                              and not rec["request_extra_words"]
                                              and all(m.get("same_tool", True) for m in rec["msgs"] if m["kind"] == "call"))
            out_rows.append(rec)

    jl_dump(out_rows, os.path.join(RES, "s1_twin_diff_rows.jsonl"))
    jdump(balance, os.path.join(RES, "s1_balance.json"))

    # ---------------- aggregates
    def agg(sel):
        if not sel:
            return {}
        loci = Counter("+".join(r["loci"]) or "none" for r in sel)
        sysec = Counter()
        for r in sel:
            for m in r["msgs"]:
                if m["kind"] == "system":
                    for k in m["sections"]:
                        sysec[k] += 1
        fields = Counter()
        unl_fields = Counter()
        for r in sel:
            for m in r["msgs"]:
                if m["kind"] == "result" and m.get("json"):
                    for p in set(generic_path(x) for x in m["only_a"] + m["only_b"] + m["changed"]):
                        fields[p] += 1
                    for p in set(generic_path(x[0]) for x in m["unlinked"]):
                        unl_fields[p] += 1
        dc = [r["diff_chars_a"] + r["diff_chars_b"] for r in sel]
        dt = [r["diff_tok_est_a"] + r["diff_tok_est_b"] for r in sel]
        share = sorted((r["diff_tok_est_a"] + r["diff_tok_est_b"]) / (r["tokens_a"] + r["tokens_b"]) for r in sel)
        tad = [r["tokens_a_minus_b"] for r in sel if r["strict"]]
        far = [r["farthest_diff_tok_from_end_est"] for r in sel]
        return {"n": len(sel), "loci": dict(loci.most_common()),
                "share_system_differs": round(sum("system" in r["loci"] for r in sel) / len(sel), 3),
                "share_request_differs": round(sum("request" in r["loci"] for r in sel) / len(sel), 3),
                "share_call_differs": round(sum("call" in r["loci"] for r in sel) / len(sel), 3),
                "share_result_differs": round(sum("result" in r["loci"] for r in sel) / len(sel), 3),
                "share_identical_prompts": round(sum(not r["loci"] for r in sel) / len(sel), 3),
                "request_kind": dict(Counter(r["request_kind"] for r in sel)),
                "system_sections_differing": dict(sysec.most_common()),
                "diff_chars_a_plus_b": {"median": st.median(dc), "mean": round(st.mean(dc), 1), "max": max(dc)},
                "diff_tok_est_a_plus_b": {"median": st.median(dt), "mean": round(st.mean(dt), 1), "max": max(dt)},
                "share_of_prompt_tokens_differing_est": {"median": round(st.median(share), 4), "mean": round(st.mean(share), 4),
                                                         "p90": round(share[int(0.9 * (len(share) - 1))], 4), "max": round(share[-1], 4)},
                "exact_token_diff_a_minus_b_strict": ({"n": len(tad), "mean": round(st.mean(tad), 1), "median": st.median(tad),
                                                       "share_a_longer": round(sum(x > 0 for x in tad) / len(tad), 3),
                                                       "share_b_longer": round(sum(x < 0 for x in tad) / len(tad), 3),
                                                       "share_equal": round(sum(x == 0 for x in tad) / len(tad), 3)} if tad else {}),
                "farthest_diff_tok_from_end_est": {"median": st.median(far), "max": max(far),
                                                   "share_gt_4096": round(sum(x > 4096 for x in far) / len(far), 3)},
                "result_fields_differing_top": dict(fields.most_common(30)),
                "result_unlinked_fields_top": dict(unl_fields.most_common(30)),
                "share_rows_with_unlinked_result_fields": round(sum(r["result_unlinked_leaves"] > 0 for r in sel) / len(sel), 3),
                "share_rows_request_extra_words": round(sum(bool(r["request_extra_words"]) for r in sel) / len(sel), 3),
                "share_single_locus": round(sum(r["single_locus"] for r in sel) / len(sel), 3),
                "share_decisive_only_heuristic": round(sum(r["decisive_only_heuristic"] for r in sel) / len(sel), 3),
                "share_member_differs": round(sum(not r["member_same"] for r in sel) / len(sel), 3),
                "share_strict_match": round(sum(r["strict"] for r in sel) / len(sel), 3)}

    summary = {"note": "loose matches (same pair, level, turn; P2 hint stripped); 'strict' share = same prompt_setting too. "
                       "Token counts for diff spans are ESTIMATES (chars / the row's exact chars-per-token); whole-prompt "
                       "token differences are exact (token_stats.jsonl).", "by_file": {}}
    for f in FILES:
        sel = [r for r in out_rows if r["file"] == f]
        summary["by_file"][f] = {"decision": agg([r for r in sel if r["turn"] == "decision"]),
                                 "lookup": agg([r for r in sel if r["turn"] != "decision"]),
                                 "decision_by_level": {lv: agg([r for r in sel if r["turn"] == "decision" and r["level"] == lv])
                                                       for lv in ("L0r", "L1", "L2", "L3")}}
    # pair-level view on sft_train decision rows at L1 (one row per pair when present)
    pairs = {}
    for r in out_rows:
        if r["file"] == "sft_train" and r["turn"] == "decision":
            pairs.setdefault(r["pair_id"], []).append(r)
    summary["sft_train_pairs_with_matched_decision_rows"] = len(pairs)
    summary["sft_train_pairs_request_differs"] = sum(any(x["request_kind"] != "identical" for x in v) for v in pairs.values())
    summary["sft_train_pairs_call_differs"] = sum(any("call" in x["loci"] for x in v) for v in pairs.values())
    summary["sft_train_pairs_system_differs_any_level"] = sum(any("system" in x["loci"] for x in v) for v in pairs.values())
    summary["sft_train_pairs_system_differs_L1_or_L2"] = sum(any("system" in x["loci"] and x["level"] in ("L1", "L2", "L3")
                                                                 for x in v) for v in pairs.values())
    summary["sft_train_pairs_result_only"] = sum(all(x["loci"] in (["result"],) for x in v) for v in pairs.values())
    jdump(summary, os.path.join(RES, "s1_twin_diff_summary.json"))
    print(json.dumps({f: {k: v for k, v in summary["by_file"][f]["decision"].items()
                          if k not in ("result_fields_differing_top", "result_unlinked_fields_top")} for f in FILES}, indent=1))
    print({k: v for k, v in summary.items() if k.startswith("sft_train_pairs")})
    for f in FILES:
        print(f, json.dumps(balance[f]["counts"]), balance[f]["pair_majority_classifier_acc_decision_rows"],
              balance[f]["prompt_setting_by_twin_decision_rows"])


if __name__ == "__main__":
    main()
