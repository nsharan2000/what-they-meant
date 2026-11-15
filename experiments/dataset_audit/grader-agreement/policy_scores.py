"""The M1 policy's sampled decisions on the GRPO prompts (experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl):
class mix by twin / level / prompt setting / step, which scenarios produce A-ask_generic and B-ask_informed, how those relate to
keyword visibility in the prompt (results/keywords_rows.jsonl from informed_keywords.py), and the GRPO group signal
(reward spread inside each 8-sample group; a group with zero spread gives zero advantage under scale_rewards='group').

-> results/policy_scores_summary.json
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

from ga_common import OUT, ROOT, adjudicate, dump_json, load_rows

SCORES = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")


def rate(c: Counter, k: str) -> float:
    n = sum(c.values())
    return round(c[k] / n, 4) if n else None


def main():
    sc = [json.loads(l) for l in open(SCORES) if l.strip()]
    grpo = load_rows("grpo_train")
    by_key = {(r["scenario_id"], r["level"]): (i, r) for i, r in enumerate(grpo)}
    kw = {}
    for l in open(f"{OUT}/keywords_rows.jsonl"):
        r = json.loads(l)
        if r["group"] == "train_grpo":
            kw[int(r["ref"].split(":")[1])] = r
    rows = []
    for s in sc:
        m = s["meta"]
        i, g = by_key.get((m["scenario_id"], m["level"]), (None, None))
        scn = json.loads(g["scenario"]) if g else None
        k = kw.get(i, {})
        rows.append({"sid": m["scenario_id"], "level": m["level"], "twin": m["twin"][:1], "step": m["step"],
                     "cls": s["components"]["decision"], "score": s["score"], "base": s["components"].get("base"),
                     "trunc": m.get("truncated"), "n_tokens": m.get("n_tokens"), "line": i,
                     "ps": g["prompt_setting"] if g else None,
                     "expect": None if scn is None else ("A" if scn["twin"].startswith("A") else
                                                         ("B_act" if scn["expected"]["decision"] == "act" else "B_approve")),
                     "accept": None if scn is None else adjudicate(scn)["expected"].get("accept"),
                     "n_visible": k.get("n_visible"), "kw_in_results": bool(k.get("kw_in_results")),
                     "kw_in_system": bool(k.get("kw_in_system")), "single_hit": k.get("n_target_hits") == 1})
    summ = {"n_samples": len(rows), "unjoined": sum(r["line"] is None for r in rows)}
    for tw in ("A", "B"):
        rs = [r for r in rows if r["twin"] == tw]
        summ[f"{tw}_classes"] = dict(Counter(r["cls"] for r in rs).most_common())
        summ[f"{tw}_by_level"] = {lv: dict(Counter(r["cls"] for r in rs if r["level"] == lv).most_common())
                                  for lv in sorted({r["level"] for r in rs})}
        summ[f"{tw}_by_prompt_setting"] = {ps: dict(Counter(r["cls"] for r in rs if r["ps"] == ps).most_common())
                                           for ps in sorted({str(r["ps"]) for r in rs})}
        summ[f"{tw}_by_step_bucket"] = {b: dict(Counter(r["cls"] for r in rs if (r["step"] or 0) // 10 == b).most_common())
                                        for b in sorted({(r["step"] or 0) // 10 for r in rs})}
        summ[f"{tw}_mean_score"] = round(sum(r["score"] for r in rs) / len(rs), 4)
    # B by expectation (approve vs act; adjudicated rows)
    summ["B_by_expect"] = {ex: dict(Counter(r["cls"] for r in rows if r["expect"] == ex).most_common()) for ex in ("B_approve", "B_act")}
    cg = [r for r in rows if r["twin"] == "B" and r["cls"] == "commit_gold"]
    summ["B_commit_gold_detail"] = dict(Counter(f"{r['sid']}|{r['expect']}|accept={r['accept']}|score={r['score']}" for r in cg))
    # A ask_generic and B ask_informed by scenario, and feature splits
    a = [r for r in rows if r["twin"] == "A"]
    b = [r for r in rows if r["twin"] == "B"]
    per = defaultdict(Counter)
    line_of = {}
    for r in rows:
        per[(r["sid"], r["level"])][r["cls"]] += 1
        line_of[(r["sid"], r["level"])] = f"grpo_train:{r['line']}"
    summ["A_ask_generic_by_scenario_level"] = sorted([[k[0], k[1], c["ask_generic"], sum(c.values()), line_of[k]]
                                                      for k, c in per.items() if k[0].endswith("A") and c["ask_generic"]],
                                                     key=lambda x: -x[2])
    summ["B_ask_informed_by_scenario_level"] = sorted([[k[0], k[1], c["ask_informed"], sum(c.values()), line_of[k]]
                                                       for k, c in per.items() if k[0].endswith("B") and c["ask_informed"]],
                                                      key=lambda x: -x[2])
    summ["A_lookup_by_scenario_level"] = sorted([[k[0], k[1], c["lookup"], sum(c.values()), line_of[k]] for k, c in per.items()
                                                 if c["lookup"]], key=lambda x: -x[2])

    def split(rs, feat, cls):
        out = {}
        for v in (True, False):
            sub = [r for r in rs if bool(r[feat]) == v]
            c = Counter(r["cls"] for r in sub)
            out[str(v)] = {"n": len(sub), cls: c[cls], "rate": rate(c, cls)}
        return out
    summ["A_ask_generic_vs_single_hit_target"] = split(a, "single_hit", "ask_generic")
    summ["A_ask_generic_vs_kw_in_results"] = split(a, "kw_in_results", "ask_generic")
    summ["A_ask_informed_vs_kw_in_results"] = split(a, "kw_in_results", "ask_informed")
    summ["B_ask_informed_vs_kw_visible_in_prompt"] = {
        "visible": {"n": sum(1 for r in b if (r["n_visible"] or 0) > 0),
                    "ask_informed": sum(1 for r in b if (r["n_visible"] or 0) > 0 and r["cls"] == "ask_informed"),
                    "any_ask": sum(1 for r in b if (r["n_visible"] or 0) > 0 and r["cls"].startswith("ask"))},
        "not_visible": {"n": sum(1 for r in b if not r["n_visible"]),
                        "ask_informed": sum(1 for r in b if not r["n_visible"] and r["cls"] == "ask_informed"),
                        "any_ask": sum(1 for r in b if not r["n_visible"] and r["cls"].startswith("ask"))}}
    # B ask_informed where NONE of the (A-twin) keywords is visible in the B prompt: the keyword had to come from the model,
    # i.e. it named the A twin's clash on a prompt that does not contain it
    leak = Counter((r["sid"], r["level"], r["line"]) for r in b if r["cls"] == "ask_informed" and not r["n_visible"])
    summ["B_ask_informed_with_no_keyword_in_prompt"] = sorted([[s, lv, f"grpo_train:{ln}", n] for (s, lv, ln), n in leak.items()],
                                                              key=lambda x: -x[3])
    # GRPO group signal
    groups = defaultdict(list)
    for r in rows:
        groups[(r["sid"], r["level"], r["step"])].append(r)
    stds, zero, one_cls, sizes = [], 0, 0, Counter()
    for g in groups.values():
        xs = [r["score"] for r in g]
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
        stds.append(sd)
        zero += sd < 1e-9
        one_cls += len({r["cls"] for r in g}) == 1
        sizes[len(g)] += 1
    summ["grpo_groups"] = {"n_groups": len(groups), "group_sizes": dict(sizes), "zero_reward_spread": zero,
                           "single_class_groups": one_cls, "mean_within_group_std": round(sum(stds) / len(stds), 4)}
    # group-normalised advantage per class ((r - group mean) / group std, std over the 8 samples; TRL scale_rewards='group'
    # uses the same centring and an unbiased-or-not std, which does not change the sign)
    adv = defaultdict(list)
    for g in groups.values():
        xs = [r["score"] for r in g]
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
        if sd < 1e-9:
            continue
        for r in g:
            adv[(r["twin"], r["cls"])].append((r["score"] - mu) / sd)
    summ["mean_group_advantage_by_class"] = {f"{tw}|{c}": {"n": len(v), "mean_adv": round(sum(v) / len(v), 3)}
                                             for (tw, c), v in sorted(adv.items(), key=lambda kv: (kv[0][0], -len(kv[1])))}
    # how much of the 'wrong class' rate depends on the keyword rule at all
    nA, nB = len(a), len(b)
    A_type_wrong = sum(r["cls"].startswith(("approve", "commit")) for r in a)
    B_type_wrong = sum(r["cls"].startswith(("ask", "tell", "statement")) for r in b)
    A_kw_dependent = sum(r["cls"] in ("ask_generic", "statement", "tell_informed") for r in a)
    summ["error_decomposition"] = {
        "A_n": nA, "A_top_class(ask_informed)": sum(r["cls"] == "ask_informed" for r in a),
        "A_approval_or_commit(type error, keyword-independent)": A_type_wrong,
        "A_approve_informed_within_that": sum(r["cls"] == "approve_informed" for r in a),
        "A_message_but_not_ask_informed(keyword/question-rule dependent)": A_kw_dependent,
        "A_lookup(eval-neutral, reward 0.0)": sum(r["cls"] == "lookup" for r in a),
        "A_invalid": sum(r["cls"] == "invalid" for r in a),
        "B_n": nB, "B_approve_or_expected_commit": sum(r["cls"].startswith("approve") or (r["expect"] == "B_act" and r["cls"] == "commit_gold") for r in b),
        "B_chat_message(type error, keyword-independent)": B_type_wrong,
        "B_other(commit on approve rows, commit_other, lookup, invalid)": nB - B_type_wrong - sum(
            r["cls"].startswith("approve") or (r["expect"] == "B_act" and r["cls"] == "commit_gold") for r in b)}
    # P0 vs P2 and by level: top-class rates
    def top(r):
        if r["twin"] == "A":
            return r["cls"] == "ask_informed"
        return r["cls"].startswith("approve") or (r["expect"] == "B_act" and r["cls"] == "commit_gold")
    summ["top_class_rate_by_prompt_setting"] = {f"{tw}|{ps}": [sum(top(r) for r in rows if r["twin"] == tw and r["ps"] == ps),
                                                              sum(1 for r in rows if r["twin"] == tw and r["ps"] == ps)]
                                                for tw in ("A", "B") for ps in ("P0", "P2")}
    summ["top_class_rate_by_level"] = {f"{tw}|{lv}": [sum(top(r) for r in rows if r["twin"] == tw and r["level"] == lv),
                                                      sum(1 for r in rows if r["twin"] == tw and r["level"] == lv)]
                                       for tw in ("A", "B") for lv in ("L0r", "L1", "L2", "L3")}
    # mean reward by step (first/last thirds) for context
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r["score"])
    summ["mean_score_by_step"] = {s: round(sum(v) / len(v), 4) for s, v in sorted(by_step.items(), key=lambda kv: (kv[0] is None, kv[0]))}
    summ["truncated"] = sum(bool(r["trunc"]) for r in rows)
    dump_json(summ, f"{OUT}/policy_scores_summary.json")
    print(json.dumps({k: v for k, v in summ.items() if not k.endswith("by_scenario_level") and k != "mean_score_by_step"}, indent=1))
    print("A ask_generic by scenario:", summ["A_ask_generic_by_scenario_level"][:15])
    print("B ask_informed by scenario:", summ["B_ask_informed_by_scenario_level"][:15])
    print("A lookup by scenario:", summ["A_lookup_by_scenario_level"][:12])


if __name__ == "__main__":
    main()
