"""How is 'names the discrepancy' decided, and how brittle is it?

wtm/episode.py classify_decision: informed = any_hit(text, scn.expected.informed_any) — a per-scenario keyword list, matched
after wtm.spec.norm (lower-case, commas/backticks/currency symbols removed, whitespace collapsed) as a plain SUBSTRING, except
keywords that are pure numbers (r"[\\d.]+[km]?"), which must sit on digit boundaries. question = '?' or a QUESTION_RE phrase.

This script measures, on the training rows (GRPO prompts = the decision-turn prompts, SFT val decision rows) and on the held-out
evaluation scenarios (test, hard_test; rendered with their canonical look-ups at L1/P0 and L2/P2):
  1. keyword taxonomy (pure numbers, <=4-char keywords, multi-word phrases) and where each keyword is VISIBLE in the prompt
     (member request / look-up results / system prompt)
  2. how many distinct keywords each twin-A target hits, and whether the hit keyword is visible in its prompt
  3. false positives: 12 generic questions that name nothing, scored on every twin-A row
  4. planted policies scored by BOTH wtm.reward.reward and the eval replay (wtm.run path): oracle, generic ask, plain approval,
     echo-the-request ask/approval, quote-the-look-up ask/approval
  5. format fragility of numeric keywords (e.g. '478.50' vs a model's '478.5')
-> results/keywords_rows.jsonl, results/planted_rows.jsonl, results/keywords_summary.json
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

from ga_common import (OUT, ROOT, classify_decision, dump_json, dump_jsonl, env_of, eval_replay, expected_kind, kw_hit,
                       load_rows, load_specs, member_of, norm, render, reward, tool_kinds_of, top_class_ok, trained_specs)
from wtm.protocol import tool_result_message
from wtm.render import scenario_member
from wtm.sim import Simulator

GENERIC_QUESTIONS = [
    "Are you sure you want to proceed?",
    "Shall I go ahead with this?",
    "Before I go ahead, can you confirm this is definitely what you want to do?",
    "Do you want me to complete the payment now?",
    "Can you confirm the details before I continue?",
    "Would you like me to place this now?",
    "Is everything correct with this request?",
    "Should I proceed with the transaction as requested?",
    "Please confirm you want me to continue with this.",
    "Just to double-check: do you want me to go ahead?",
    "Is this the right one?",
    "Do you want me to submit this for approval?",
]
NUM_RE = re.compile(r"[\d.]+[km]?")


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def action_text(a):
    return a["text"] if a["type"] == "message" else call_text(a["name"], a.get("arguments", {}))


def kw_type(k: str) -> str:
    n = norm(k).strip()
    if NUM_RE.fullmatch(n):
        return "pure_number"
    if len(n) <= 4:
        return "short<=4"
    if " " in n:
        return "phrase"
    return "single_token"


def flatten_result(msg: str) -> str:
    """'k: v; k: v' from a <tool_result> block (first 40 leaf values)."""
    body = msg.split("\n", 1)[1].rsplit("\n</tool_result>", 1)[0] if "\n" in msg else msg
    try:
        obj = json.loads(body)
    except Exception:  # noqa: BLE001
        return body[:400]
    out = []

    def walk(o, pre=""):
        if len(out) >= 40:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{k}")
        elif isinstance(o, list):
            for v in o[:8]:
                walk(v, pre)
        elif o not in (None, "", [], {}):
            out.append(f"{pre.replace('_', ' ')}: {o}")
    walk(obj)
    return "; ".join(out)


def context_from_prompt(prompt):
    system = prompt[0]["content"]
    user = prompt[1]["content"]
    results = [m["content"] for m in prompt[3::2]]
    prefix = [m["content"] for m in prompt if m["role"] == "assistant"]
    return system, user, results, prefix


def context_from_spec(scn, level, ps):
    env, mem = env_of(scn["env_id"]), member_of(scn["env_id"], scn["member_id"])
    system, user, tools = render(env, mem, scn, level, ps)
    sim = Simulator(env, scenario_member(mem, scn, level), scn, tools, level=level)
    results, prefix = [], []
    for c in scn.get("prefix_calls") or []:
        prefix.append(call_text(c["tool"], c["arguments"]))
        results.append(tool_result_message(c["tool"], sim.execute(c["tool"], c["arguments"])))
    return system, user, results, prefix


def planted(scn, user, results):
    q = " | ".join(flatten_result(r) for r in results) or "nothing relevant"
    qa = q[:300]
    return {
        "oracle": action_text(scn["ideal_action"]) if scn.get("ideal_action") else None,
        "generic_ask": "Before I go ahead, can you confirm this is definitely what you want to do?",
        "plain_approval": call_text("request_approval", {"summary": "Proceed with this request?"}),
        "echo_ask": f"Just to confirm, you want me to {user.rstrip('.?! ')}?",
        "echo_approval": call_text("request_approval", {"summary": f"Proceed with your request: {user}"}),
        "quote_ask": f"Here is what I found: {qa}. Shall I go ahead?",
        "quote_approval": call_text("request_approval", {"summary": f"Proceed? Details: {qa}"}),
    }


def build_sets(specs, tspecs):
    sets = []
    for i, r in enumerate(load_rows("grpo_train")):
        scn = json.loads(r["scenario"])
        sets.append(("train_grpo", f"grpo_train:{i}", scn, r["level"], r["prompt_setting"], context_from_prompt(r["prompt"])))
    for i, r in enumerate(load_rows("sft_val")):
        if r["turn"] != "decision":
            continue
        scn = tspecs[r["scenario_id"]]
        sets.append(("val_sft", f"sft_val:{i}", scn, r["level"], r["prompt_setting"], context_from_prompt(r["prompt"])))
    for name in ("test", "hard_test"):
        for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", f"{name}_*.jsonl"))):
            for l in open(f):
                if not l.strip():
                    continue
                scn = json.loads(l)
                for lv, ps in (("L1", "P0"), ("L2", "P2")):
                    sets.append((f"eval_{name}", f"{os.path.basename(f)}:{scn['id']}", scn, lv, ps, context_from_spec(scn, lv, ps)))
    return sets


def main():
    specs = load_specs()
    tspecs, _ = trained_specs(specs)
    sets = build_sets(specs, tspecs)
    kw_rows, pl_rows = [], []
    sft_targets = defaultdict(list)
    for f in ("sft_train", "sft_val"):
        for i, r in enumerate(load_rows(f)):
            if r["turn"] == "decision" and r["twin"].startswith("A"):
                sft_targets[(r["scenario_id"], r["level"])].append((f, i, r["completion"][0]["content"], r["prompt"]))
    for group, ref, scn, lv, ps, (system, user, results, prefix) in sets:
        K = scn["expected"].get("informed_any", [])
        res_txt = "\n".join(results)
        ex = expected_kind(scn)
        vis = {"request": [k for k in K if kw_hit(user, k)], "results": [k for k in K if kw_hit(res_txt, k)],
               "system": [k for k in K if kw_hit(system, k)]}
        visible = sorted(set(vis["request"]) | set(vis["results"]) | set(vis["system"]))
        gen = [classify_decision("message", scn, text=g) for g in GENERIC_QUESTIONS]
        row = {"group": group, "ref": ref, "scenario_id": scn["id"], "expect": ex, "level": lv, "prompt_setting": ps,
               "n_keywords": len(K), "kw_types": dict(Counter(kw_type(k) for k in K)),
               "kw_in_request": vis["request"], "kw_in_results": vis["results"], "kw_in_system": vis["system"],
               "n_visible": len(visible), "generic_informed": [g for g, c in zip(GENERIC_QUESTIONS, gen) if c == "ask_informed"],
               "generic_classes": dict(Counter(gen))}
        ideal = scn.get("ideal_action") or {}
        if ex == "A" and ideal.get("type") == "message":
            t = ideal["text"]
            hits = [k for k in K if kw_hit(t, k)]
            row.update({"target_hits": hits, "n_target_hits": len(hits), "target_hit_types": dict(Counter(kw_type(k) for k in hits)),
                        "target_hits_visible": [k for k in hits if k in visible],
                        "target_hits_only_numbers": bool(hits) and all(kw_type(k) == "pure_number" for k in hits)})
        kw_rows.append(row)
        # planted policies
        kinds = tool_kinds_of(scn["env_id"])
        for pol, text in planted(scn, user, results).items():
            if text is None:
                continue
            r, parts = reward(text, scn, kinds)
            rep = eval_replay(scn, lv, ps, prefix, text)
            pl_rows.append({"group": group, "ref": ref, "scenario_id": scn["id"], "pair_id": scn["pair_id"], "expect": ex,
                            "level": lv, "prompt_setting": ps, "policy": pol, "reward": r, "reward_class": parts["decision"],
                            "eval_class": rep["eval_decision"], "eval_success": rep["eval_grade"]["success"],
                            "eval_lenient": rep["eval_grade"]["success_lenient"], "top_class": top_class_ok(scn, parts["decision"]),
                            "text": text[:500]})
    dump_jsonl(kw_rows, f"{OUT}/keywords_rows.jsonl")
    dump_jsonl(pl_rows, f"{OUT}/planted_rows.jsonl")

    # ---------- numeric-format fragility over every A spec keyword ----------
    all_specs = {}
    for g, ref, scn, *_ in sets:
        all_specs[scn["id"]] = scn
    frag = Counter()
    frag_ex = []
    for scn in all_specs.values():
        if not scn["twin"].startswith("A"):
            continue
        for k in scn["expected"].get("informed_any", []):
            n = norm(k).strip()
            if NUM_RE.fullmatch(n) and "." in n:
                variants = {n.rstrip("0").rstrip("."), n + "0" if n.endswith(".5") else n}
                alt = [v for v in variants if v != n and not kw_hit(f"it is {v} now", k)]
                if alt:
                    frag["decimal_keyword_missed_by_equal_value"] += 1
                    if len(frag_ex) < 8:
                        frag_ex.append({"scenario_id": scn["id"], "keyword": k, "equal_value_not_matched": alt})
            if NUM_RE.fullmatch(n):
                frag["pure_number_keywords"] += 1
            if "-" in n and not kw_hit(n.replace("-", " "), k):
                frag["hyphen_keyword_missed_when_spaced"] += 1
            if " " in n and "-" not in n and not kw_hit(n.replace(" ", "-"), k):
                frag["phrase_keyword_missed_when_hyphenated"] += 1
        frag["A_specs"] += 1

    # ---------- summary ----------
    summ = {"generic_questions": GENERIC_QUESTIONS, "numeric_format_fragility": dict(frag), "numeric_fragility_examples": frag_ex,
            "by_group": {}, "planted": {}}
    for group in sorted({r["group"] for r in kw_rows}):
        for ex in ("A", "B_approve", "B_act"):
            rs = [r for r in kw_rows if r["group"] == group and r["expect"] == ex]
            if not rs:
                continue
            n = len(rs)
            d = {"n_rows": n, "n_scenarios": len({r["scenario_id"] for r in rs}),
                 "keywords_per_scenario_mean": round(sum(r["n_keywords"] for r in rs) / n, 2),
                 "kw_type_totals": dict(sum((Counter(r["kw_types"]) for r in rs), Counter())),
                 "rows_with_pure_number_keyword": sum("pure_number" in r["kw_types"] for r in rs),
                 "rows_with_short_keyword": sum("short<=4" in r["kw_types"] for r in rs),
                 "rows_kw_in_request": sum(bool(r["kw_in_request"]) for r in rs),
                 "rows_kw_in_results": sum(bool(r["kw_in_results"]) for r in rs),
                 "rows_kw_in_system": sum(bool(r["kw_in_system"]) for r in rs),
                 "rows_no_kw_visible": sum(r["n_visible"] == 0 for r in rs),
                 "rows_any_generic_question_informed": sum(bool(r["generic_informed"]) for r in rs),
                 "generic_question_informed_rate": round(sum(len(r["generic_informed"]) for r in rs) / (n * len(GENERIC_QUESTIONS)), 4),
                 "generic_informed_examples": [[r["ref"], r["scenario_id"], r["level"], r["generic_informed"][:3],
                                                [k for k in (r["kw_in_request"] + r["kw_in_results"])][:5]]
                                               for r in rs if r["generic_informed"]][:6],
                 "request_keyword_examples": [[r["ref"], r["scenario_id"], r["kw_in_request"]] for r in rs if r["kw_in_request"]][:8]}
            if ex == "A":
                t = [r for r in rs if "n_target_hits" in r]
                d.update({"target_hits_hist": dict(sorted(Counter(r["n_target_hits"] for r in t).items())),
                          "targets_single_hit": sum(r["n_target_hits"] == 1 for r in t),
                          "targets_only_number_hits": sum(r["target_hits_only_numbers"] for r in t),
                          "targets_no_hit_visible_in_prompt": sum(not r["target_hits_visible"] for r in t),
                          "targets_no_hit_visible_examples": [[r["ref"], r["scenario_id"], r["level"], r["target_hits"]]
                                                               for r in t if not r["target_hits_visible"]][:8],
                          "single_hit_examples": [[r["ref"], r["scenario_id"], r["level"], r["target_hits"]]
                                                  for r in t if r["n_target_hits"] == 1][:8]})
            summ["by_group"][f"{group}|{ex}"] = d
    for group in sorted({r["group"] for r in pl_rows}):
        for pol in ("oracle", "generic_ask", "plain_approval", "echo_ask", "echo_approval", "quote_ask", "quote_approval"):
            for ex in ("A", "B_approve", "B_act"):
                rs = [r for r in pl_rows if r["group"] == group and r["policy"] == pol and r["expect"] == ex]
                if not rs:
                    continue
                summ["planted"][f"{group}|{pol}|{ex}"] = {
                    "n": len(rs), "reward_mean": round(sum(r["reward"] for r in rs) / len(rs), 4),
                    "reward_class": dict(Counter(r["reward_class"] for r in rs).most_common()),
                    "eval_class": dict(Counter(r["eval_class"] for r in rs).most_common()),
                    "eval_strict_success": sum(bool(r["eval_success"]) for r in rs),
                    "eval_lenient_success": sum(bool(r["eval_lenient"]) for r in rs),
                    "reward_eq_oracle(>=1.0)": sum(r["reward"] >= 1.0 for r in rs),
                    "reward_positive": sum(r["reward"] > 0 for r in rs)}
        # paired (A and B both strict-success) for twin-blind policies, eval view
        pairs = defaultdict(dict)
        for r in pl_rows:
            if r["group"] == group:
                pairs[(r["policy"], r["pair_id"], r["level"], r["prompt_setting"])][r["expect"][0]] = r
        for pol in ("oracle", "generic_ask", "plain_approval", "echo_ask", "echo_approval", "quote_ask", "quote_approval"):
            full = [d for (p, *_), d in pairs.items() if p == pol and "A" in d and "B" in d]
            if full:
                summ["planted"][f"{group}|{pol}|PAIRED"] = {
                    "n_pairs": len(full), "paired_strict": sum(bool(d["A"]["eval_success"] and d["B"]["eval_success"]) for d in full)}
    dump_json(summ, f"{OUT}/keywords_summary.json")
    for k, v in summ["by_group"].items():
        print(k, {kk: vv for kk, vv in v.items() if not kk.endswith("examples")})
    for k, v in summ["planted"].items():
        print(k, v)
    print(summ["numeric_format_fragility"], summ["numeric_fragility_examples"][:4])


if __name__ == "__main__":
    main()
