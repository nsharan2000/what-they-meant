"""Spec-level checks of what the grader can and cannot recognise, over every scenario spec (train/val seed + variants, eval
test/val/hard files), plus the SFT treatment of bare [system, user] prompts.

  gold_unmatchable      a gold_commits match key that can never occur in json.dumps(args) (tool names such as 'pay_batch';
                        decimal strings like '212000.00' that a JSON float serialises as 212000.0)
  B_act_ideal_not_gold  twin-B 'act' rows whose ideal_action is not classified commit_gold (the reward then pays -0.4 for the
                        ideal; the eval still passes it because decision_grade counts ANY commit as strict success for 'act')
  A_literal_is_gold     twin-A rows where an approval card / commit that just restates the member's literal request is classed
                        approve_gold / commit_gold (reward +0.6 / -0.9, eval lenient success for approve_gold)
  B_act_any_commit      decision_grade(B, exp='act') strict success for commit_other / commit_harmful (reward -0.5 / -1.0)
  adjudicated_in_train  rows whose label adjudication accepts 'act' while wtm.reward ignores expected.accept
  bare_prompt_targets   SFT rows whose prompt is only [system, user]: how many teach a look-up vs a decision
-> results/signature_summary.json
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

from ga_common import (OUT, ROOT, adjudicate, adjudications, classify_decision, decision_grade, dump_json, kw_hit, load_rows,
                       load_specs, reward, tool_kinds_of)


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def main():
    specs = load_specs()
    origin = {sid: "train_or_val_spec" for sid in specs}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        for l in open(f):
            if l.strip():
                s = json.loads(l)
                if s["id"] not in specs:
                    specs[s["id"]] = s
                    origin[s["id"]] = os.path.basename(f).rsplit("_", 1)[0]
    splits = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    pair_split = {pid: k for k in ("train", "val", "test") for pid in splits[k]}
    tool_names = {n for e in ("wallet", "bills", "shopping", "bizops") for n in tool_kinds_of(e)}
    out = {"gold_unmatchable": [], "B_act_ideal_not_gold": [], "A_literal_is_gold": [], "harmful_in_request": Counter()}
    for sid, s in sorted(specs.items()):
        where = pair_split.get(s["pair_id"], origin[sid])
        for g in s.get("gold_commits") or []:
            for k in g.get("match", []):
                if k in tool_names:
                    out["gold_unmatchable"].append([sid, where, k, "tool name (classify_decision only sees the arguments)"])
                elif k.replace(".", "").isdigit() and k.endswith(".00"):
                    out["gold_unmatchable"].append([sid, where, k, "decimal string; a JSON float argument serialises as N.0"])
        if s["twin"].startswith("B") and s["expected"]["decision"] == "act" and s.get("ideal_action"):
            ia = s["ideal_action"]
            kinds = tool_kinds_of(s["env_id"])
            cls = classify_decision(kinds.get(ia["name"], "invalid"), s, args=ia.get("arguments", {}))
            if cls != "commit_gold":
                r, _ = reward(call_text(ia["name"], ia.get("arguments", {})), s, kinds)
                out["B_act_ideal_not_gold"].append([sid, where, cls, r, decision_grade(s, cls)["success"], s.get("gold_commits")])
        if s["twin"].startswith("A"):
            lit_card = classify_decision("ui", s, args={"summary": s["user_message"]})
            lit_commit = classify_decision("commit", s, args={"note": s["user_message"]})
            if lit_card == "approve_gold" or lit_commit == "commit_gold":
                out["A_literal_is_gold"].append([sid, where, lit_card, lit_commit, s.get("gold_commits"), s["user_message"]])
            hm = [k for h in s.get("harmful_commits") or [] for k in h.get("match", [])]
            out["harmful_in_request"]["has_harmful_signature"] += bool(hm)
            out["harmful_in_request"]["literal_request_matches_harmful"] += bool(hm) and lit_commit == "commit_harmful"
            out["harmful_in_request"]["A_specs"] += 1
    # decision_grade on B 'act' for every commit class
    fake_act = {"twin": "B_should_act", "expected": {"decision": "act", "informed_any": []}}
    out["B_act_grade_by_class"] = {c: decision_grade(fake_act, c)["success"] for c in ("commit_gold", "commit_other", "commit_harmful")}
    out["B_act_reward_by_class"] = {c: reward_table_value(c) for c in ("commit_gold", "commit_other", "commit_harmful")}
    # adjudicated rows in the trained files
    adj = adjudications()
    cnt = Counter()
    for f in ("sft_train", "sft_val", "dpo_train", "grpo_train"):
        for r in load_rows(f):
            if r["scenario_id"] in adj and (r.get("turn", "decision") == "decision"):
                cnt[f] += 1
    out["adjudicated_rows_in_training_files"] = dict(cnt)
    out["adjudicated_commit_gold_reward_vs_eval"] = {}
    for sid in sorted(adj):
        if sid in specs:
            s = specs[sid]
            out["adjudicated_commit_gold_reward_vs_eval"][sid] = {
                "reward_table": "B_ACT" if s["expected"]["decision"] == "act" else "B_APPROVE",
                "eval_strict_success_after_adjudication": decision_grade(adjudicate(s), "commit_gold")["success"],
                "eval_strict_success_raw_run_py": decision_grade(s, "commit_gold")["success"]}
    # bare [system, user] prompts in SFT
    bare = Counter()
    ex = []
    for f in ("sft_train", "sft_val"):
        for i, r in enumerate(load_rows(f)):
            if len(r["prompt"]) == 2:
                bare[f"{f}|{r['turn']}"] += 1
                if r["turn"] == "decision" and len(ex) < 6:
                    ex.append([f, i, r["scenario_id"], r["level"]])
    out["bare_prompt_targets"] = dict(bare)
    out["bare_prompt_decision_examples"] = ex
    # GRPO prompts: do they contain every prefix look-up of their spec (i.e. is the next correct move the decision)?
    mism, bare_grpo = [], Counter()
    for i, r in enumerate(load_rows("grpo_train")):
        s = json.loads(r["scenario"])
        n_asst = sum(m["role"] == "assistant" for m in r["prompt"])
        if n_asst != len(s.get("prefix_calls") or []):
            mism.append([i, r["scenario_id"], r["level"], n_asst, len(s.get("prefix_calls") or [])])
        if n_asst == 0:
            bare_grpo[r["scenario_id"]] += 1
    out["grpo_prompt_lookups_ne_spec_prefix_calls"] = mism
    out["grpo_rows_with_bare_prompt"] = {"n": sum(bare_grpo.values()), "by_scenario": dict(bare_grpo)}
    out["harmful_in_request"] = dict(out["harmful_in_request"])
    dump_json(out, f"{OUT}/signature_summary.json")
    print(json.dumps(out, indent=1, default=str)[:6000])


def reward_table_value(c):
    from wtm.reward import B_ACT
    return B_ACT.get(c)


if __name__ == "__main__":
    main()
