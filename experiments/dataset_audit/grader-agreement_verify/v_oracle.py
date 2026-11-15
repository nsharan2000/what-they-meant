"""Targets vs both scorers (own replay), DPO ordering, render parity, spec drift.

For every row of the four trained files:
  reward path : wtm.reward.reward(target, spec, env tool kinds)
  eval path   : render + run_episode(decision mode) replaying the prompt's own assistant turns, then the target
  parity      : the system prompt and messages the eval shows before the target == the stored prompt
-> results/v_oracle_rows.jsonl, results/v_oracle_summary.json
"""
from __future__ import annotations

import json
from collections import Counter

from vcommon import EP, RW, adjudicate, dump, eval_class, expect, kinds, rows, spec_files, target_text

specs = spec_files()
grpo = rows("grpo_train")
emb = {}
for r in grpo:
    s = json.loads(r["scenario"])
    emb.setdefault(s["id"], s)
drift = [sid for sid, s in emb.items() if json.dumps(s, sort_keys=True) != json.dumps(specs.get(sid), sort_keys=True)]


def top(scn, cls):
    e = expect(scn)
    if e == "A":
        return cls == "ask_informed"
    if e == "B_act":
        return cls == "commit_gold"
    return cls.startswith("approve")


out, S = [], {"spec_drift_grpo_vs_files": drift, "n_embedded_specs": len(emb)}
for name in ("sft_train", "sft_val", "dpo_train", "grpo_train"):
    for i, r in enumerate(rows(name)):
        sid = r["scenario_id"]
        scn = json.loads(r["scenario"]) if name == "grpo_train" else specs[sid]
        k = kinds(scn["env_id"])
        prompt = r["prompt"]
        prefix = [m["content"] for m in prompt if m["role"] == "assistant"]
        if name == "grpo_train":
            tgt = target_text(scn)
        elif name == "dpo_train":
            tgt = r["chosen"][0]["content"]
        else:
            tgt = r["completion"][0]["content"]
        rew, parts = RW.reward(tgt, scn, k)
        res, drv, system, user = eval_class(scn, r["level"], r["prompt_setting"], prefix, tgt)
        n = len(prefix)
        ev = res["events"][n] if len(res["events"]) > n else {}
        seen_sys, seen_msgs = drv.seen[n] if len(drv.seen) > n else (None, None)
        parity_sys = seen_sys == prompt[0]["content"]
        parity_msgs = seen_msgs == [dict(m) for m in prompt[1:]]
        row = {"file": name, "i": i, "sid": sid, "lv": r["level"], "ps": r["prompt_setting"], "turn": r.get("turn", "decision"),
               "expect": expect(scn), "reward": rew, "reward_class": parts["decision"], "eval_decision": res["decision_grade"]["decision"],
               "eval_success": res["decision_grade"]["success"], "target_event": ev.get("tool_kind") or ev.get("kind"),
               "parity_sys": parity_sys, "parity_msgs": parity_msgs}
        if name == "dpo_train":
            rej = r["rejected"][0]["content"]
            rr, rp = RW.reward(rej, scn, k)
            res2, *_ = eval_class(scn, r["level"], r["prompt_setting"], prefix, rej)
            adj = adjudicate(scn)
            g2 = EP.decision_grade(adj, res2["decision_grade"]["decision"])
            row.update({"rej_reward": rr, "rej_class": rp["decision"], "rej_eval_decision": res2["decision_grade"]["decision"],
                        "rej_eval_success_raw": res2["decision_grade"]["success"], "rej_eval_success_adjudicated": g2["success"],
                        "rej_eval_lenient_adjudicated": g2["success_lenient"]})
        if row["turn"] == "decision":
            row["top_reward"] = top(scn, parts["decision"])
            row["top_eval"] = top(scn, row["eval_decision"])
        out.append(row)

dump(out, "v_oracle_rows.jsonl")
dec = [r for r in out if r["turn"] == "decision"]
pos = [r for r in dec]           # sft decision targets + dpo chosen + grpo ideals
S["n_rows_total"] = len(out)
S["n_positive_targets"] = len(pos)
S["positive_by_file"] = dict(Counter(r["file"] for r in pos))
S["positive_top_reward"] = sum(r["top_reward"] for r in pos)
S["positive_top_eval"] = sum(r["top_eval"] for r in pos)
S["positive_reward_class_ne_eval"] = sum(r["reward_class"] != r["eval_decision"] for r in pos)
S["positive_reward_values"] = dict(Counter(r["reward"] for r in pos))
S["positive_eval_success"] = sum(bool(r["eval_success"]) for r in pos)
S["positive_classes_by_expect"] = {e: dict(Counter(r["reward_class"] for r in pos if r["expect"] == e)) for e in ("A", "B_approve", "B_act")}
look = [r for r in out if r["turn"] != "decision"]
S["lookup_targets"] = len(look)
S["lookup_reward_values"] = dict(Counter(r["reward"] for r in look))
S["lookup_target_eval_event"] = dict(Counter(r["target_event"] for r in look))
S["parity_sys"] = [sum(r["parity_sys"] for r in out), len(out)]
S["parity_msgs"] = [sum(r["parity_msgs"] for r in out), len(out)]
S["parity_fail_examples"] = [[r["file"], r["i"], r["sid"], r["lv"], r["ps"]] for r in out if not (r["parity_sys"] and r["parity_msgs"])][:10]
d = [r for r in out if r["file"] == "dpo_train"]
S["dpo_n"] = len(d)
S["dpo_chosen_gt_rejected"] = sum(r["reward"] > r["rej_reward"] for r in d)
S["dpo_min_margin"] = round(min(r["reward"] - r["rej_reward"] for r in d), 4)
S["dpo_rej_eval_success_raw"] = sum(bool(r["rej_eval_success_raw"]) for r in d)
S["dpo_rej_eval_success_adjudicated"] = sum(bool(r["rej_eval_success_adjudicated"]) for r in d)
S["dpo_rej_eval_lenient_adjudicated"] = sum(bool(r["rej_eval_lenient_adjudicated"]) for r in d)
S["dpo_rej_lenient_examples"] = [[r["i"], r["sid"], r["lv"], r["rej_eval_decision"]] for r in d if r["rej_eval_lenient_adjudicated"]][:20]
S["dpo_rej_classes"] = dict(Counter(f"{r['expect']}|{r['rej_class']}" for r in d).most_common())
S["dpo_rej_reward_class_ne_eval"] = sum(r["rej_class"] != r["rej_eval_decision"] for r in d)
S["dpo_rej_reward_vs_eval_mismatch_examples"] = [[r["i"], r["sid"], r["lv"], r["rej_class"], r["rej_eval_decision"]]
                                                 for r in d if r["rej_class"] != r["rej_eval_decision"]][:15]
dump(S, "v_oracle_summary.json")
print(json.dumps(S, indent=1))
