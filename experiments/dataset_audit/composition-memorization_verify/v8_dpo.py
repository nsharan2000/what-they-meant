"""CM-7: grader class of DPO rejected sides vs M1's actual error modes (independent re-derivation with wtm.reward.decision_of).

DPO rows carry no scenario; each DPO prompt is matched to the GRPO row with the identical prompt (which carries the scenario).
Output: results/v8_dpo.json
"""
import json
from collections import Counter

from vcommon import cur, dump, is_call, call_name
from wtm.reward import decision_of, reward
from wtm.protocol import parse_action

dpo, grpo = cur("dpo_train"), cur("grpo_train")
gp = {json.dumps(g["prompt"], ensure_ascii=False): g for g in grpo}
out = {"dpo_rows": len(dpo), "matched_to_grpo": 0}
cls = {"A": Counter(), "B": Counter()}
chosen_cls = {"A": Counter(), "B": Counter()}
pivot = Counter()
rej_score_pos = 0
examples = []
for i, d in enumerate(dpo):
    g = gp.get(json.dumps(d["prompt"], ensure_ascii=False))
    if g is None:
        continue
    out["matched_to_grpo"] += 1
    scn, kinds = json.loads(g["scenario"]), json.loads(g["tool_kinds"])
    rj, ch = d["rejected"][0]["content"], d["chosen"][0]["content"]
    c_rej, _ = decision_of(rj, scn, kinds)
    c_ch, _ = decision_of(ch, scn, kinds)
    t = d["twin"][0]
    cls[t][c_rej] += 1
    chosen_cls[t][c_ch] += 1
    if reward(rj, scn, kinds)[0] > 0:
        rej_score_pos += 1
    pa, pb = parse_action(ch, set(kinds)), parse_action(rj, set(kinds))
    a_call, b_call = pa.kind == "tool_call", pb.kind == "tool_call"
    if pb.kind == "invalid":
        pivot["rejected_invalid"] += 1
    elif a_call != b_call:
        pivot["first_token_message_vs_call"] += 1
    elif a_call and b_call:
        pivot["which_tool" if pa.name != pb.name else "same_tool"] += 1
    else:
        pivot["both_messages"] += 1
    # does the rejected side start with the literal '<tool_call>' (same first token '<' as a tool-call chosen)?
    pivot["rejected_starts_with_lt"] += rj.lstrip().startswith("<")
    pivot["chosen_starts_with_lt"] += ch.lstrip().startswith("<")
    pivot["first_char_differs"] += ch.lstrip()[:1] != rj.lstrip()[:1]
    if i < 8:
        examples.append({"line": i, "twin": t, "rejected_class": c_rej, "rejected": rj[:140]})
out["rejected_class_by_twin"] = {t: dict(c.most_common()) for t, c in cls.items()}
out["chosen_class_by_twin"] = {t: dict(c.most_common()) for t, c in chosen_cls.items()}
out["rejected_with_reward_gt_0"] = rej_score_pos
out["pivot"] = dict(pivot)
out["rejected_bare_json_or_unclosed"] = sum(1 for d in dpo if not is_call(d["rejected"][0]["content"].strip()) and
                                            d["rejected"][0]["content"].strip().startswith("{"))
# how often is the rejected side the error M1 actually makes (A -> approval card, B -> chat message)?
out["rejected_A_approval_cards"] = sum(v for k, v in cls["A"].items() if k.startswith("approve"))
out["rejected_B_chat_messages"] = sum(v for k, v in cls["B"].items() if k in ("ask_generic", "ask_informed", "tell_informed", "statement"))
out["examples"] = examples
dump(out, "v8_dpo.json")
print(json.dumps(out, indent=1, ensure_ascii=False))
