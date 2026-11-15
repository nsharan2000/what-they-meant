"""Counterpoint to CM-1's dilution story: the act-vs-approve pivot (tool-name token, 18 SFT rows) vs the ask-vs-approve pivot
(first token, 497 rows). M1 commit rate on act-expected vs approve-expected twin-B prompts, and the cue each act scenario carries.
Output: results/v13_act.json"""
import json, os
from collections import Counter, defaultdict
from vcommon import ROOT, cur, dump, load_jsonl, target, request, is_call, call_name
S = load_jsonl(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))
grpo = cur("grpo_train"); tr = cur("sft_train")
spec = {(g["scenario_id"], g["level"]): json.loads(g["scenario"]) for g in grpo}
act_rows = [r for r in tr if r["turn"] == "decision" and is_call(target(r)) and call_name(target(r)) != "request_approval"]
out = {"sft_act_rows": len(act_rows), "sft_act_scenarios": sorted({r["scenario_id"] for r in act_rows})}
per = defaultdict(Counter)
for s in S:
    m = s["meta"]
    if m["twin"].startswith("B"):
        per[(m["scenario_id"], m["level"], spec[(m["scenario_id"], m["level"])]["expected"]["decision"])][s["components"]["decision"]] += 1
rows = []
for (sid, lv, e), c in sorted(per.items()):
    n = sum(c.values()); com = sum(v for k, v in c.items() if k.startswith("commit"))
    rows.append({"sid": sid, "level": lv, "expected": e, "n": n, "commit": com, "approve": sum(v for k, v in c.items() if k.startswith("approve")),
                 "message": sum(v for k, v in c.items() if k in ("ask_generic", "ask_informed", "tell_informed", "statement"))})
act = [r for r in rows if r["expected"] == "act"]; app = [r for r in rows if r["expected"] != "act"]
out["act_prompts"] = act
out["commit_rate_act_prompts"] = round(sum(r["commit"] for r in act) / sum(r["n"] for r in act), 4)
out["commit_rate_approve_prompts"] = round(sum(r["commit"] for r in app) / sum(r["n"] for r in app), 4)
out["message_rate_act_prompts"] = round(sum(r["message"] for r in act) / sum(r["n"] for r in act), 4)
out["message_rate_approve_prompts"] = round(sum(r["message"] for r in app) / sum(r["n"] for r in app), 4)
# the cue: does the act scenario's system prompt carry a standing instruction / does the request differ from its approve-type twins?
cues = {}
for r in act:
    g = next(x for x in grpo if x["scenario_id"] == r["sid"] and x["level"] == r["level"])
    sysm = g["prompt"][0]["content"]
    i = sysm.find("### Standing instructions")
    cues[f'{r["sid"]}/{r["level"]}'] = {"request": request(g), "standing_instructions": sysm[i:i + 300] if i >= 0 else None}
out["act_cues"] = cues
dump(out, "v13_act.json"); print(json.dumps(out, indent=1, ensure_ascii=False))
