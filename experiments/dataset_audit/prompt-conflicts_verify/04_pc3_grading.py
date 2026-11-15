"""V04 - PC3 grading check: what does a prompt-sanctioned DIRECT commit get on the 5 authorisation-covered 'approve' B scenarios
under (a) the GRPO reward (wtm.reward.reward, hidden labels as written in grpo_train) and (b) the programmatic grader after the label
adjudication (wtm.episode.decision_grade(wtm.labels.adjudicate(spec))) that experiments/replication/analysis.py applies?
The commit call is built from the env's commit tool with the ideal approval summary's facts as arguments.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/04_pc3_grading.py -> results/04_pc3_grading.json
"""
import json

from vcommon import dump, envs, reward, rows, specs
from wtm.episode import decision_grade
from wtm.labels import adjudicate

SP = specs()
E = envs()
G = {r["scenario_id"]: r for r in rows("grpo_train")}
COMMIT = {"shopping": "place_order", "bills": "pay_bill"}
out = {}
for sid in ("S04B", "V042B", "V043B", "V044B", "V223B"):
    scn = json.loads(G[sid]["scenario"])
    kinds = {t["name"]: t["kind"] for t in E[scn["env_id"]]["tools"]}
    summ = scn["ideal_action"]["arguments"].get("summary", "")
    call = '<tool_call>' + json.dumps({"name": COMMIT[scn["env_id"]], "arguments": {"summary": summ}}) + '</tool_call>'
    tot, comp = reward(call, scn, kinds)
    adj = adjudicate(SP[sid])
    g = decision_grade(adj, comp["decision"])
    card = '<tool_call>' + json.dumps({"name": "request_approval", "arguments": {"summary": summ}}) + '</tool_call>'
    out[sid] = {"commit_call": call, "grpo_reward": tot, "decision_class": comp["decision"], "gold_commits": scn.get("gold_commits"),
                "adjudicated_accept": adj["expected"].get("accept"), "grader_strict_success": g["success"],
                "grader_lenient_success": g["success_lenient"], "card_reward": reward(card, scn, kinds)[0]}
p = dump("04_pc3_grading.json", out)
print(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote", p)
