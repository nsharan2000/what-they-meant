"""V01 - one record per row of the 4 training files with prompt segments, target decision class (wtm.reward.decision_of =
the GRPO reward's class = the evaluation grader's class) and exact prompt tokens (experiments/dataset_audit/results/token_stats.jsonl).
Also checks the rendering contract independently (policy prefix = env template, tail = '' or exactly prompts/golden_short.md).
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/01_table.py
Output: results/table.jsonl, results/01_checks.json
"""
import collections
import json
import os

from vcommon import FILES, HINT, OUT, ROOT, decision_of, dump, envs, listed_tools, n_lookups, parse_action, reward, rows, \
    segments, specs

E = envs()
S = specs()
tok = {}
for l in open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")):
    t = json.loads(l)
    tok[(t["file"].replace(".jsonl", ""), t["i"])] = t["prompt_tokens"]

checks = collections.Counter()
out = []
for f in FILES:
    for i, r in enumerate(rows(f)):
        env = E[r["env_id"]]
        sysm = r["prompt"][0]
        assert sysm["role"] == "system"
        seg = segments(sysm["content"], env, r["level"])
        checks[f"tail:{'hint' if seg['tail'] == HINT else ('empty' if not seg['tail'] else 'OTHER')}:{r['prompt_setting']}"] += 1
        kinds = {t["name"]: t["kind"] for t in env["tools"]}
        scn = json.loads(r["scenario"]) if f == "grpo_train" else S[r["scenario_id"]]
        if f == "grpo_train" and json.dumps(scn, sort_keys=True) != json.dumps(S[r["scenario_id"]], sort_keys=True):
            checks["grpo_scenario_differs_from_spec"] += 1
        listed = listed_tools(seg["tools"])
        rec = {"file": f, "i": i, "sid": r["scenario_id"], "pair": r["pair_id"], "twin": r["twin"], "t": r["twin"][0],
               "env": r["env_id"], "level": r["level"], "ps": r["prompt_setting"], "turn": r.get("turn", "decision"),
               "n_lookups": n_lookups(r["prompt"]), "n_msgs": len(r["prompt"]), "expected": scn["expected"]["decision"],
               "tags": scn.get("tags", []), "policy_chars": len(seg["policy"]), "member_chars": len(seg["member"]),
               "tools_chars": len(seg["tools"]), "tail_chars": len(seg["tail"]),
               "post_chars": sum(len(m["content"]) for m in r["prompt"][1:]), "listed": listed,
               "prompt_tokens": tok.get((f, i)), "user": r["prompt"][1]["content"]}
        for role in ("completion", "chosen", "rejected"):
            if role in r:
                txt = r[role][0]["content"]
                dec, strict = decision_of(txt, scn, kinds)
                act = parse_action(txt, set(listed))
                rec[role] = {"text": txt, "decision": dec, "reward": reward(txt, scn, kinds)[0], "kind": act.kind,
                             "tool": act.name or None, "tool_listed": (act.name in listed) if act.name else None,
                             "strict": act.strict_ok, "flags": act.flags}
        out.append(rec)
with open(os.path.join(OUT, "table.jsonl"), "w") as fh:
    for rec in out:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
p = dump("01_checks.json", {"checks": dict(checks), "n": collections.Counter(r["file"] for r in out),
                            "decision_rows": collections.Counter(r["file"] for r in out if r["turn"] == "decision")})
print(json.dumps(dict(checks), indent=1))
print("wrote", p)
