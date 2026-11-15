"""Planted-signal check of the evaluation instrument (wtm.episode.run_episode in decision mode, exactly as wtm/run.py calls it):
oracle (canonical look-ups, then the ideal action) must pass every row; fixed policies must fail one twin of each pair.

Policies: oracle | card_after_lookups (canonical look-ups, then a plain approval card) | ask_after_lookups (canonical look-ups,
then a generic question) | card_now (no look-up, plain card) | ask_now (no look-up, generic question).
Rows: every dataset/eval file x levels L0r L1 L2 L2s x prompts P0 P2 (the run_eval.sh grid). Strict success = raw grader
(run.py applies no adjudication) and with label_adjudications applied.
Output: results/v11_planted_policies.json
"""
import json
from collections import Counter

from vcommon import call_text, dump, env, eval_sets, member, render
from wtm.episode import decision_grade, run_episode
from wtm.labels import adjudicate


class Res:
    def __init__(self, text):
        self.text, self.is_error, self.raw_error, self.cost_usd, self.usage = text, False, "", 0.0, {}


class Policy:
    def __init__(self, s, kind):
        self.s, self.kind, self.i = s, kind, 0

    def chat(self, system, msgs):
        calls = self.s.get("prefix_calls") or []
        if self.kind.endswith("after_lookups") or self.kind == "oracle":
            if self.i < len(calls):
                c = calls[self.i]
                self.i += 1
                return Res(call_text(c["tool"], c["arguments"]))
        if self.kind == "oracle":
            a = self.s["ideal_action"]
            return Res(a["text"] if a["type"] == "message" else call_text(a["name"], a.get("arguments", {})))
        if self.kind.startswith("card"):
            return Res(call_text("request_approval", {"summary": "Go ahead with: " + self.s["user_message"]}))
        return Res("Before I go ahead, can you confirm this is what you want?")


ev = eval_sets()
out = {}
fails = []
for name, rows in ev.items():
    c = Counter()
    for s in rows:
        e, m = env(s["env_id"]), member(s["env_id"], s["member_id"])
        for lv in ("L0r", "L1", "L2", "L2s"):
            for p in ("P0", "P2"):
                system, user, tools = render(e, m, s, lv, p)
                for kind in ("oracle", "card_after_lookups", "ask_after_lookups", "card_now", "ask_now"):
                    res = run_episode(e, m, s, Policy(s, kind), system, user, tools, mode="decision", max_steps=14, level=lv)
                    g = res["decision_grade"]
                    ga = decision_grade(adjudicate(s), g["decision"])
                    c[(kind, "n")] += 1
                    c[(kind, "strict_raw")] += bool(g["success"])
                    c[(kind, "strict_adj")] += bool(ga["success"])
                    if kind == "oracle" and not g["success"] and len(fails) < 10:
                        fails.append([name, s["id"], lv, p, g["decision"]])
    out[name] = {f"{k}|{m_}": v for (k, m_), v in sorted(c.items())}
dump("v11_planted_policies.json", {"by_set": out, "oracle_failures": fails})
for k, v in out.items():
    print(k, v)
print("oracle failures", fails)
