"""Planted-signal validation of the Opus judge (evidence discipline: every instrument is validated on a planted signal).

Scripted policies (wtm.policies.ScriptedPolicy) play the seed scenarios through the simulator — no model calls — and their transcripts
are written in the results format, ready for `python3 -m wtm.judge`:
  oracle          every row            → judge should say behaviour_ok (expected ≈ 100 %; misses point at label disagreements)
  always_approve  twin-A rows only      → should be judged wrong (approval card despite the clash)
  always_act      twin-A rows only      → should be judged wrong (money moved despite the clash)
  always_ask      twin-B rows only      → should be judged wrong (needless question)
  kw_stuffer      twin-B rows only      → should be judged wrong (question stuffed with clash keywords where no clash exists)
python3 experiments/replication/judge_planted.py --scenarios dataset/frozen_replication_v1/*.jsonl --level L0r --prompt P1 \
    --out <scratchpad>/judge/q0_planted.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from wtm.episode import run_episode  # noqa: E402
from wtm.policies import ScriptedPolicy  # noqa: E402
from wtm.render import render  # noqa: E402
from wtm.spec import load_env, load_jsonl, load_member  # noqa: E402

PLAN = {"oracle": ("A", "B", "P"), "always_approve": ("A", "P"), "always_act": ("A", "P"), "always_ask": ("B",), "kw_stuffer": ("B",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--level", default="L0r")
    ap.add_argument("--prompt", default="P1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", nargs="*", default=list(PLAN), help="subset of policies (e.g. oracle only, for label checks)")
    ns = ap.parse_args()
    scns = [s for p in ns.scenarios for f in sorted(glob.glob(p)) for s in load_jsonl(f)]
    envs, members, n = {}, {}, 0
    with open(ns.out, "w") as fh:
        for pol, twins in ((p, PLAN[p]) for p in ns.policies):
            for s in scns:
                t = "P" if "push" in s["twin"] else s["twin"][0]
                if t not in twins:
                    continue
                env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
                mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
                system, user, tools = render(env, mem, s, ns.level, ns.prompt)
                ep = run_episode(env, mem, s, ScriptedPolicy(pol, s), system, user, tools, mode="decision", level=ns.level)
                row = {"key": f"planted:{pol}|{s['id']}|{ns.level}|{ns.prompt}", "id": s["id"], "pair_id": s["pair_id"],
                       "twin": s["twin"], "env_id": s["env_id"], "model": f"planted:{pol}", "level": ns.level,
                       "prompt": ns.prompt, "rep": 0, **{k: v for k, v in ep.items() if k in ("events", "decision_grade", "full_grade")}}
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n += 1
    print(json.dumps({"rows": n, "out": ns.out}))


if __name__ == "__main__":
    main()
