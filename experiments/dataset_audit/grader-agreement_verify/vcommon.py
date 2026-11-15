"""Independent helpers for the grader-agreement VERIFICATION (written from scratch; does not import the analyst's ga_common).

Read-only over the repo. Run every script with PYTHONDONTWRITEBYTECODE=1 (no .pyc written into wtm/).
Outputs go to results/ next to this file.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)

from wtm import episode as EP  # noqa: E402
from wtm import reward as RW  # noqa: E402
from wtm import spec as SP  # noqa: E402
from wtm.labels import adjudicate  # noqa: E402
from wtm.protocol import parse_action  # noqa: E402
from wtm.render import render  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
CUR = os.path.join(ROOT, "dataset", "curated_v121")
SCORES = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")


def jl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def rows(name):
    return jl(os.path.join(CUR, f"{name}.jsonl"))


def spec_files():
    """Current spec rows from the two globs v1.2.1 was built from."""
    out = {}
    for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            for s in jl(f):
                out[s["id"]] = s
    return out


def eval_specs(prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", f"{prefix}_*.jsonl"))):
        for s in jl(f):
            out[s["id"]] = (os.path.basename(f), s)
    return out


_ENV, _MEM = {}, {}


def env(eid):
    if eid not in _ENV:
        _ENV[eid] = SP.load_env(eid)
    return _ENV[eid]


def member(eid, mid):
    if (eid, mid) not in _MEM:
        _MEM[(eid, mid)] = SP.load_member(eid, mid)
    return _MEM[(eid, mid)]


def kinds(eid):
    return {t["name"]: t["kind"] for t in env(eid)["tools"]}


def call(name, args):
    return '<tool_call>' + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + '</tool_call>'


def target_text(scn):
    a = scn.get("ideal_action") or {}
    if not a:
        return None
    return a["text"] if a["type"] == "message" else call(a["name"], a.get("arguments", {}))


def expect(scn):
    if scn["twin"].startswith("A"):
        return "A"
    return "B_act" if scn["expected"]["decision"] == "act" else "B_approve"


class Replay:
    def __init__(self, turns):
        self.turns, self.i, self.seen = list(turns), 0, []

    def chat(self, system, msgs):
        self.seen.append((system, json.loads(json.dumps(msgs))))
        t = self.turns[self.i] if self.i < len(self.turns) else ""
        self.i += 1

        class R:
            pass
        r = R()
        r.text, r.is_error, r.raw_error, r.cost_usd, r.usage = t, False, "", 0.0, {}
        return r


def eval_class(scn, level, ps, prefix_turns, target):
    """Eval path exactly as wtm/run.py does it: render -> run_episode(mode=decision, max_steps=14)."""
    e = env(scn["env_id"])
    m = member(scn["env_id"], scn["member_id"])
    system, user, tools = render(e, m, scn, level, ps)
    d = Replay(list(prefix_turns) + [target])
    out = EP.run_episode(e, m, scn, d, system, user, tools, mode="decision", max_steps=14, level=level)
    return out, d, system, user


def sha_rows(rs):
    return hashlib.sha256(json.dumps(rs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dump(obj, name):
    p = os.path.join(RES, name)
    with open(p, "w") as fh:
        if name.endswith(".jsonl"):
            for r in obj:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        else:
            json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    return p
