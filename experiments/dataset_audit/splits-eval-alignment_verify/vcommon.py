"""Shared loaders for the splits-eval-alignment VERIFIER (independent of the analyst's common.py).

Read-only: every input lives in the repo; outputs go to experiments/dataset_audit/splits-eval-alignment_verify/results/.
Uses the repo's own wtm code (render / Simulator / protocol / grader) because that code IS the system under test.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)

from wtm.render import render, golden_text, scenario_member  # noqa: E402
from wtm.sim import Simulator  # noqa: E402
from wtm.protocol import tool_result_message  # noqa: E402

CUR = os.path.join(ROOT, "dataset", "curated_v121")


def jl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def specs():
    """seed pairs + synthetic variants (what build_datasets / make_eval_set read)."""
    return [r for p in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl")
            for f in sorted(glob.glob(os.path.join(ROOT, p))) for r in jl(f)]


def hard_specs():
    return [r for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios_hard", "*.jsonl"))) for r in jl(f)]


def eval_sets():
    """{'test': rows, 'val': rows, 'hard_test': rows, 'hard_val': rows, 'hard_train': rows} from dataset/eval/."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        b = os.path.basename(f)
        name = "_".join(b.split("_")[:2]) if b.startswith("hard_") else b.split("_")[0]
        out.setdefault(name, []).extend(jl(f))
    return out


def curated(name):
    return jl(os.path.join(CUR, f"{name}.jsonl"))


def splits():
    return json.load(open(os.path.join(ROOT, "dataset", "splits.json")))


_ENV, _MEM = {}, {}


def env(env_id):
    if env_id not in _ENV:
        _ENV[env_id] = json.load(open(os.path.join(ROOT, "envs", env_id, "env.json")))
    return _ENV[env_id]


def member(env_id, member_id):
    k = (env_id, member_id)
    if k not in _MEM:
        _MEM[k] = json.load(open(os.path.join(ROOT, "envs", env_id, "members", f"{member_id}.json")))
    return _MEM[k]


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def episode_view(s, level, prompt):
    """What the EVAL loop shows at the decision point when the policy issues exactly the canonical look-ups:
    system, first user message, then (call, result) pairs, served by the eval's Simulator (scenario_member)."""
    e = env(s["env_id"])
    m = member(s["env_id"], s["member_id"])
    system, user, tools = render(e, m, s, level, prompt)
    sim = Simulator(e, scenario_member(m, s, level), s, tools, level=level)
    msgs = [{"role": "user", "content": user}]
    for c in s.get("prefix_calls") or []:
        msgs.append({"role": "assistant", "content": call_text(c["tool"], c["arguments"])})
        msgs.append({"role": "user", "content": tool_result_message(c["tool"], sim.execute(c["tool"], c["arguments"]))})
    return system, user, tools, msgs


def target_text(r):
    return (r.get("completion") or r.get("chosen"))[0]["content"]


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    print("WROTE", os.path.relpath(p, ROOT))
    return p


ID_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("handle", re.compile(r"(?<![\w.@])@[A-Za-z0-9][\w.-]*[A-Za-z0-9_]")),
    ("upi", re.compile(r"\b[\w.-]+@(?:ok\w+|ybl|paytm|upi|ibl|axl|apl)\b")),
    ("doc_id", re.compile(r"\b[A-Z]{2,6}-[A-Z0-9]*\d[A-Z0-9-]*\b")),
    ("hash_id", re.compile(r"#\d{3,}\b")),
    ("masked", re.compile(r"(?:\*{2,}|[xX•]{2,}|ending(?: in)?\s)\s?(\d{3,})")),
    ("phone", re.compile(r"\+?\d[\d -]{7,}\d")),
]


def identifiers(text):
    found = set()
    for kind, rx in ID_PATTERNS:
        for m in rx.finditer(text or ""):
            v = (m.group(1) if rx.groups else m.group(0)).strip().rstrip(".")
            if len(v) >= 4:
                found.add((kind, v))
    return found
