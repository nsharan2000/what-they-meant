"""Shared loaders for the composition-memorization audit (read-only on every input; stdlib + numpy/pandas only)."""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.protocol import parse_action  # noqa: E402
from wtm.reward import reward  # noqa: E402

DATA = os.path.join(ROOT, "dataset", "curated_v121")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TOKEN_STATS = os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")
SFT_RESULT = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m1_sft", "result.json")
GRPO_SCORES = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")
GRPO_RESULT = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "result.json")
os.makedirs(OUT, exist_ok=True)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_view(name):
    return load_jsonl(os.path.join(DATA, f"{name}.jsonl"))


def target_of(r):
    side = r.get("completion") or r.get("chosen")
    return side[0]["content"]


def scenarios():
    out = {}
    for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            for s in load_jsonl(f):
                out[s["id"]] = s
    return out


_KINDS = {}


def tool_kinds(env_id):
    if env_id not in _KINDS:
        env = json.load(open(os.path.join(ROOT, "envs", env_id, "env.json")))
        _KINDS[env_id] = {t["name"]: t["kind"] for t in env["tools"]}
    return _KINDS[env_id]


def source_of(scenario_id: str) -> str:
    """S..A/B = seed pair twin (user-authored target), S..P = pushback row, V... = synthetic variant (open-teacher target)."""
    if scenario_id.startswith("V"):
        return "variant"
    if scenario_id.endswith("P"):
        return "pushback"
    return "seed"


def target_kind(text: str, env_id: str) -> str:
    """message | lookup:<tool> | verify:<tool> | ui:<tool> | commit:<tool> | invalid (wtm.protocol parser, env tool kinds)."""
    kinds = tool_kinds(env_id)
    a = parse_action(text, set(kinds))
    if a.kind == "message":
        return "message"
    if a.kind == "tool_call":
        return f"{kinds.get(a.name, 'unknown')}:{a.name}"
    return "invalid"


def grade(text: str, scn: dict, env_id: str):
    """(reward, components) from wtm.reward, i.e. the GRPO reward / evaluation grader applied to a target."""
    return reward(text, scn, tool_kinds(env_id))


WORD_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)*|[^\sA-Za-z\d]")


def words(text: str):
    return WORD_RE.findall(text.lower())


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def dump(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    print(f"WROTE {p}")
    return p
