"""Shared helpers for the independent verification of the 'prompt-conflicts' lens (written from scratch; does not import the
analyst's scripts). Reads the repo's own grader (wtm.reward / wtm.episode / wtm.protocol) because that IS the training reward.
Run every script with `python3 -B` (no __pycache__ written into wtm/).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

from wtm.protocol import PROTOCOL_INSTRUCTIONS, parse_action  # noqa: E402
from wtm.reward import decision_of, reward  # noqa: E402

DATA = os.path.join(ROOT, "dataset", "curated_v121")
FILES = ["sft_train", "sft_val", "dpo_train", "grpo_train"]


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def rows(fname):
    return jl(os.path.join(DATA, fname + ".jsonl"))


def specs():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset/scenarios/*.jsonl"))) + \
            sorted(glob.glob(os.path.join(ROOT, "dataset/scenarios_synth/*.jsonl"))):
        for s in jl(f):
            out[s["id"]] = s
    return out


def envs():
    return {e: json.load(open(os.path.join(ROOT, "envs", e, "env.json"))) for e in ("wallet", "bills", "shopping", "bizops")}


HINT = open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()


def segments(system: str, env: dict, level: str) -> dict:
    """Split a rendered system prompt into policy / member / tools / protocol / hint by the env template's literal pieces."""
    tmpl = env["system_prompt"]["L0" if level == "L0r" else "L1"].replace("{{today}}", "2026-11-12")
    pre = tmpl.split("{{member_block}}")[0]
    assert system.startswith(pre), "policy prefix mismatch"
    rest = system[len(pre):]
    t_i = rest.find("\n\n## Tools\n")
    p_i = rest.find(PROTOCOL_INSTRUCTIONS)
    assert t_i >= 0 and p_i > t_i
    member = rest[:t_i]
    tools = rest[t_i + 2:p_i].rstrip()
    tail = rest[p_i + len(PROTOCOL_INSTRUCTIONS):]
    return {"policy": pre, "member": member, "tools": tools, "protocol": PROTOCOL_INSTRUCTIONS, "tail": tail.strip()}


TOOL_LINE = re.compile(r"^- ([a-z_][a-z0-9_]*)\(", re.M)


def listed_tools(tools_block: str) -> list[str]:
    return TOOL_LINE.findall(tools_block)


def target_text(r: dict, role: str = None) -> str | None:
    if role:
        return r[role][0]["content"]
    for k in ("completion", "chosen"):
        if k in r:
            return r[k][0]["content"]
    return None


def n_lookups(prompt: list[dict]) -> int:
    return sum(m["role"] == "assistant" for m in prompt)


def bootstrap_diff(groups_a: dict, groups_b: dict, n=4000, seed=7):
    """Cluster bootstrap over prompts: groups_* = {prompt_key: [0/1 outcomes]}. Returns diff of pooled rates (a - b) and 95% CI."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ka, kb = list(groups_a), list(groups_b)
    def rate(keys, g):
        s = sum(sum(g[k]) for k in keys)
        t = sum(len(g[k]) for k in keys)
        return s / t if t else float("nan")
    point = rate(ka, groups_a) - rate(kb, groups_b)
    diffs = []
    for _ in range(n):
        sa = [ka[i] for i in rng.integers(0, len(ka), len(ka))]
        sb = [kb[i] for i in rng.integers(0, len(kb), len(kb))]
        diffs.append(rate(sa, groups_a) - rate(sb, groups_b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_pp": round(100 * point, 1), "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "n_prompts": [len(ka), len(kb)], "rate_a": round(rate(ka, groups_a), 3), "rate_b": round(rate(kb, groups_b), 3)}


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    return p
