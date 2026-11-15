"""Shared loaders for the prompt-conflicts lens (read-only over the repo; writes only under this folder).

Run every script with `python3 -B` (no __pycache__ is written into the repo's wtm/ package).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from wtm.protocol import PROTOCOL_INSTRUCTIONS, parse_action  # noqa: E402,F401
from wtm.episode import classify_decision, is_question  # noqa: E402,F401
from wtm.reward import reward, decision_of  # noqa: E402,F401
from wtm.spec import any_hit  # noqa: E402,F401

DATA = os.path.join(ROOT, "dataset", "curated_v121")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
FILES = ("sft_train", "sft_val", "dpo_train", "grpo_train")
TODAY = "2026-11-12"
HINT = open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()
ENVS = ("wallet", "bills", "shopping", "bizops")


def load_rows(name: str) -> list[dict]:
    with open(os.path.join(DATA, f"{name}.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_specs() -> dict:
    out = {}
    for p in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
        for f in sorted(glob.glob(os.path.join(ROOT, p))):
            for l in open(f):
                if l.strip():
                    s = json.loads(l)
                    out[s["id"]] = s
    return out


def load_envs() -> dict:
    return {e: json.load(open(os.path.join(ROOT, "envs", e, "env.json"))) for e in ENVS}


def tool_kinds(env: dict) -> dict:
    return {t["name"]: t["kind"] for t in env["tools"]}


def policy_prefix(env: dict, level: str) -> str:
    key = "L0" if level.startswith("L0") else "L1"
    return env["system_prompt"][key].split("{{member_block}}")[0].replace("{{today}}", TODAY)


def split_system(text: str, env: dict, level: str) -> dict:
    """Split a rendered system prompt into policy / member / tools / protocol / tail (P2 hint or nothing).

    Raises if the policy prefix or the protocol block is not found verbatim (the rendering contract)."""
    pre = policy_prefix(env, level)
    if not text.startswith(pre):
        raise ValueError("policy prefix mismatch")
    rest = text[len(pre):]
    ti = rest.find("\n\n## Tools\n")
    pi = rest.find("\n\n## How to act\n")
    if ti < 0 or pi < 0 or pi < ti:
        raise ValueError("tools/protocol markers not found")
    member = rest[:ti]
    tools = rest[ti + 2:pi]
    after = rest[pi + 2:]
    if not after.startswith(PROTOCOL_INSTRUCTIONS):
        raise ValueError("protocol block differs from wtm.protocol.PROTOCOL_INSTRUCTIONS")
    tail = after[len(PROTOCOL_INSTRUCTIONS):].strip()
    return {"policy": pre, "member": member, "tools": tools, "protocol": PROTOCOL_INSTRUCTIONS, "tail": tail,
            "offsets": {"policy": 0, "member": len(pre), "tools": len(pre) + ti + 2, "protocol": len(pre) + pi + 2,
                        "tail": len(pre) + pi + 2 + len(PROTOCOL_INSTRUCTIONS), "end": len(text)}}


TOOL_DOC_RE = re.compile(r"^- ([a-z_][a-z0-9_]*)\(", re.M)


def tools_in_docs(tools_block: str) -> list[str]:
    return TOOL_DOC_RE.findall(tools_block)


def target_text(r: dict, key: str = "completion") -> str:
    return r[key][0]["content"]


def n_lookups(prompt: list[dict]) -> int:
    return sum(1 for m in prompt if m["role"] == "assistant" and "<tool_call>" in m["content"])


def dump(name: str, obj) -> str:
    path = os.path.join(RES, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    return path


def dump_jsonl(name: str, rows: list[dict]) -> str:
    path = os.path.join(RES, name)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)
