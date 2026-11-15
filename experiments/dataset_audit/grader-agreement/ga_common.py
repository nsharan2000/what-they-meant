"""Shared loaders for the grader-agreement audit (read-only over the repo; outputs go to results/ next to this file).

Reproduces the two scoring paths exactly:
  * GRPO reward  : wtm.reward.reward(completion, scenario, tool_kinds) with tool_kinds = {tool.name: tool.kind for env tools}
                   (training/build_datasets.py line 172 / training/train_grpo.py decision_reward).
  * eval grader  : wtm.run -> render(env, member, scn, level, prompt) -> wtm.episode.run_episode(..., mode="decision") with a
                   replay driver that plays the row's own look-up calls and then the target text, so the Simulator, the allowed
                   tool set of the level, narration handling and commit-failure handling are all the real ones.
Run every script with PYTHONDONTWRITEBYTECODE=1 so no __pycache__ is written into wtm/.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wtm.episode import run_episode, decision_grade, classify_decision, is_question, is_narration  # noqa: E402
from wtm.labels import adjudicate, adjudications  # noqa: E402
from wtm.render import render  # noqa: E402
from wtm.reward import reward  # noqa: E402
from wtm.spec import load_env, load_member, any_hit, kw_hit, norm  # noqa: E402

DATA = os.path.join(ROOT, "dataset", "curated_v121")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
FILES = ("sft_train", "sft_val", "dpo_train", "grpo_train")


def load_rows(name: str) -> list[dict]:
    with open(os.path.join(DATA, f"{name}.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_specs() -> dict:
    """Current scenario specs (the two globs build_datasets.py v1.2.1 was run with)."""
    out = {}
    for p in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
        for f in sorted(glob.glob(os.path.join(ROOT, p))):
            for l in open(f):
                if l.strip():
                    s = json.loads(l)
                    out[s["id"]] = s
    return out


def trained_specs(specs: dict) -> tuple[dict, dict]:
    """Scenario as trained = the JSON embedded in grpo_train rows (what the GRPO reward read). Falls back to the current spec
    for scenarios without a GRPO row (validation pairs, pushback rows). Returns (spec_by_id, source_by_id)."""
    emb, src = {}, {}
    for r in load_rows("grpo_train"):
        s = json.loads(r["scenario"])
        emb.setdefault(s["id"], s)
    out = {}
    for sid, s in specs.items():
        out[sid] = emb.get(sid, s)
        src[sid] = "grpo_embedded" if sid in emb else "current_spec_file"
    for sid, s in emb.items():
        if sid not in out:
            out[sid], src[sid] = s, "grpo_embedded_only"
    return out, src


_ENVS, _MEMBERS = {}, {}


def env_of(env_id: str) -> dict:
    if env_id not in _ENVS:
        _ENVS[env_id] = load_env(env_id)
    return _ENVS[env_id]


def member_of(env_id: str, member_id: str) -> dict:
    k = (env_id, member_id)
    if k not in _MEMBERS:
        _MEMBERS[k] = load_member(env_id, member_id)
    return _MEMBERS[k]


def tool_kinds_of(env_id: str) -> dict:
    return {t["name"]: t["kind"] for t in env_of(env_id)["tools"]}


@dataclass
class _Res:
    text: str
    is_error: bool = False
    raw_error: str = ""
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)


class ReplayDriver:
    """Plays fixed assistant turns in order; after they run out it returns '' (parsed as invalid/empty)."""

    def __init__(self, turns: list[str]):
        self.turns, self.i, self.seen = list(turns), 0, []

    def chat(self, system, msgs):
        self.seen.append((system, [dict(m) for m in msgs]))
        t = self.turns[self.i] if self.i < len(self.turns) else ""
        self.i += 1
        return _Res(text=t)


def eval_replay(scn: dict, level: str, prompt_setting: str, prefix_turns: list[str], target: str) -> dict:
    """Run the evaluation episode exactly like wtm/run.py (render + run_episode, decision mode, max_steps 14) with the row's
    look-up turns followed by `target`. Returns the decision grade plus how the TARGET turn itself was handled."""
    env, mem = env_of(scn["env_id"]), member_of(scn["env_id"], scn["member_id"])
    system, first_user, tools = render(env, mem, scn, level, prompt_setting)
    drv = ReplayDriver(prefix_turns + [target])
    res = run_episode(env, mem, scn, drv, system, first_user, tools, mode="decision", max_steps=14, level=level)
    k = len(prefix_turns)
    ev = res["events"][k] if len(res["events"]) > k else {}
    # which event set the decision? (decision is set once, at the first ui/commit tool call or non-narration message)
    dec_idx = None
    for i, e in enumerate(res["events"]):
        if e.get("kind") == "tool_call" and e.get("tool_kind") in ("ui", "commit"):
            dec_idx = i
            break
        if e.get("kind") == "message" and not e.get("narration"):
            dec_idx = i
            break
    target_kind = ev.get("kind")
    if target_kind == "tool_call":
        target_kind = f"tool_call:{ev.get('tool_kind')}"
    elif target_kind == "message" and ev.get("narration"):
        target_kind = "message:narration"
    # parity: the conversation the eval shows before the target vs what training showed
    seen_system, seen_msgs = drv.seen[k] if len(drv.seen) > k else (None, None)
    return {"eval_decision": res["decision_grade"]["decision"], "eval_grade": res["decision_grade"],
            "target_event_kind": target_kind, "target_is_decision": dec_idx == k,
            "prefix_tool_kinds": [e.get("tool_kind") for e in res["events"][:k]],
            "eval_system": seen_system, "eval_msgs_before_target": seen_msgs, "allowed_tools": tools}


def top_class_ok(scn: dict, decision: str) -> bool:
    """The task's definition of 'top class for its twin' (A: ask_informed; B approve: approve_*; B act: commit_gold)."""
    if scn["twin"].startswith("A"):
        return decision == "ask_informed"
    if scn["expected"]["decision"] == "act":
        return decision == "commit_gold"
    return decision.startswith("approve")


def expected_kind(scn: dict) -> str:
    if scn["twin"].startswith("A"):
        return "A"
    return "B_act" if scn["expected"]["decision"] == "act" else "B_approve"


def dump_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
