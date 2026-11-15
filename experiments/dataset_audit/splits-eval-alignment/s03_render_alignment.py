"""3-4. Are evaluation prompts built exactly like training prompts?

(a) Re-render EVERY training row (sft_train, sft_val, dpo_train, grpo_train) from the current specs with wtm/render.py at the row's
    level / prompt_setting and rebuild its look-up prefix exactly as training/build_datasets.py prefix_messages() does; compare
    message by message (system, request, look-up calls, results) and the target.
(b) Replay an ORACLE driver (canonical look-ups, then the ideal action) through the evaluation loop (wtm.episode.run_episode, the
    function wtm/run.py calls) and compare the eval-time decision context with the training decision row (exact string equality).
(c) Oracle replay on the evaluation scenario files at the evaluated levels/prompts (run_eval.sh: L0r L1 L2 L2s x P0 P2): decision
    class, raw success (run.py has no adjudication), adjudicated success, and look-up results that are errors / empty.
(d) System-prompt anatomy per level: date line, number of tools, hint placement, protocol block, 'Most relevant records'.

python3 -B experiments/dataset_audit/splits-eval-alignment/s03_render_alignment.py -> results/s03_render_alignment.json
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter, defaultdict

from common import dump, eval_files, scen_files, views
from wtm.episode import decision_grade, run_episode
from wtm.labels import adjudicate
from wtm.protocol import PROTOCOL_INSTRUCTIONS, tool_result_message
from wtm.render import TODAY, golden_text, render
from wtm.sim import Simulator
from wtm.spec import load_env, load_member

by_id = {r["id"]: r for r in scen_files("seed") + scen_files("synth") + scen_files("hard")}
envs, members = {}, {}


def env_mem(s):
    env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
    mem = members.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
    return env, mem


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def action_text(a):
    return a["text"] if a["type"] == "message" else call_text(a["name"], a.get("arguments", {}))


def prefix_messages(env, member, scn, level, prompt):   # = training/build_datasets.py prefix_messages
    system, user, tools = render(env, member, scn, level, prompt)
    sim = Simulator(env, member, scn, tools, level=level)
    msgs = [{"role": "user", "content": user}]
    for c in scn.get("prefix_calls", []):
        msgs.append({"role": "assistant", "content": call_text(c["tool"], c["arguments"])})
        msgs.append({"role": "user", "content": tool_result_message(c["tool"], sim.execute(c["tool"], c["arguments"]))})
    return system, user, msgs, tools


class Res:
    def __init__(self, text):
        self.text, self.is_error, self.usage, self.cost_usd, self.raw_error = text, False, {}, 0.0, ""


class Oracle:
    """Canonical look-ups (scenario.prefix_calls), then the ideal action — the training targets, replayed through the eval loop."""

    def __init__(self, scn):
        self.s = scn

    def chat(self, system, msgs):
        k = sum(m["role"] == "assistant" for m in msgs)
        pcs = self.s.get("prefix_calls") or []
        return Res(call_text(pcs[k]["tool"], pcs[k]["arguments"]) if k < len(pcs) else action_text(self.s["ideal_action"]))


def first_diff(a, b, ctx=60):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            return {"op": tag, "train": a[max(0, i1 - ctx): i2 + ctx], "rerender": b[max(0, j1 - ctx): j2 + ctx]}
    return None


out = {}
# ---------------------------------------------------------------- (a) training rows vs re-render, (b) vs oracle eval replay
V = views()
cmp = defaultdict(Counter)
examples = defaultdict(list)
oracle_cmp = Counter()
oracle_examples = []
cache = {}
for v, rows in V.items():
    for i, r in enumerate(rows):
        s = by_id[r["scenario_id"]]
        env, mem = env_mem(s)
        key = (s["id"], r["level"], r["prompt_setting"])
        if key not in cache:
            cache[key] = prefix_messages(env, mem, s, r["level"], r["prompt_setting"])
        system, user, msgs, tools = cache[key]
        turn = r.get("turn", "decision")
        if turn == "decision":
            exp_prompt = [{"role": "system", "content": system}] + msgs
            exp_target = action_text(s["ideal_action"])
        else:
            k = int(turn.replace("lookup", ""))
            exp_prompt = [{"role": "system", "content": system}] + msgs[:1 + 2 * (k - 1)]
            pc = s["prefix_calls"][k - 1]
            exp_target = call_text(pc["tool"], pc["arguments"])
        got = r["prompt"]
        same_len = len(got) == len(exp_prompt)
        per_msg = [same_len and got[j]["role"] == exp_prompt[j]["role"] and got[j]["content"] == exp_prompt[j]["content"]
                   for j in range(min(len(got), len(exp_prompt)))]
        ok_sys = bool(per_msg) and per_msg[0]
        ok_rest = same_len and all(per_msg[1:])
        tgt = (r.get("completion") or r.get("chosen"))
        ok_tgt = True if v == "grpo_train" else tgt[0]["content"] == exp_target
        cmp[v]["rows"] += 1
        cmp[v]["system_identical"] += ok_sys
        cmp[v]["messages_after_system_identical"] += ok_rest
        cmp[v]["target_identical"] += ok_tgt
        if not (ok_sys and ok_rest and ok_tgt) and len(examples[v]) < 5:
            d = first_diff(got[0]["content"], exp_prompt[0]["content"]) if not ok_sys else None
            examples[v].append({"i": i, "scenario_id": s["id"], "level": r["level"], "prompt_setting": r["prompt_setting"],
                                "turn": turn, "system_identical": ok_sys, "rest_identical": ok_rest, "target_identical": ok_tgt,
                                "system_diff": d})
        # (b) the eval loop with an oracle, decision rows only (once per scenario/level/prompt)
        if turn == "decision" and v in ("sft_train", "grpo_train", "sft_val"):
            res = run_episode(env, mem, s, Oracle(s), system, user, tools, mode="decision", max_steps=14, level=r["level"])
            ev_prompt = [{"role": "system", "content": system}] + res["messages"][:-1]
            same = ev_prompt == got
            oracle_cmp[f"{v}:decision_context_identical"] += same
            oracle_cmp[f"{v}:n"] += 1
            if not same and len(oracle_examples) < 5:
                oracle_examples.append({"view": v, "i": i, "scenario_id": s["id"], "level": r["level"],
                                        "n_msgs_train": len(got), "n_msgs_eval": len(ev_prompt),
                                        "diff": first_diff(json.dumps(got, ensure_ascii=False), json.dumps(ev_prompt, ensure_ascii=False))})
out["a_training_rows_vs_rerender"] = {k: dict(v) for k, v in cmp.items()}
out["a_examples_of_mismatch"] = examples
out["b_oracle_eval_replay_vs_training_decision_rows"] = dict(oracle_cmp)
out["b_examples_of_mismatch"] = oracle_examples

# ---------------------------------------------------------------- (c) oracle on evaluation scenario files at evaluated levels
EVAL_LEVELS, EVAL_PROMPTS = ["L0r", "L1", "L2", "L2s"], ["P0", "P2"]
ev = eval_files()
grid = defaultdict(Counter)
fails = []
empty_results = []
for name, rows in ev.items():
    group = name.rsplit("_", 1)[0]
    for s in rows:
        env, mem = env_mem(s)
        for lv in EVAL_LEVELS:
            for p in EVAL_PROMPTS:
                system, user, tools = render(env, mem, s, lv, p)
                res = run_episode(env, mem, s, Oracle(s), system, user, tools, mode="decision", max_steps=14, level=lv)
                g = res["decision_grade"]
                g_adj = decision_grade(adjudicate(s), g["decision"])
                for e in res["events"]:
                    if e.get("tool_kind") in ("lookup", "verify"):
                        rtxt = json.dumps(e.get("result"), ensure_ascii=False)
                        if "error" in rtxt.lower()[:40] or "no matching records" in rtxt.lower():
                            empty_results.append([name, s["id"], lv, p, e.get("tool"), rtxt[:160]])
                kind = "A" if s["twin"].startswith("A") else "B"
                grid[(group, lv, p)][f"{kind}_n"] += 1
                grid[(group, lv, p)][f"{kind}_strict_raw"] += bool(g["success"])
                grid[(group, lv, p)][f"{kind}_strict_adjudicated"] += bool(g_adj["success"])
                if not g["success"] or not g_adj["success"]:
                    fails.append([name, s["id"], lv, p, g["decision"], bool(g["success"]), bool(g_adj["success"]),
                                  s["expected"].get("decision")])
out["c_oracle_eval_grid"] = {"|".join(k): dict(v) for k, v in sorted(grid.items())}
out["c_oracle_failures"] = fails
out["c_oracle_failures_distinct_ids"] = sorted({(f[0], f[1], f[4], f[5], f[6], f[7]) for f in fails})
out["c_lookup_results_error_or_empty"] = empty_results[:40]
out["c_n_lookup_results_error_or_empty"] = len(empty_results)

# ---------------------------------------------------------------- (d) anatomy of the system prompt per level / prompt
hint = golden_text("P2")
anat = defaultdict(Counter)
DATE_RE = re.compile(r"Today is (\d{4}-\d{2}-\d{2})")
for v, rows in V.items():
    for r in rows:
        sysm = r["prompt"][0]["content"]
        lv, p = r["level"], r["prompt_setting"]
        a = anat[f"{v}|{lv}|{p}"]
        a["n"] += 1
        a["has_today_line"] += bool(DATE_RE.search(sysm))
        a["today_is_" + (DATE_RE.search(sysm).group(1) if DATE_RE.search(sysm) else "none")] += 1
        a["has_protocol_block"] += PROTOCOL_INSTRUCTIONS in sysm
        a["has_hint"] += hint in sysm
        a["hint_is_last_text"] += sysm.rstrip().endswith(hint)
        a["has_most_relevant_records"] += "## Most relevant records" in sysm
        a["n_tools_" + str(sysm.split("## Tools", 1)[-1].split("## How to act")[0].count("\n- "))] += 1
for name, rows in ev.items():
    group = name.rsplit("_", 1)[0]
    for s in rows:
        env, mem = env_mem(s)
        for lv in EVAL_LEVELS:
            for p in EVAL_PROMPTS:
                sysm, user, tools = render(env, mem, s, lv, p)
                a = anat[f"eval:{group}|{lv}|{p}"]
                a["n"] += 1
                a["has_today_line"] += bool(DATE_RE.search(sysm))
                a["has_protocol_block"] += PROTOCOL_INSTRUCTIONS in sysm
                a["has_hint"] += hint in sysm
                a["hint_is_last_text"] += sysm.rstrip().endswith(hint)
                a["has_most_relevant_records"] += "## Most relevant records" in sysm
                a["n_tools_" + str(len(tools))] += 1
out["d_anatomy"] = {k: dict(v) for k, v in sorted(anat.items())}
out["d_today_constant"] = TODAY
out["d_hint_text"] = hint
print(json.dumps({k: v for k, v in out.items() if k.startswith(("a_", "b_"))}, indent=1, ensure_ascii=False)[:5000])
print(json.dumps(out["c_oracle_eval_grid"], indent=0)[:3000])
print("oracle failures:", len(fails), out["c_oracle_failures_distinct_ids"][:20])
print("lookup error/empty:", len(empty_results), empty_results[:8])
print(dump("s03_render_alignment.json", out))
