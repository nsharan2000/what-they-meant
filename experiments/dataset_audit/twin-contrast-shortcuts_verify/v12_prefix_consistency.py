"""V12 (verifier): do the look-up results inside the val/train decision rows equal what the evaluation-time simulator returns
for the same canonical calls? Compares this verifier's own rendering of the val specs (cache/eval_rows.jsonl, v6) with
sft_val decision rows, and re-renders the train decision rows' specs the same way.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v12_prefix_consistency.py -> results/v12_prefix_consistency.json
"""
import sys
sys.dont_write_bytecode = True
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import HERE, T, all_specs, curated, jl, parts, save
from training.build_datasets import prefix_messages
from wtm.spec import load_env, load_member

ev = {(r["scenario_id"], r["level"]): r for r in jl(os.path.join(HERE, "cache", "eval_rows.jsonl")) if r["split"] == "val"}
out = {}
va = [r for r in curated("sft_val") if r["turn"] == "decision" and T(r) in "AB"]
same = diff = missing = 0
for r in va:
    e = ev.get((r["scenario_id"], r["level"]))
    if e is None:
        missing += 1
        continue
    if parts(e["prompt"])[3] == parts(r["prompt"])[3] and parts(e["prompt"])[2] == parts(r["prompt"])[2]:
        same += 1
    else:
        diff += 1
out["val"] = {"same": same, "differ": diff, "missing_render": missing}
specs = all_specs()
envs, mems = {}, {}
same = diff = 0
for r in curated("sft_train"):
    if r["turn"] != "decision" or T(r) not in "AB":
        continue
    s = specs[r["scenario_id"]]
    env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
    mem = mems.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
    system, user, msgs = prefix_messages(env, mem, s, r["level"], r["prompt_setting"])
    full = [{"role": "system", "content": system}] + msgs
    if full == r["prompt"]:
        same += 1
    else:
        diff += 1
out["train_full_prompt_rerender"] = {"identical": same, "differ": diff}
save(out, "v12_prefix_consistency.json")
print(out)
