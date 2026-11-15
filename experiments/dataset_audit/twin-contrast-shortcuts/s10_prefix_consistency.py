"""Step 10 (negative check): training decision prompts carry look-up results produced by training/build_datasets.prefix_messages,
which runs the simulator on the RAW member; evaluation (wtm.episode.run_episode) runs it on scenario_member(member, scn, level)
(+ scenario_records, + l2_extra_records at L2/L2s/L3). Do the trained look-up results equal what the evaluation simulator returns
for the same calls? (A mismatch would make the twin evidence differ between training and evaluation.)
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s10_prefix_consistency.py -> results/s10_prefix_consistency.json
"""
import sys
sys.dont_write_bytecode = True
import json, os  # noqa: E401,E402
from collections import Counter  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, jdump, load_rows, load_specs  # noqa: E402
from wtm.protocol import tool_result_message  # noqa: E402
from wtm.render import render, scenario_member  # noqa: E402
from wtm.sim import Simulator  # noqa: E402
from wtm.spec import load_env, load_member  # noqa: E402


def main():
    specs, _ = load_specs()
    envs, mems = {}, {}
    out = {}
    for f in ("sft_train", "sft_val"):
        c, bad = Counter(), []
        for r in load_rows(f):
            if r["turn"] != "decision":
                continue
            s = specs[r["scenario_id"]]
            env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
            mem = mems.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
            _, _, tools = render(env, mem, s, r["level"], r["prompt_setting"])
            sim = Simulator(env, scenario_member(mem, s, r["level"]), s, tools, level=r["level"])
            got = [tool_result_message(cl["tool"], sim.execute(cl["tool"], cl["arguments"])) for cl in s.get("prefix_calls", [])]
            trained = [m["content"] for m in r["prompt"][3::2]]
            same = got == trained
            c["identical" if same else "different"] += 1
            if not same:
                bad.append([f, r["_i"], r["scenario_id"], r["level"]])
        out[f] = {"decision_rows": dict(c), "different_examples": bad[:10]}
    jdump(out, os.path.join(RES, "s10_prefix_consistency.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
