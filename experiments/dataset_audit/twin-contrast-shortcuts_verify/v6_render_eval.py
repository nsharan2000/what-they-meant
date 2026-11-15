"""V6 (verifier): render the evaluation scenario files at the training levels (L0r/L1/L2, + L3 when a noisy message exists), P0,
with the canonical look-ups executed by the simulator (training.build_datasets.prefix_messages = how training rows are built).
This is the decision-point view the evaluated model would see IF it makes the canonical look-up (the evaluation itself lets
the model choose its look-ups, so this is an approximation). Output goes to this verifier's cache only.

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v6_render_eval.py -> cache/eval_rows.jsonl
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import HERE, T, eval_specs  # noqa: E402

from training.build_datasets import prefix_messages  # noqa: E402
from wtm.spec import load_env, load_member  # noqa: E402


def main():
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    path = os.path.join(HERE, "cache", "eval_rows.jsonl")
    envs, mems = {}, {}
    n = 0
    with open(path, "w") as fh:
        for name in ("test", "val", "hard_test", "hard_val", "hard_train"):
            for s in eval_specs(name):
                if T(s) == "P":
                    continue
                env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
                mem = mems.setdefault((s["env_id"], s["member_id"]), load_member(s["env_id"], s["member_id"]))
                for lv in ("L0r", "L1", "L2", "L3"):
                    if lv == "L3" and not s.get("user_message_noisy"):
                        continue
                    system, user, msgs = prefix_messages(env, mem, s, lv, "P0")
                    fh.write(json.dumps({"split": name, "scenario_id": s["id"], "pair_id": s["pair_id"], "twin": s["twin"],
                                         "env_id": s["env_id"], "level": lv, "prompt_setting": "P0", "turn": "decision",
                                         "cluster": (s.get("source") or {}).get("post_id", s["pair_id"]),
                                         "prompt": [{"role": "system", "content": system}] + msgs}, ensure_ascii=False) + "\n")
                    n += 1
    print("rows", n, path)


if __name__ == "__main__":
    main()
