"""Offline tests for GRPO v2 (plain python3, no torch): reward variant regression + build_grpo_v2.py invariants.

    python3 training/tests/test_grpo_v2.py [--pristine <path to the reward.py before the v2 change>] [--data <grpo_train.jsonl>]

1. wtm.reward variant "v1" (the default) returns exactly what the pre-change reward.py returns, for every GRPO row and a set of
   canned completions covering every decision class; variant "v2" differs only for twin-A approve_informed (+0.25 -> -1.0).
2. build_grpo_v2.py: 134 same-setting cells (118 with the exclusion list, = experiments/dataset_audit/lead/results/
   grpo_cells.json), exactly steps x 4 rows, [A,B,A,B] per step, one level per step, same setting per cell, no row twice,
   rows copied byte for byte, deterministic.
Prints TEST_GRPO_V2_OK.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def load_pristine(path: str):
    src = open(path).read().replace("from .episode import", "from wtm.episode import").replace(
        "from .protocol import", "from wtm.protocol import")
    ns: dict = {"__name__": "reward_pristine"}
    exec(compile(src, path, "exec"), ns)   # noqa: S102 — our own file, test only
    return ns["reward"]


def canned(scn: dict, tool_kinds: dict) -> list[str]:
    exp = scn["expected"]
    lookups = [t for t, k in tool_kinds.items() if k in ("lookup", "verify")]
    commits = [t for t, k in tool_kinds.items() if k == "commit"]
    kw = (exp.get("informed_any") or ["the amount"])[0]
    out = [exp.get("ideal_message") or f"Before I pay, the {kw} looks different. Is that right?",
           "Shall I go ahead?", f"Quick check: is the {kw} what you expect?",
           "I have reviewed your account.", "", "<tool_call>{not json</tool_call>",
           '<tool_call>{"name": "request_approval", "arguments": {"summary": "Proceed with this payment?"}}</tool_call>',
           '<tool_call>{"name": "request_approval", "arguments": {"summary": "Proceed? Note: ' + kw + '"}}</tool_call>']
    if lookups:
        out.append('<tool_call>{"name": "' + lookups[0] + '", "arguments": {}}</tool_call>')
    for g in (scn.get("gold_commits") or [])[:1]:
        out.append("<tool_call>" + json.dumps({"name": g.get("tool") or g.get("name"), "arguments": g.get("args") or
                                               g.get("arguments") or {}}) + "</tool_call>")
    if commits:
        out.append('<tool_call>{"name": "' + commits[0] + '", "arguments": {"amount": 1}}</tool_call>')
    return out


def test_reward(pristine_path: str, data: str) -> None:
    from wtm.reward import reward
    old = load_pristine(pristine_path)
    n = n_diff = 0
    classes = set()
    for line in open(data):
        r = json.loads(line)
        scn, tk = json.loads(r["scenario"]), json.loads(r["tool_kinds"])
        for c in canned(scn, tk):
            a, pa = old(c, scn, tk)
            b, pb = reward(c, scn, tk)                  # default = v1
            b1, _ = reward(c, scn, tk, variant="v1")
            v2, p2 = reward(c, scn, tk, variant="v2")
            assert (a, pa) == (b, pb) == (b1, pb), (r["scenario_id"], c[:60], a, b)
            classes.add(pa["decision"])
            if v2 != a:
                n_diff += 1
                assert scn["twin"].startswith("A") and pa["decision"] == "approve_informed", (r["scenario_id"], pa)
                assert p2["base"] == -1.0 and pa["base"] == 0.25, (pa, p2)
            elif scn["twin"].startswith("A") and pa["decision"] == "approve_informed":
                raise AssertionError("v2 must change twin-A approve_informed")
            n += 1
    try:
        reward("x", scn, tk, variant="v3")
        raise AssertionError("unknown variant accepted")
    except ValueError:
        pass
    print(f"reward: {n} (row, completion) cases, v1 == pristine on all; v2 differs on {n_diff} (all twin-A approve_informed); "
          f"classes seen {sorted(classes)}")
    assert n_diff > 0 and {"ask_informed", "approve_generic", "approve_informed", "lookup", "invalid"} <= classes


def run_builder(data: str, out: str, *extra: str) -> dict:
    cmd = [sys.executable, os.path.join(ROOT, "training", "build_grpo_v2.py"), "--data", data, "--out", out, "--steps", "36",
           *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "GRPO_V2_DATA_OK" in p.stdout, p.stdout + p.stderr
    return json.load(open(os.path.join(out, "manifest.json")))


def test_builder(data: str) -> None:
    src = [l for l in open(data).read().split("\n") if l.strip()]
    excl = os.path.join(ROOT, "experiments", "dataset_audit", "lead", "results", "exclusion_list.json")
    with tempfile.TemporaryDirectory() as td:
        m1 = run_builder(data, os.path.join(td, "a"))
        m2 = run_builder(data, os.path.join(td, "b"))
        mx = run_builder(data, os.path.join(td, "x"), "--exclude", excl)
        assert m1["output_sha256"] == m2["output_sha256"], "not deterministic"
        assert m1["same_setting_cells"] == 134 and m1["eligible_cells"] == 134, m1["eligible_cells"]
        assert mx["eligible_cells"] == 118, mx["eligible_cells"]
        flagged = {x["line"] for x in json.load(open(excl))["grpo_train"]["rows"]}
        for m, d in ((m1, "a"), (mx, "x")):
            out = [l for l in open(os.path.join(td, d, "grpo_train.jsonl")).read().split("\n") if l.strip()]
            assert len(out) == 144
            used = [i for s in m["step_composition"] for i in s["src_lines"]]
            assert len(used) == len(set(used)) == 144
            assert [src[i] for i in used] == out, "rows must be copied byte for byte, in step order"
            for k in range(0, 144, 4):
                rs = [json.loads(x) for x in out[k:k + 4]]
                assert [r["twin"][0] for r in rs] == ["A", "B", "A", "B"]
                assert len({r["level"] for r in rs}) == 1
                assert rs[0]["pair_id"] == rs[1]["pair_id"] != rs[2]["pair_id"] == rs[3]["pair_id"]
                assert rs[0]["prompt_setting"] == rs[1]["prompt_setting"] and rs[2]["prompt_setting"] == rs[3]["prompt_setting"]
            if d == "x":
                assert not (set(used) & flagged), "an excluded row was used"
        print(f"builder: deterministic sha {m1['output_sha256'][:12]}; cells 134 / 118 excl; steps per level "
              f"{m1['steps_per_level']} / {mx['steps_per_level']}; distinct pairs {m1['distinct_pairs']} / {mx['distinct_pairs']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", default=os.environ.get("WTM_REWARD_PRISTINE"),
                    help="reward.py as it was before the variant change (oracle for v1)")
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl"))
    ns = ap.parse_args()
    if ns.pristine:
        test_reward(ns.pristine, ns.data)
    else:
        print("reward regression SKIPPED (no --pristine given)")
    test_builder(ns.data)
    print("TEST_GRPO_V2_OK")


if __name__ == "__main__":
    main()
