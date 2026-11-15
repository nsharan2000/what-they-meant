"""Verifier (dpo-pairs lens), step 1: independent replay of every DPO pair's rejected side + decision classes.

Reads only (never writes outside experiments/dataset_audit/dpo-pairs_verify/):
  dataset/curated_v121/dpo_train.jsonl, dataset/scenarios*/..., dataset/label_adjudications.json,
  experiments/replication/results/grid_*.jsonl (the failure pool the build reads), envs/*/env.json
Re-implements the rejected-side rule of training/build_datasets.py (load_failures + reward <= 0, else synthetic) but keeps
provenance (file, line, model, grid prompt, tool_results, position of the event in its episode).
Output: results/v1_rows.jsonl (one line per DPO row, same order as the file) and results/v1_summary.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.reward import reward, decision_of  # noqa: E402
from wtm.episode import decision_grade  # noqa: E402
from wtm.labels import adjudicate  # noqa: E402
from wtm.protocol import parse_action  # noqa: E402
from wtm.spec import load_env  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
BASE_PREFIX = ("hf:CohereLabs", "vllm:tiny")   # as in training/build_datasets.py BASE_MODELS


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


dpo = jl(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))
scns = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        for s in jl(f):
            scns[s["id"]] = s
envs = {}


def kinds_of(env_id):
    if env_id not in envs:
        envs[env_id] = {t["name"]: t["kind"] for t in load_env(env_id)["tools"]}
    return envs[env_id]


# ---- failure pool with provenance (sorted file order, file line order) ----
pool = defaultdict(list)
files = sorted(glob.glob(os.path.join(ROOT, "experiments/replication/results/grid_*.jsonl")))
for f in files:
    for li, line in enumerate(open(f)):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["decision_grade"]["success"] or not r.get("events") or not r["model"].startswith(BASE_PREFIX):
            continue
        ev = r["events"]
        last = ev[-1].get("raw")
        if not last:
            continue
        prev = ev[:-1]
        pool[(r["id"], r["level"])].append({
            "raw": last, "file": os.path.basename(f), "line": li, "model": r["model"].split("@")[0],
            "grid_prompt": r["prompt"], "tool_results": r.get("tool_results"), "n_events": len(ev),
            "prev_kinds": [e.get("kind") + (":" + e.get("tool", "") if e.get("kind") == "tool_call" else "") for e in prev],
            "prev_calls": [[e.get("tool"), e.get("args")] for e in prev if e.get("kind") == "tool_call"],
            "prev_results_err": [bool(isinstance(e.get("result"), dict) and e["result"].get("error")) or
                                 (isinstance(e.get("result"), str) and "error" in e.get("result", "").lower()[:40])
                                 for e in prev if e.get("kind") == "tool_call"],
            "grid_decision": r["decision_grade"]["decision"],
        })


def synth_text(s):
    # copy of training/build_datasets.py synthetic_rejected (re-typed, to check the build's two templates)
    if s["twin"].startswith("A"):
        g = (s.get("gold_commits") or [{}])[0]
        if g:
            args = {"summary": f"Proceed with your request: {s['user_message']}"}
            return '<tool_call>' + json.dumps({"name": "request_approval", "arguments": args}, ensure_ascii=False) + '</tool_call>'
        return "Done — I've taken care of that for you."
    return "Before I go ahead, can you confirm this is definitely what you want to do?"


rows = []
for i, d in enumerate(dpo):
    s = scns[d["scenario_id"]]
    k = kinds_of(s["env_id"])
    chosen = d["chosen"][0]["content"]
    rej = d["rejected"][0]["content"]
    cands = pool.get((d["scenario_id"], d["level"]), [])
    pick = next((c for c in cands if reward(c["raw"], s, k)[0] <= 0), None)
    if pick is not None:
        src, exp_rej = "observed", pick["raw"]
    else:
        src, exp_rej = ("synthetic_after_failures" if cands else "synthetic_no_failure"), synth_text(s)
    cr, cinfo = reward(chosen, s, k)
    rr, rinfo = reward(rej, s, k)
    sa = adjudicate(s)
    g_ch = decision_grade(sa, cinfo["decision"])
    g_rj = decision_grade(sa, rinfo["decision"])
    pa_r = parse_action(rej, set(k))
    pa_c = parse_action(chosen, set(k))
    canon = [[c["tool"], c["arguments"]] for c in s.get("prefix_calls") or []]
    rows.append({
        "idx": i, "scenario_id": d["scenario_id"], "pair_id": d["pair_id"], "twin": d["twin"][:1], "env_id": d["env_id"],
        "level": d["level"], "prompt_setting": d["prompt_setting"], "expected": s["expected"]["decision"],
        "replay_matches": exp_rej == rej, "rej_source": src,
        "prov": ({kk: vv for kk, vv in pick.items() if kk != "raw"} if pick else None),
        "prev_equals_canonical": (pick["prev_calls"] == canon) if pick else None,
        "n_candidates": len(cands),
        "chosen_reward": cr, "chosen_dec": cinfo["decision"], "chosen_strict_fmt": cinfo["format"] > 0,
        "rej_reward": rr, "rej_dec": rinfo["decision"], "rej_strict_fmt": rinfo["format"] > 0,
        "chosen_eval_strict": g_ch["success"], "chosen_eval_lenient": g_ch["success_lenient"],
        "rej_eval_strict": g_rj["success"], "rej_eval_lenient": g_rj["success_lenient"],
        "adjudicated": sa is not s,
        "rej_parse_kind": pa_r.kind, "rej_parse_name": pa_r.name, "rej_flags": pa_r.flags,
        "chosen_parse_kind": pa_c.kind, "chosen_parse_name": pa_c.name, "chosen_flags": pa_c.flags,
        "rej_tool_kind": k.get(pa_r.name) if pa_r.kind == "tool_call" else None,
        "chosen_tool_kind": k.get(pa_c.name) if pa_c.kind == "tool_call" else None,
        "chosen_text": chosen, "rej_text": rej,
    })

with open(os.path.join(OUT, "v1_rows.jsonl"), "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

C = Counter
summ = {
    "n": len(rows),
    "replay_matches": sum(r["replay_matches"] for r in rows),
    "replay_mismatch_idx": [r["idx"] for r in rows if not r["replay_matches"]],
    "rej_source": dict(C(r["rej_source"] for r in rows)),
    "chosen_reward": dict(C(r["chosen_reward"] for r in rows)),
    "rej_reward_max": max(r["rej_reward"] for r in rows), "rej_reward_min": min(r["rej_reward"] for r in rows),
    "chosen_dec_by_twin_expected": dict(C(f"{r['twin']}/{r['expected']}/{r['chosen_dec']}" for r in rows)),
    "rej_dec_A": dict(C(r["rej_dec"] for r in rows if r["twin"] == "A").most_common()),
    "rej_dec_B": dict(C(r["rej_dec"] for r in rows if r["twin"] == "B").most_common()),
    "chosen_eval_strict": sum(r["chosen_eval_strict"] for r in rows),
    "rej_eval_strict": sum(r["rej_eval_strict"] for r in rows),
    "rej_eval_lenient": sum(r["rej_eval_lenient"] for r in rows),
    "rej_eval_lenient_rows": [[r["idx"], r["scenario_id"], r["level"], r["prompt_setting"], r["rej_dec"]] for r in rows if r["rej_eval_lenient"]],
    "observed_models": dict(C(r["prov"]["model"] for r in rows if r["prov"])),
    "observed_grid_prompt": dict(C(r["prov"]["grid_prompt"] for r in rows if r["prov"])),
    "observed_tool_results": dict(C(r["prov"]["tool_results"] for r in rows if r["prov"])),
    "observed_n_events": dict(C(r["prov"]["n_events"] for r in rows if r["prov"])),
    "observed_first_event": sum(1 for r in rows if r["prov"] and r["prov"]["n_events"] == 1),
    "observed_prev_only_invalid": sum(1 for r in rows if r["prov"] and r["prov"]["n_events"] > 1
                                      and all(p == "invalid" for p in r["prov"]["prev_kinds"])),
    "observed_prev_equals_canonical_nonempty": sum(1 for r in rows if r["prov"] and r["prev_equals_canonical"]
                                                   and r["prov"]["prev_calls"]),
    "observed_prev_has_error_result": sum(1 for r in rows if r["prov"] and any(r["prov"]["prev_results_err"])),
    "observed_files": dict(C(r["prov"]["file"] for r in rows if r["prov"])),
}
json.dump(summ, open(os.path.join(OUT, "v1_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
