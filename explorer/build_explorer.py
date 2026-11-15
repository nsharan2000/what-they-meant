"""Build the dataset-explorer data files from the repo (never retypes numbers: everything is read from results files).

python3 explorer/build_explorer.py [--scenarios dataset/scenarios/*.jsonl] [--results experiments/replication/results/grid_*.jsonl]
        [--judge experiments/replication/results/judge_*.jsonl] [--out explorer/site]
Writes explorer/site/index.html (from explorer/template.html) and explorer/site/data/{manifest.json, env_<id>.json, roll_<id>.json}.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.render import render  # noqa: E402
from wtm.spec import load_env, load_jsonl, load_member  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL_ORDER = ["L0", "L0r", "L0p", "L1", "L2", "L2s", "L3"]


def model_label(m: str) -> str:
    kind, rest = m.split(":", 1)
    if kind == "claude":
        return {"haiku": "Claude Haiku 5.5", "sonnet": "Claude Sonnet 5.5", "opus": "Claude Opus 5.5",
                "fable": "Claude Fable 5.2"}.get(rest, rest)
    name = rest.split("@")[0].split(":")[0].split("/")[-1]
    return name.replace("-", " ").replace("tiny aya", "Tiny Aya").replace("aya expanse", "Aya Expanse")


def _dump(obj, path: str) -> None:
    """JSON without U+FFFD (a few model outputs carry broken bytes; the artifact host refuses replacement characters)."""
    with open(path, "w") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False).replace("\ufffd", "?"))


def compact_events(evts: list[dict]) -> list[dict]:
    out = []
    for e in evts:
        k = e.get("kind")
        if k == "tool_call":
            res = e.get("result")
            out.append({"t": "call", "name": e.get("tool"), "kind": e.get("tool_kind"), "args": e.get("args", {}),
                        "result": res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)})
        elif k == "message":
            out.append({"t": "msg", "text": e.get("text", ""), "informed": e.get("informed"), "reply": e.get("user_reply")})
        elif k == "invalid":
            out.append({"t": "invalid", "raw": str(e.get("raw", ""))[:600]})
        elif k == "error":
            out.append({"t": "error", "text": e.get("error", "")[:300]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl",
                                                       "dataset/scenarios_hard/*.jsonl"])
    ap.add_argument("--hardsets", nargs="*", default=["experiments/hardset/results/hardset_mined.json",
                                                      "experiments/hardset/results/hardset_generated.json"])
    ap.add_argument("--splits", default="dataset/splits.json")
    ap.add_argument("--results", nargs="*", default=["experiments/replication/results/grid_*.jsonl"])
    ap.add_argument("--judge", nargs="*", default=["experiments/replication/results/judge_*.jsonl"])
    ap.add_argument("--out", default=os.path.join(HERE, "site"))
    ap.add_argument("--models", nargs="*", default=None, help="substrings of model specs to include (default: all)")
    ns = ap.parse_args()
    scns = [s for p in ns.scenarios for f in sorted(glob.glob(p)) for s in load_jsonl(f)]
    splits = json.load(open(ns.splits)) if os.path.exists(ns.splits) else {}
    pair_split = {pid: k for k in ("train", "val", "test") for pid in splits.get(k, [])}
    hard_info = {}
    for hp in ns.hardsets:
        if os.path.exists(hp):
            h = json.load(open(hp))
            for p in h.get("pairs", []):
                if p["pair"] in set(h.get("hard_pairs", [])):
                    hard_info[p["pair"]] = {"n_fail": p["n_fail"], "of": len(h.get("models", [])),
                                            "failed_by": [m.split(":")[-1] if m.startswith("claude") else m.split(":")[1].split("/")[-1]
                                                          for m in p["failed_by"]]}
    judged = {}
    for p in ns.judge:
        for f in glob.glob(p):
            for j in load_jsonl(f):
                judged[j["key"]] = j.get("judge")
    rolls = defaultdict(list)
    for p in ns.results:
        for f in glob.glob(p):
            for r in load_jsonl(f):
                if r.get("rep", 0) == 0 and (not ns.models or any(m in r["model"] for m in ns.models)):
                    rolls[r["id"]].append(r)
    os.makedirs(os.path.join(ns.out, "data"), exist_ok=True)
    by_env = defaultdict(list)
    for s in scns:
        by_env[s["env_id"]].append(s)
    manifest = {"envs": [], "models": sorted({model_label(r["model"]) for rs in rolls.values() for r in rs}),
                "levels": LEVEL_ORDER}
    for env_id, rows in by_env.items():
        env = load_env(env_id)
        members, prompts = {}, {}
        for s in rows:
            mk = s["member_id"]
            if mk not in members:
                members[mk] = load_member(env_id, mk)
            for lv in ("L0", "L1", "L2"):
                key = f"{s['id']}|{lv}" if lv == "L0" else f"{mk}|{lv}"
                if key not in prompts:
                    prompts[key] = render(env, members[mk], s, lv, "P0")[0]
        env_out = {"env_id": env_id, "app_name": env.get("app_name", env_id), "summary": env.get("summary", ""),
                   "audience": env.get("audience", ""), "tools": [{k: t.get(k) for k in ("name", "kind", "description", "parameters")}
                                                                  for t in env["tools"]],
                   "members": members, "prompts": prompts,
                   "scenarios": [{k: s.get(k) for k in ("id", "pair_id", "twin", "member_id", "sub_domain", "tags", "user_message",
                                                        "user_message_noisy", "source", "decisive_facts", "expected", "user_script",
                                                        "l0_tools", "synthetic", "variant_of", "clash_type", "label_audit",
                                                        "scenario_records")}
                                 | {"seed_conversation": s.get("seed", {}).get("conversation", []), "split": pair_split.get(s["pair_id"]),
                                    "hard": hard_info.get(s["pair_id"]), "eval_only": bool(s.get("eval_only")),
                                    "hard_pattern": s.get("hard_pattern")}
                                 for s in rows]}
        roll_out = {}
        for s in rows:
            for r in rolls.get(s["id"], []):
                roll_out.setdefault(s["id"], []).append({
                    "model": model_label(r["model"]), "level": r["level"], "prompt": r["prompt"], "mode": r.get("mode"),
                    "grade": r["decision_grade"], "judge": judged.get(r["key"]), "events": compact_events(r.get("events", [])),
                    "system_chars": r.get("system_chars"), "n_tools": r.get("n_tools"), "cost_usd": r.get("cost_usd")})
        _dump(env_out, os.path.join(ns.out, "data", f"env_{env_id}.json"))
        _dump(roll_out, os.path.join(ns.out, "data", f"roll_{env_id}.json"))
        stats = defaultdict(lambda: [0, 0])
        for sid, rs in roll_out.items():
            for r in rs:
                stats[sid][0] += 1
                stats[sid][1] += int(bool(r["grade"]["success"]))
        manifest["envs"].append({"env_id": env_id, "app_name": env_out["app_name"], "summary": env_out["summary"],
                                 "scenarios": [{"id": s["id"], "twin": s["twin"], "user_message": s["user_message"],
                                                "tags": s.get("tags", []), "n_roll": stats[s["id"]][0], "n_pass": stats[s["id"]][1],
                                                "syn": bool(s.get("synthetic")), "variant_of": s.get("variant_of"),
                                                "split": pair_split.get(s["pair_id"]) or ("hard" if s.get("eval_only") else None),
                                                "hard": bool(hard_info.get(s["pair_id"])), "gen_hard": bool(s.get("eval_only"))}
                                               for s in rows]})
    _dump(manifest, os.path.join(ns.out, "data", "manifest.json"))
    shutil.copy(os.path.join(HERE, "template.html"), os.path.join(ns.out, "index.html"))
    sizes = {f: os.path.getsize(os.path.join(ns.out, "data", f)) for f in os.listdir(os.path.join(ns.out, "data"))}
    print(json.dumps({"envs": list(by_env), "n_scenarios": len(scns), "n_rollouts": sum(len(v) for v in rolls.values()),
                      "bytes": sizes}, indent=1))


if __name__ == "__main__":
    main()
