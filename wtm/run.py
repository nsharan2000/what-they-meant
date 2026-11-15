"""Run a replication grid: models × scenarios × realism levels × prompt settings → results JSONL (resumable).

Model spec strings:
  claude:<alias|id>                   pure Claude Code agent (haiku, sonnet, opus, fable or a full id)
  hf:<repo_id>:<provider>             HF Inference Providers router, e.g. hf:CohereLabs/tiny-aya-global:cohere
  vllm:<served_name>@<base_url>       local vLLM (OpenAI-compatible), e.g. vllm:tiny-aya-global@http://localhost:8001/v1

Always ends with a RUN_SUMMARY line (written / errors / skipped / already_done / jobs), even when a job crashes, so a wrapper can
tell a complete run from one that silently dropped episodes. written = rows appended to --out (with --keep-errors that includes
driver-error rows); errors = episodes NOT written (retried on resume). --strict makes any unwritten episode exit 2 (run_eval.sh
uses it); without it the exit status is unchanged (0 unless a job crashes).
served_adapters() reads the label -> served-name map that run_eval.sh appends to <out>.adapters.txt (shared by the analysis
scripts, so they query exactly the adapter versions the last eval served).

Usage:
  python3 -m wtm.run --scenarios dataset/scenarios.jsonl --models claude:haiku hf:CohereLabs/tiny-aya-global:cohere \
      --levels L0 L1 L2 --prompts P0 P1 --mode decision --out experiments/replication/results/grid.jsonl --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time

from .render import render
from .spec import ROOT, load_env, load_jsonl, load_member
from .episode import run_episode

_LOCK = threading.Lock()


def make_driver(spec: str, cache_dir: str, max_tokens: int = 1024, temperature: float = 0.0):
    kind, rest = spec.split(":", 1)
    if kind == "claude":
        from .drivers.claude_pure import ClaudePure
        return ClaudePure(model=rest)
    from .drivers.openai_compat import OpenAICompat, HF_ROUTER
    if kind == "hf":
        return OpenAICompat(model=rest, base_url=HF_ROUTER, cache_dir=cache_dir, max_tokens=max_tokens,
                            temperature=temperature)
    if kind == "vllm":
        name, url = rest.split("@", 1)
        return OpenAICompat(model=name, base_url=url, cache_dir=cache_dir, max_tokens=max_tokens,
                            temperature=temperature)
    raise ValueError(spec)


def job_key(model: str, sid: str, level: str, prompt: str, mode: str, rep: int) -> str:
    return f"{model}|{sid}|{level}|{prompt}|{mode}|r{rep}"


def served_adapters(path: str) -> dict[str, str]:
    """{label: served name} from the LAST '=== ... run_eval READY' block of run_eval.sh's adapters.txt (older blocks name older
    adapter versions). Raises ValueError when the file has no such block."""
    block = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("=== ") and " run_eval READY" in line:
                block = {}
            elif block is not None:
                parts = line.split()   # '<date> <time> <label> -> <served> = <path>'
                if len(parts) >= 5 and parts[3] == "->":
                    block[parts[2]] = parts[4]
    if block is None:
        raise ValueError(f"{path} has no '=== ... run_eval READY' block (written by run_eval.sh once its server is up)")
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=[os.path.join(ROOT, "dataset", "scenarios", "*.jsonl")])
    ap.add_argument("--ids", nargs="*", default=None, help="restrict to these scenario ids")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", default=["L0", "L1"])
    ap.add_argument("--prompts", nargs="+", default=["P0"])
    ap.add_argument("--mode", default="decision", choices=["decision", "full"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--keep-errors", action="store_true", help="write episodes that ended in a driver error (default: retry them)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true", help="exit 2 if any episode errored (it is then missing from --out)")
    ns = ap.parse_args()

    import glob as _glob
    scns = [r for p in ns.scenarios for f in sorted(_glob.glob(p)) for r in load_jsonl(f)]
    if ns.ids:
        scns = [s for s in scns if s["id"] in set(ns.ids)]
    os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
    done = set()
    if os.path.exists(ns.out):
        for r in load_jsonl(ns.out):
            done.add(r["key"])
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(ns.out)), "..", "cache")
    drivers = {m: make_driver(m, cache_dir, ns.max_tokens, ns.temperature) for m in ns.models}
    envs, members = {}, {}

    jobs = []
    for m in ns.models:
        for s in scns:
            for lv in ns.levels:
                for p in ns.prompts:
                    for rep in range(ns.reps):
                        k = job_key(m, s["id"], lv, p, ns.mode, rep)
                        if k not in done:
                            jobs.append((m, s, lv, p, rep, k))
    print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)
    # preload every environment and member up front: long local runs may lose read/write access to ~/Desktop later (log 21:00)
    for s in scns:
        envs.setdefault(s["env_id"], load_env(s["env_id"]))
        mk = (s["env_id"], s["member_id"])
        if mk not in members:
            members[mk] = load_member(*mk)
    from .render import golden_text
    golden_text()   # warm the golden prompt read

    abort = {"reason": None}

    def run(job):
        m, s, lv, p, rep, k = job
        if abort["reason"]:
            return "skipped"
        env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
        mk = (s["env_id"], s["member_id"])
        if mk not in members:
            members[mk] = load_member(*mk)
        system, first_user, tools = render(env, members[mk], s, lv, p)
        try:
            res = run_episode(env, members[mk], s, drivers[m], system, first_user, tools, mode=ns.mode, max_steps=ns.max_steps,
                              level=lv)
        except Exception as e:  # noqa: BLE001 — usage-limit aborts must not be recorded as finished episodes
            if type(e).__name__ == "UsageLimitError":
                abort["reason"] = str(e)
                return "aborted"
            import traceback
            print(f"EPISODE_ERROR {k}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", flush=True)
            return "error"   # not written → retried on the next (resumed) run
        if res.get("decision_grade", {}).get("decision") == "error" and not ns.keep_errors:
            errs = [e.get("error", "") for e in res.get("events", []) if e.get("kind") == "error"]
            print(f"EPISODE_INFRA_ERROR {k}: {(errs[-1] if errs else '')[:200]}", flush=True)
            return "error"   # driver/infrastructure failure (not model behaviour) → not written, retried on resume
        has_raw = any("result_raw" in r for r in s.get("tool_rules", []))
        row = {"key": k, "tool_results": "raw" if (has_raw and lv != "L0") else "seed",
               "conversion": s.get("conversion", ""), "model": m, "id": s["id"], "pair_id": s["pair_id"], "twin": s["twin"], "env_id": s["env_id"],
               "level": lv, "prompt": p, "mode": ns.mode, "rep": rep, "tags": s.get("tags", []),
               "system_chars": len(system), "n_tools": len(tools), **res, "t": time.time()}
        try:
            line = json.dumps(row, ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            print(f"EPISODE_ERROR {k}: unserialisable row: {e}", flush=True)
            return "error"
        with _LOCK, open(ns.out, "a") as fh:
            fh.write(line + "\n")
        return "written"

    counts = {"written": 0, "errors": 0, "skipped": 0}

    def run_counted(job):
        try:
            d = run(job)
        except BaseException:   # still crashes the run as before; counted so RUN_SUMMARY stays truthful
            with _LOCK:
                counts["errors"] += 1
            raise
        with _LOCK:
            counts[{"written": "written", "error": "errors"}.get(d, "skipped")] += 1   # "skipped" / "aborted"
        return d

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(max_workers=ns.workers) as ex:
            for i, d in enumerate(ex.map(run_counted, jobs), 1):
                if i % 20 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
        if abort["reason"]:
            print(f"ABORTED_USAGE_LIMIT: {abort['reason']}", flush=True)
        spent = {m: round(getattr(d, "total_cost_usd", 0.0), 4) for m, d in drivers.items()}
        toks = {m: (getattr(d, "total_prompt_tokens", 0), getattr(d, "total_completion_tokens", 0)) for m, d in drivers.items()}
        print(json.dumps({"claude_cost_usd": spent, "open_tokens(prompt,completion)": toks}), flush=True)
    finally:
        print(f"RUN_SUMMARY written={counts['written']} errors={counts['errors']} skipped={counts['skipped']} "
              f"already_done={len(done)} jobs={len(jobs)}", flush=True)
    if ns.strict and counts["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
