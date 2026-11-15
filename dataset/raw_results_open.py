"""Generate `result_raw` (raw API fields, no verdict words) for every tool rule with an OPEN model (no Claude), with validation.

Why: the realism ladder's L0r anchor and the realistic L1/L2 levels need tool outputs that state facts as fields instead of
benchmark-style verdicts ("matches last payment", "no photo"). Validation per rule (retry ×3 with feedback):
  * valid JSON object
  * no verdict words (wtm.invariants.VERDICT_RE)
  * every identifier / number in the seed result (dataset/convert_seed.idents) appears in the raw JSON (normalised)
Writes result_raw into dataset/scenarios/*.jsonl in place (rules shared by twins get the same output; cache in dataset/_raw_cache.json).

python3 dataset/raw_results_open.py [--model Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra] [--workers 8]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.convert_seed import idents  # noqa: E402
from wtm.drivers.openai_compat import OpenAICompat  # noqa: E402
from wtm.invariants import VERDICT_RE  # noqa: E402
from wtm.spec import ROOT, load_env, norm  # noqa: E402

SYSTEM = """You convert a short human-written summary of an API response into the raw JSON the API itself would return.
Rules: output ONE JSON object only, no prose, no code fences. Keep every fact, name, identifier, number, amount, currency, date and
time from the summary exactly as written. Express judgements as plain fields instead of words: e.g. "no photo" becomes
"photo_url": null; "never paid" becomes "payments_from_member": 0; "matches last payment" becomes the record fields that would let a
reader compare (do NOT say it matches). Never use these words anywhere: match, matches, mismatch, suspicious, scam, fraud, warning,
risk, never paid, no photo, already paid, duplicate, verified, unverified, legit, safe, caution, wrong, correct, fake.
If the summary itself uses such a word for a fact (e.g. "duplicate event ignored", "callback verified"), express the fact as fields
(e.g. "events_received": 2, "events_processed": 1; "callback_status": "completed") instead of the word.
Use realistic snake_case field names for the named tool."""


def prompt_for(tool: dict, args_hint: list, result: str, feedback: str = "") -> str:
    return (f"Tool: {tool['name']} — {tool['description']}\nCalled with arguments matching: {args_hint}\n"
            f"Human-written summary of the response: {result}\n" + (f"\nYour previous attempt was rejected: {feedback}\n" if feedback else "") +
            "\nReturn the raw JSON response object.")


def validate(raw_text: str, seed_result: str):
    m = re.search(r"\{.*\}", raw_text or "", re.S)
    if not m:
        return None, "no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e})"
    blob = json.dumps(obj, ensure_ascii=False)
    hit = VERDICT_RE.search(blob)
    if hit:
        return None, f"contains the verdict word {hit.group(0)!r}"
    nb = norm(blob)
    nums = {round(float(x), 2) for x in re.findall(r"-?\d+(?:\.\d+)?", blob.replace(",", ""))}

    def present(t: str) -> bool:
        cands = {t, t.replace(",", ""), t.lstrip("*"), re.sub(r"(?i)^ending\s+", "", t)}
        if any(norm(c) in nb for c in cands if c):
            return True
        num = t.replace(",", "").lstrip("$€£₹")
        if re.fullmatch(r"\d+(\.\d+)?", num):
            return round(float(num), 2) in nums
        return False
    missing = [t for t in idents(seed_result) if not present(t)]
    if missing:
        return None, f"lost these facts: {missing[:5]}"
    return obj, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra")
    ap.add_argument("--workers", type=int, default=8)
    ns = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios", "*.jsonl")))
    rows = {f: [json.loads(l) for l in open(f) if l.strip()] for f in files}
    cache_p = os.path.join(ROOT, "dataset", "_raw_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    envs = {}
    jobs = {}
    for f, rs in rows.items():
        for r in rs:
            env = envs.setdefault(r["env_id"], load_env(r["env_id"]))
            tools = {t["name"]: t for t in env["tools"]}
            for rule in r["tool_rules"]:
                if isinstance(rule["result"], str):
                    k = hashlib.sha256(json.dumps([r["env_id"], rule["tool"], rule["result"]]).encode()).hexdigest()[:16]
                    jobs[k] = (tools.get(rule["tool"], {"name": rule["tool"], "description": ""}), rule["match"], rule["result"])
    todo = [k for k in jobs if k not in cache]
    print(f"{len(jobs)} distinct rule results, {len(todo)} to generate", flush=True)
    drv = OpenAICompat(model=ns.model, temperature=0.0, max_tokens=700,
                       cache_dir=os.path.join(ROOT, "experiments", "smoke", "cache_raw"))
    lock = threading.Lock()
    failures = {}

    def work(k):
        tool, hint, result = jobs[k]
        fb = ""
        for attempt in range(3):
            res = drv.chat(SYSTEM, [{"role": "user", "content": prompt_for(tool, hint, result, fb)}])
            obj, fb = validate(res.text, result)
            if obj is not None:
                with lock:
                    cache[k] = obj
                return True
            drv.temperature = 0.3   # vary on retry (cache key includes temperature)
        failures[k] = {"tool": tool["name"], "result": result, "last_error": fb}
        return False

    with cf.ThreadPoolExecutor(max_workers=ns.workers) as ex:
        ok = sum(ex.map(work, todo))
    json.dump(cache, open(cache_p, "w"), indent=1, ensure_ascii=False)
    n_set = 0
    for f, rs in rows.items():
        for r in rs:
            for rule in r["tool_rules"]:
                if isinstance(rule["result"], str):
                    k = hashlib.sha256(json.dumps([r["env_id"], rule["tool"], rule["result"]]).encode()).hexdigest()[:16]
                    if k in cache:
                        rule["result_raw"] = cache[k]
                        n_set += 1
        with open(f, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(failures, open(os.path.join(ROOT, "dataset", "_raw_failures.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"generated_ok": ok, "failed": len(failures), "rules_with_raw": n_set,
                      "tokens": [drv.total_prompt_tokens, drv.total_completion_tokens]}), flush=True)


if __name__ == "__main__":
    main()
