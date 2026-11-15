"""Pick which episodes the Opus judge reviews (cost control, documented): writes a results-format JSONL that wtm.judge consumes.

  * every Claude-arm episode whose decision is ambiguous for keyword grading (all A-twin non-commit decisions; all B-twin questions/
    statements; any approve_gold/approve_informed/tell_informed);
  * key open models (Tiny Aya Global, Aya Expanse 32B, Qwen3-235B, DeepSeek-V3.2) at L0r/L2: the same ambiguous categories;
  * a deterministic 20 % calibration sample of the clear-cut rest (to measure grader–judge agreement).
  * --full: every row of the selected models/levels/prompts (label-validity pass: one transcript per scenario row).
python3 experiments/replication/select_for_judge.py --results <grid files…> --out <dir>/judge_queue.jsonl
    [--models <substr> …] [--levels L0r L2] [--prompts P0 P1] [--full] [--no-claude]
"""
import argparse
import glob
import hashlib
import json

AMBIG_A = {"ask_informed", "tell_informed", "ask_generic", "statement", "approve_gold", "approve_informed", "approve_generic",
           "commit_gold", "commit_other", "invalid", "no_decision"}
AMBIG_B = {"ask_informed", "ask_generic", "tell_informed", "statement", "commit_gold", "commit_other"}
# open models: only categories where keyword grading can plausibly be wrong (direct payments and plain cards are clear-cut)
AMBIG_A_OPEN = {"ask_informed", "tell_informed", "ask_generic", "statement", "approve_gold", "approve_informed"}
AMBIG_B_OPEN = {"ask_informed", "ask_generic", "tell_informed", "statement"}
KEY = ("tiny-aya-global", "aya-expanse-32b", "Qwen3-235B", "DeepSeek-V3.2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=float, default=0.20)
    ap.add_argument("--models", nargs="*", default=list(KEY), help="substrings of open models to include")
    ap.add_argument("--levels", nargs="*", default=["L0r", "L2"], help="levels for open models")
    ap.add_argument("--prompts", nargs="*", default=["P0", "P1"])
    ap.add_argument("--full", action="store_true", help="include every selected row (label-validity pass)")
    ap.add_argument("--no-claude", action="store_true")
    ns = ap.parse_args()
    picked, seen = [], set()
    for p in ns.results:
        for f in glob.glob(p):
            for l in open(f):
                r = json.loads(l)
                if r["key"] in seen:
                    continue
                seen.add(r["key"])
                d = r["decision_grade"]["decision"]
                claude = r["model"].startswith("claude:") and not ns.no_claude
                key_open = (not r["model"].startswith("claude:") and any(k in r["model"] for k in ns.models)
                            and r["level"] in ns.levels)
                if not (claude or key_open) or r["prompt"] not in ns.prompts:
                    continue
                if claude:
                    ambiguous = d in (AMBIG_A if r["twin"].startswith("A") else AMBIG_B)
                else:
                    ambiguous = d in (AMBIG_A_OPEN if r["twin"].startswith("A") else AMBIG_B_OPEN)
                h = int(hashlib.sha256(r["key"].encode()).hexdigest(), 16) % 1000 / 1000
                if ns.full or ambiguous or h < ns.calib:
                    r["judge_reason"] = "ambiguous" if ambiguous else ("calibration" if h < ns.calib else "full")
                    picked.append(r)
    with open(ns.out, "w") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in picked:
        k = ("claude" if r["model"].startswith("claude") else "open", r["judge_reason"])
        by[k] = by.get(k, 0) + 1
    print(json.dumps({"picked": len(picked), "by": {f"{a}/{b}": v for (a, b), v in sorted(by.items())}}))


if __name__ == "__main__":
    main()
