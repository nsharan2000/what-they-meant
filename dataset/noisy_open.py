"""L3 noisy requests with an OPEN model (no Claude): casual / typo'd / locale-code-mixed rewrites of each member request, with every
entity, amount, handle, number and date preserved (validated). Twins of a pair get rewrites of their own request text.

python3 dataset/noisy_open.py [--model Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.convert_seed import idents  # noqa: E402
from wtm.drivers.openai_compat import OpenAICompat  # noqa: E402
from wtm.spec import ROOT, load_member, norm  # noqa: E402

SYSTEM = """You rewrite a short message a customer typed into a payments app so it reads like a real hurried message: lowercase or
inconsistent case, casual punctuation, a filler like "pls", "asap" or "thx" where natural, at most one typo and ONLY in a filler or
small word (never in a content word). Keep EVERY content word of the original (every word of 4+ letters) and EVERY name, amount,
currency, handle, email, phone number, account number, date and product exactly as written (copy them character for character).
Do not add new requests or new facts. If told the customer is in India, mix in a little natural Hinglish (e.g. "bhai", "jaldi",
"karo") around the unchanged details; if in Germany, at most one German filler word. Output only the rewritten message."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra")
    ns = ap.parse_args()
    drv = OpenAICompat(model=ns.model, temperature=0.7, max_tokens=200, cache_dir=os.path.join(ROOT, "experiments", "smoke", "cache_noisy2"))
    stats = {"ok": 0, "fallback": 0}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios", "*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        for r in rows:
            msg = r["user_message"]
            country = str(load_member(r["env_id"], r["member_id"]).get("profile", {}).get("country", ""))
            stop = {"this", "that", "with", "from", "have", "they", "them", "their", "your", "please", "just", "into", "will"}
            content = [w for w in re.findall(r"[A-Za-z][A-Za-z'&-]{3,}", msg) if w.lower() not in stop]
            need = [t for t in idents(msg)] + re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", msg) + content
            best = None
            for attempt in range(4):
                drv.temperature = 0.7 + 0.1 * attempt
                out = drv.chat(SYSTEM, [{"role": "user", "content": f"Customer location: {country or 'unknown'}\nMessage: {msg}"}]).text.strip()
                out = out.strip('"').strip()
                if out and out != msg and all(norm(t) in norm(out) for t in need):
                    best = out
                    break
            if best:
                r["user_message_noisy"] = best
                stats["ok"] += 1
            else:
                r["user_message_noisy"] = msg.lower()
                stats["fallback"] += 1
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats | {"tokens": [drv.total_prompt_tokens, drv.total_completion_tokens]}))


if __name__ == "__main__":
    main()
