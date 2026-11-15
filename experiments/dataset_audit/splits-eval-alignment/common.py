"""Shared loaders for the splits-eval-alignment audit (read-only on every repo file; run every script with python3 -B)."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CUR = os.path.join(ROOT, "dataset", "curated_v121")
VIEWS = ("sft_train", "sft_val", "dpo_train", "grpo_train")


def jl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def views():
    return {v: jl(os.path.join(CUR, f"{v}.jsonl")) for v in VIEWS}


def scen_files(kind):
    pat = {"seed": "dataset/scenarios/*.jsonl", "synth": "dataset/scenarios_synth/*.jsonl",
           "hard": "dataset/scenarios_hard/*.jsonl"}[kind]
    return [r for f in sorted(glob.glob(os.path.join(ROOT, pat))) for r in jl(f)]


def eval_files():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        out[os.path.basename(f)[:-6]] = jl(f)
    return out


def splits():
    return json.load(open(os.path.join(ROOT, "dataset", "splits.json")))


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


TOK_RE = re.compile(r"[a-z0-9]+(?:[.@'_-][a-z0-9]+)*")


def toks(s: str) -> set:
    """normalised-token set for Jaccard: NFKC, lower case, currency symbols and thousands commas dropped."""
    t = unicodedata.normalize("NFKC", s or "").lower().replace("’", "'")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    return set(TOK_RE.findall(t))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def target_text(r: dict) -> str:
    c = r.get("completion") or r.get("chosen")
    return c[0]["content"]


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    return p
