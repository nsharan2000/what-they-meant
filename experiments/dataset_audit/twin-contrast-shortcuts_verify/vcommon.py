"""Independent helpers for the VERIFIER of the 'twin-contrast-shortcuts' lens (written from scratch; does not import the
analyst's common.py). Read-only over the repo; bytecode writing disabled so importing wtm/* creates no __pycache__ files.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(HERE, "results")
CUR = os.path.join(ROOT, "dataset", "curated_v121")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def jl(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def save(obj, name):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)


def curated(name):
    rows = jl(os.path.join(CUR, f"{name}.jsonl"))
    for i, r in enumerate(rows):
        r["_i"] = i
        r.setdefault("turn", "decision")
    return rows


def T(r):
    """twin letter: A, B or P (pushback)"""
    t = r["twin"]
    return "P" if t.startswith("A_pushback") else t[0]


HINT = open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()


def nohint(system):
    s = system.rstrip()
    return s[: -len(HINT)].rstrip() if s.endswith(HINT) else s


def parts(prompt):
    sysm = prompt[0]["content"]
    req = prompt[1]["content"] if len(prompt) > 1 else ""
    calls = [m["content"] for m in prompt[2:] if m["role"] == "assistant"]
    results = [m["content"] for m in prompt[2:] if m["role"] == "user"]
    return sysm, req, calls, results


RX_RESULT = re.compile(r'^<tool_result name="([^"]*)">\n(.*)\n</tool_result>$', re.S)


def result_obj(content):
    m = RX_RESULT.match(content.strip())
    if not m:
        return None, content
    try:
        return json.loads(m.group(2)), m.group(2)
    except Exception:  # noqa: BLE001
        return None, m.group(2)


def leaves(obj):
    out = []
    if isinstance(obj, dict):
        if not obj:
            out.append(("", {}))
        for k, v in obj.items():
            out += [((k + ("." + p if p else "")), x) for p, x in leaves(v)]
    elif isinstance(obj, list):
        if not obj:
            out.append(("", []))
        for v in obj:
            out += [("[]" + ("." + p if p else ""), x) for p, x in leaves(v)]
    else:
        out.append(("", obj))
    return out


def struct_counts(obj):
    vals = [v for _, v in leaves(obj)] if isinstance(obj, (dict, list)) else []
    return {"leaves": len(vals),
            "null": sum(v is None for v in vals),
            "false": sum(v is False for v in vals),
            "true": sum(v is True for v in vals),
            "zero": sum((not isinstance(v, bool)) and isinstance(v, (int, float)) and v == 0 for v in vals),
            "empty": sum(v in ([], {}) for v in vals)}


def all_specs():
    """id -> spec, over seed + synthetic training/val specs and the eval files; with the file group."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios", "*.jsonl"))):
        for s in jl(f):
            out[s["id"]] = dict(s, _group="seed")
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "scenarios_synth", "*.jsonl"))):
        for s in jl(f):
            out.setdefault(s["id"], dict(s, _group="synth"))
    return out


def eval_specs(name):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", f"{name}_*.jsonl"))):
        rows += jl(f)
    return rows


def splits():
    sp = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    pair_split = {p: k for k in ("train", "val", "test") for p in sp[k]}
    cluster = {}
    for cid, c in sp["clusters"].items():
        for p in c["pairs"]:
            cluster[p] = cid
    return pair_split, cluster


def token_stats():
    rows = jl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))
    return {(r["file"].replace(".jsonl", ""), r["i"]): r for r in rows}


def match_twins(rows, key=("pair_id", "level", "turn")):
    """{(pair, level, turn): {'A': row, 'B': row}} for decision/lookup rows; pushback rows ignored."""
    d = defaultdict(dict)
    for r in rows:
        t = T(r)
        if t == "P":
            continue
        k = tuple(r[x] for x in key)
        d[k].setdefault(t, []).append(r)
    return d


def sign_test_p(k, n):
    """two-sided exact binomial sign test, p=0.5"""
    if n == 0:
        return None
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a,b],[c,d]] two-sided Fisher exact p (sum of probabilities <= observed)."""
    from math import comb
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def words(t):
    return WORD.findall(t.lower())
