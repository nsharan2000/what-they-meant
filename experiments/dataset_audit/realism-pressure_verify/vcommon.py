"""Independent loaders + stats for the VERIFY pass of the realism-pressure lens (does not import the analyst's code).

Reads (read-only): experiments/replication/results/grid_*.jsonl, experiments/training_eval/results/eval_base_v12.jsonl,
the hard-set panel logs in the session scratchpad (not in the repo), dataset/curated_v121/*, dataset/splits.json.
Run every script with `python3 -B` from anywhere.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
GRID = os.path.join(ROOT, "experiments", "replication", "results")
EVALB = os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl")
PANEL = ("scratch/"
         "c2ef7a67-d083-4a26-86a4-38d6c96fa533/scratchpad/hard")
CUR = os.path.join(ROOT, "dataset", "curated_v121")

# spec family = rows rendered from the same scenario-spec version (contrasts only within one family)
FAM = {"grid_open_P0": "seed_open", "grid_open_P1": "seed_open", "grid_open_controls": "seed_open",
       "grid_local_tinyaya": "seed_local", "grid_spark": "seed_spark", "grid_claude": "seed_claude",
       "grid_open_variants": "variants", "eval_base_v12": "eval_v12",
       "panel_hard_claude": "panel_hard", "panel_hard_open": "panel_hard",
       "panel_mined_claude": "panel_mined", "panel_mined_open": "panel_mined", "panel_mined_fable": "panel_mined"}


def files():
    for f in sorted(glob.glob(os.path.join(GRID, "grid_*.jsonl"))):
        yield os.path.basename(f)[:-6], f
    yield "eval_base_v12", EVALB
    for f in sorted(glob.glob(os.path.join(PANEL, "panel_*.jsonl"))):
        n = os.path.basename(f)[:-6]
        if n != "panel_hard_5models":          # a re-packaged subset of panel_hard_claude + panel_hard_open
            yield n, f


def mname(m: str) -> str:
    if m.startswith("claude:"):
        return "claude-" + m.split(":")[1]
    m = m.split("@")[0]
    m = re.sub(r"^(hf|vllm):", "", m)
    m = m.split(":")[0]
    return m.split("/")[-1]


def ref(path: str, i: int) -> str:
    return (os.path.relpath(path, ROOT) if path.startswith(ROOT) else path) + f":{i}"


def episodes(keep_events: bool = False):
    """Yield one dict per A/B decision-mode episode (rep 0, pushback rows and driver errors excluded)."""
    for src, path in files():
        fam = FAM.get(src)
        with open(path) as fh:
            for i, l in enumerate(fh):
                r = json.loads(l)
                tw = {"A_should_ask": "A", "B_should_act": "B"}.get(r["twin"])
                if tw is None or r.get("mode", "decision") != "decision" or r.get("rep", 0) != 0:
                    continue
                dec = r["decision_grade"]["decision"]
                e = {"src": src, "fam": fam, "path": path, "line": i, "model": mname(r["model"]), "id": r["id"],
                     "pair": r["pair_id"], "twin": tw, "level": r["level"], "prompt": r["prompt"], "dec": dec,
                     "succ": bool(r["decision_grade"]["success"]), "env": r.get("env_id")}
                if keep_events:
                    e["events"] = r.get("events", [])
                yield e


# ---- decision taxonomy (wtm/episode.py classify_decision / decision_grade)
def ind(dec: str, tw: str) -> dict:
    ask = dec in ("ask_informed", "ask_generic")
    appr = dec.startswith("approve")
    com = dec.startswith("commit")
    d = {"ask": int(ask), "approve": int(appr), "commit": int(com), "act": int(appr or com),
         "informed": int(dec in ("ask_informed", "tell_informed"))}
    if tw == "A":
        d["literal"] = int(dec in ("approve_generic", "commit_other", "commit_harmful"))
    else:
        d["needless"] = int(dec in ("ask_informed", "ask_generic", "tell_informed"))
    return d


# ---- clusters (source posts) for bootstrap
def cluster_map():
    sp = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    m = {}
    for c, v in sp["clusters"].items():
        for p in v["pairs"]:
            m[p] = c
    return m


def cluster(pair: str, cm: dict) -> str:
    if pair in cm:
        return cm[pair]
    return cm.get("S" + pair[1:3], pair)      # V<ss>k / H<ss>k are variants of seed S<ss>


def boot(vals, clusters=None, n=10000, seed=7):
    """mean and 95% percentile CI; cluster bootstrap when clusters is given."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    else:
        u, inv = np.unique(np.asarray(clusters), return_inverse=True)
        s = np.bincount(inv, weights=v, minlength=len(u))
        c = np.bincount(inv, minlength=len(u)).astype(float)
        w = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=n).astype(float)
        means = (w @ s) / np.maximum(w @ c, 1e-12)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def dump(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=float)
    return p


def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]
