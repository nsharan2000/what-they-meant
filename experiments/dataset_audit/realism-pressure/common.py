"""Shared loaders, decision taxonomy, metrics and bootstrap for the realism-pressure audit lens.

Read-only over the repo. Inputs:
  experiments/replication/results/grid_*.jsonl          (replication grid: base + other models, levels x P0/P1)
  experiments/training_eval/results/eval_base_v12.jsonl (base Tiny Aya Global, v1.2 specs, L0r/L1/L2/L2s x P0/P2)
  <session scratchpad>/hard/panel_*.jsonl               (hard-set panel: big models under P2; NOT in the repo)
  experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl (+ dataset/curated_v121/grpo_train.jsonl)
Run every script with `python3 -B` (no __pycache__ written into wtm/).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GRID_DIR = os.path.join(ROOT, "experiments", "replication", "results")
EVAL_BASE = os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl")
PANEL_DIR = ("scratch/"
             "c2ef7a67-d083-4a26-86a4-38d6c96fa533/scratchpad/hard")
SCORES = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")
GRPO = os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl")
SFT = os.path.join(ROOT, "dataset", "curated_v121", "sft_train.jsonl")
TOKSTATS = os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")
SPLITS = os.path.join(ROOT, "dataset", "splits.json")

# ---------------------------------------------------------------- decision taxonomy (wtm/episode.py classify_decision)
ASK = {"ask_informed", "ask_generic"}                       # a chat question
INFORMED_CHAT = {"ask_informed", "tell_informed"}           # names the discrepancy in chat (A strict success)
APPROVE = {"approve_informed", "approve_generic", "approve_gold"}
COMMIT = {"commit_gold", "commit_other", "commit_harmful"}
A_LITERAL = {"approve_generic", "commit_other", "commit_harmful"}   # grader's A-side harm set: acts on the literal request
B_NEEDLESS = {"ask_informed", "ask_generic", "tell_informed"}       # grader's B-side needless set
OTHER = {"invalid", "no_decision", "lookup"}
ERROR = {"error"}

METRICS_A = ["ask", "ask_informed", "informed_chat", "approve", "commit", "act", "literal_act", "statement", "other", "success"]
METRICS_B = ["ask", "needless", "approve", "commit", "act", "statement", "other", "success"]


def indicators(decision: str, twin: str, success: bool | None = None) -> dict:
    """0/1 indicators for one first decision. twin = 'A' | 'B'."""
    d = decision
    out = {
        "ask": int(d in ASK),
        "ask_informed": int(d == "ask_informed"),
        "informed_chat": int(d in INFORMED_CHAT),
        "approve": int(d in APPROVE),
        "commit": int(d in COMMIT),
        "act": int(d in APPROVE or d in COMMIT),
        "statement": int(d == "statement"),
        "other": int(d in OTHER),
    }
    if twin == "A":
        out["literal_act"] = int(d in A_LITERAL)
    else:
        out["needless"] = int(d in B_NEEDLESS)
    if success is not None:
        out["success"] = int(bool(success))
    return out


def short_model(m: str) -> str:
    if m.startswith("claude:"):
        return "claude-" + m.split(":", 1)[1]
    m = m.split("@")[0]
    for p in ("hf:", "vllm:"):
        if m.startswith(p):
            m = m[len(p):]
    m = m.split(":")[0]
    return m.split("/")[-1]


def twin_letter(t: str) -> str | None:
    if t == "A_should_ask":
        return "A"
    if t == "B_should_act":
        return "B"
    return None              # A_pushback (3 seed rows) excluded: a different construct


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- clusters (source posts) for the bootstrap
def pair_cluster_map() -> dict:
    sp = json.load(open(SPLITS))
    p2c = {}
    for c, v in sp["clusters"].items():
        for p in v["pairs"]:
            p2c[p] = c
    return p2c


def cluster_of(pair_id: str, p2c: dict) -> str:
    if pair_id in p2c:
        return p2c[pair_id]
    seed = "S" + pair_id[1:3]             # V<seed><k> and H<seed><k> are variants of seed pair S<seed>
    return p2c.get(seed, pair_id)


# ---------------------------------------------------------------- episode rows
def iter_episode_files():
    """(source_name, path) for every behaviour log this lens reads."""
    for f in sorted(glob.glob(os.path.join(GRID_DIR, "grid_*.jsonl"))):
        yield os.path.basename(f).replace(".jsonl", ""), f
    yield "eval_base_v12", EVAL_BASE
    for f in sorted(glob.glob(os.path.join(PANEL_DIR, "panel_*.jsonl"))):
        name = os.path.basename(f).replace(".jsonl", "")
        if name == "panel_hard_5models":     # subset copy of panel_hard_claude + panel_hard_open (no new episodes)
            continue
        yield name, f


def load_episodes() -> list[dict]:
    p2c = pair_cluster_map()
    rows = []
    for src, path in iter_episode_files():
        with open(path) as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                tw = twin_letter(r["twin"])
                if tw is None or r.get("mode", "decision") != "decision":
                    continue
                g = r["decision_grade"]
                rows.append({
                    "source": src, "file": os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, "line": i,
                    "model": short_model(r["model"]), "model_full": r["model"], "id": r["id"], "pair_id": r["pair_id"],
                    "cluster": cluster_of(r["pair_id"], p2c), "twin": tw, "env": r.get("env_id"), "level": r["level"],
                    "prompt": r["prompt"], "rep": r.get("rep", 0), "decision": g["decision"], "success": bool(g["success"]),
                    "tool_results": r.get("tool_results"), "conversion": r.get("conversion"),
                    "system_chars": r.get("system_chars"), "prompt_tokens": (r.get("tokens") or {}).get("prompt"),
                })
    return rows


def first_text(path: str, line: int, maxlen: int = 220) -> str:
    """The decisive raw output of an episode row (for examples)."""
    with open(path if os.path.isabs(path) else os.path.join(ROOT, path)) as fh:
        for i, l in enumerate(fh):
            if i == line:
                r = json.loads(l)
                evs = [e for e in r.get("events", []) if not e.get("narration")]
                raw = evs[-1].get("raw", "") if evs else ""
                return raw.replace("\n", " ")[:maxlen]
    return ""


# ---------------------------------------------------------------- bootstrap
def boot_mean_ci(values: np.ndarray, clusters: np.ndarray | None = None, n_boot: int = 10000, seed: int = 0):
    """Mean with a percentile bootstrap CI. If clusters is given, resample clusters (weights via multinomial counts)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        means = values[idx].mean(axis=1)
    else:
        cl = np.asarray(clusters)
        uniq, inv = np.unique(cl, return_inverse=True)
        k = len(uniq)
        s = np.bincount(inv, weights=values, minlength=k)
        c = np.bincount(inv, minlength=k).astype(float)
        w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(float)
        means = (w @ s) / np.maximum(w @ c, 1e-12)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def boot_diff_unpaired(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """mean(b) - mean(a) with independent resampling of both groups (units already aggregated per prompt/scenario)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = mb - ma
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def dump(obj, name: str):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=float)
    return path
