"""Shared helpers for the twin-contrast / label-shortcut audit (lens: twin-contrast-shortcuts).

Read-only over the repo: loads the v1.2.1 training views (dataset/curated_v121), the scenario specs, the evaluation specs and the
exact token statistics (experiments/dataset_audit/results/token_stats.jsonl). Nothing here writes outside this lens folder.
Bytecode writing is disabled so importing wtm/* never creates __pycache__ files in the repo.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

LENS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(LENS, "..", "..", ".."))
RES = os.path.join(LENS, "results")
CACHE = os.path.join(LENS, "cache")
CUR = os.path.join(ROOT, "dataset", "curated_v121")
FILES = ["sft_train", "sft_val", "dpo_train", "grpo_train"]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CUE_WORDS = {   # regex stems; the first eight are the ones the audit brief names
    "flag": r"\bflag\w*", "warning": r"\bwarn\w*", "unusual": r"\bunusual\w*", "mismatch": r"\bmis-?match\w*",
    "duplicate": r"\bduplicat\w*", "verify": r"\b(?:un)?verif\w*", "suspicious": r"\bsuspic\w*", "note": r"\bnotes?\b",
    "scam": r"\bscam\w*", "fraud": r"\bfraud\w*", "risk": r"\brisk\w*", "alert": r"\balert\w*", "caution": r"\bcaution\w*",
    "new": r"\bnew\b", "first": r"\bfirst\b", "never": r"\bnever\b", "changed": r"\bchang(?:e|ed|es)\b", "pending": r"\bpending\b",
    "failed": r"\bfail\w*", "expired": r"\bexpir\w*", "not": r"\bnot\b|n't\b", "unknown": r"\bunknown\b|\bunrecogni\w*",
    "null": r"\bnull\b", "false": r"\bfalse\b", "true": r"\btrue\b",
}
CUE_RX = {k: re.compile(v, re.I) for k, v in CUE_WORDS.items()}


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=str)


def jl_dump(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(name, base=CUR):
    rows = load_jsonl(os.path.join(base, f"{name}.jsonl"))
    for i, r in enumerate(rows):
        r["_i"], r["_file"] = i, name
        r.setdefault("turn", "decision")          # DPO / GRPO rows are decision turns (prompt = decision prompt)
    return rows


def twin_letter(t: str) -> str:
    return "P" if t.startswith("A_pushback") else t[0]


_HINT = None


def hint_text() -> str:
    global _HINT
    if _HINT is None:
        _HINT = open(os.path.join(ROOT, "prompts", "golden_short.md")).read().strip()
    return _HINT


def strip_hint(system: str) -> str:
    """wtm.render appends the P2 hint as system.rstrip() + '\\n\\n' + hint; undo it (and rstrip P0 the same way)."""
    h = hint_text()
    s = system.rstrip()
    if s.endswith(h):
        s = s[: -len(h)].rstrip()
    return s


def has_hint(system: str) -> bool:
    return system.rstrip().endswith(hint_text())


def components(prompt: list[dict]) -> dict:
    """prompt = [system, request, (assistant call, tool result)*] -> named parts."""
    out = {"system": prompt[0]["content"], "request": prompt[1]["content"] if len(prompt) > 1 else "",
           "calls": [m["content"] for m in prompt[2::2] if m["role"] == "assistant"],
           "results": [m["content"] for m in prompt[3::2] if m["role"] == "user"]}
    return out


RESULT_RX = re.compile(r'^<tool_result name="([^"]*)">\n(.*)\n</tool_result>$', re.S)
CALL_RX = re.compile(r"^<tool_call>(.*)</tool_call>$", re.S)


def parse_result(content: str):
    m = RESULT_RX.match(content.strip())
    if not m:
        return None, content
    name, payload = m.group(1), m.group(2)
    try:
        return name, json.loads(payload)
    except json.JSONDecodeError:
        return name, payload


def parse_call(content: str):
    m = CALL_RX.match(content.strip())
    if not m:
        return None, {}
    try:
        obj = json.loads(m.group(1))
        return obj.get("name"), obj.get("arguments", {})
    except json.JSONDecodeError:
        return None, {"_raw": m.group(1)}


def flatten(obj, prefix=""):
    """JSON -> {path: leaf}; list items are indexed; empty containers are leaves ([] / {})."""
    out = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix or "$"] = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj:
            out[prefix or "$"] = []
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix or "$"] = obj
    return out


def generic_path(p: str) -> str:
    return re.sub(r"\[\d+\]", "[]", p)


SECTION_RX = re.compile(r"^(#{2,3}) (.*)$", re.M)


def system_sections(system: str) -> list[tuple[int, int, str]]:
    """[(start, end, section_name)] over the system prompt; '### x' sections are named '## Member / x'. The P2 hint and the
    L2s 'Most relevant records' block are separate sections."""
    marks = []
    top = "preamble"
    for m in SECTION_RX.finditer(system):
        name = m.group(2).strip()
        if m.group(1) == "##":
            top = name
            marks.append((m.start(), top))
        else:
            marks.append((m.start(), f"{top} / {re.sub(r' [(].*', '', name)}"))
    h = hint_text()
    hpos = system.rfind(h)
    if hpos >= 0 and system.rstrip().endswith(h):
        marks.append((hpos, "P2 hint"))
    marks = sorted(marks)
    out = []
    starts = [0] + [p for p, _ in marks]
    names = ["preamble"] + [n for _, n in marks]
    for i, (s, n) in enumerate(zip(starts, names)):
        e = starts[i + 1] if i + 1 < len(starts) else len(system)
        if e > s:
            out.append((s, e, n))
    return out


def section_at(sections, pos):
    for s, e, n in sections:
        if s <= pos < e:
            return n
    return sections[-1][2] if sections else "?"


def member_block(system: str) -> str:
    s = strip_hint(system)
    a = s.find("## Member")
    if a < 0:                         # L0r: the seed's member card sits under its own header in the L0 template
        a = 0
    b = s.find("## Tools", a)
    return s[a:b if b > 0 else len(s)]


def load_token_stats():
    path = os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")
    out = {}
    for r in load_jsonl(path):
        out[(r["file"].replace(".jsonl", ""), r["i"])] = r
    return out


def load_specs():
    """Every scenario spec by id: training/val/test pairs (dataset/scenarios + scenarios_synth) and the hard set (dataset/eval)."""
    specs, origin = {}, {}
    for pat, tag in (("dataset/scenarios/*.jsonl", "seed"), ("dataset/scenarios_synth/*.jsonl", "synth")):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            for s in load_jsonl(f):
                specs[s["id"]] = s
                origin[s["id"]] = tag
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
        name = os.path.basename(f).rsplit("_", 1)[0]          # test / val / hard_test / hard_val / hard_train
        for s in load_jsonl(f):
            if s["id"] not in specs:
                specs[s["id"]] = s
                origin[s["id"]] = name
    return specs, origin


def load_split():
    sp = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
    return {pid: k for k in ("train", "val", "test") for pid in sp[k]}


def eval_specs(name: str) -> list[dict]:
    return [s for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", f"{name}_*.jsonl"))) for s in load_jsonl(f)]


WORD_RX = re.compile(r"[a-z]+|\d+|[@$€£₹#]", re.I)


def words(text: str, digits_to_zero=True) -> list[str]:
    toks = WORD_RX.findall(text.lower())
    if digits_to_zero:
        toks = ["0" if t.isdigit() else t for t in toks]
    return toks


def norm_txt(t) -> str:
    t = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
    t = t.lower().replace(",", "").replace("’", "'")
    t = re.sub(r"[$€£₹]", "", t)
    return re.sub(r"\s+", " ", t)
