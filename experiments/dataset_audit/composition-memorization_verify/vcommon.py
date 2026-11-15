"""Shared loaders for the composition-memorization VERIFY pass (independent of the analyst's common.py).
Read-only access to dataset/ and experiments/; outputs go to this folder's results/."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CUR = os.path.join(ROOT, "dataset", "curated_v121")
sys.path.insert(0, ROOT)


def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def cur(name):
    return load_jsonl(os.path.join(CUR, name + ".jsonl"))


def target(r):
    c = r.get("completion") or r.get("chosen")
    return c[0]["content"]


def request(r):
    """the member's request = first user message after the system prompt"""
    return r["prompt"][1]["content"]


def is_call(text):
    return text.lstrip().startswith("<tool_call>")


def call_name(text):
    if not is_call(text):
        return None
    body = text.strip()[len("<tool_call>"):]
    body = body[: body.rfind("</tool_call>")]
    return json.loads(body)["name"]


def dump(obj, name):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    print("WROTE", p)
    return p
