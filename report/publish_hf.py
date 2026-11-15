"""Publish the hackathon artefacts to Hugging Face: model repos (LoRA adapters + cards), one dataset repo, two collections.

Names and visibility were confirmed by the user (2026-11-14): owner sharan-nagarajan, PUBLIC. Dry run by default: checks that
every source exists, scans every file for credentials (an HF token pattern or the .env contents) and prints what would be
uploaded. --go creates the repos (exist_ok) and uploads. Run inside lx-exp (adapters live under /workspace/runs; HF_TOKEN is in the
container env) from /workspace:
  python3 report/publish_hf.py --plan report/publish_plan_models.json     # dry run (inside lx-exp); data plan: on the Mac
  python3 report/publish_hf.py --plan report/publish_plan_models.json --go
Plan (JSON): {"owner": "...", "models": [{"repo", "src", "card"}], "dataset": {"repo", "card", "items": [{"src", "dst"}]},
"collections": [{"title", "description", "items": ["model:<repo>" | "dataset:<repo>"]}]}
Prints PUBLISH_PLAN_OK (dry run) or PUBLISH_DONE <urls> (--go).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

TOKEN_RE = re.compile(rb"hf_[A-Za-z0-9]{30,}")
# Private details that must never be published (the Claude Code account e-mail's domain, other services on the shared GPU box,
# credentials): regexes, one per line, in a LOCAL git-ignored file (report/.private_patterns) or $WTM_PRIVATE_PATTERNS — never in
# this public file. Missing file = only the token checks below.
def _private_re():
    src = os.environ.get("WTM_PRIVATE_PATTERNS")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".private_patterns")
    if not src and os.path.exists(path):
        src = "|".join(l.strip() for l in open(path) if l.strip() and not l.startswith("#"))
    # generic credential prefixes; the [t] keeps this source line from matching itself in the pre-push scan
    pats = [p for p in (src or "").split("|") if p] + [r"github_pa[t]_", r"ghp_[A-Za-z0-9]{20,}", r"sk-an[t]-"]
    return re.compile("|".join(f"(?:{p})" for p in pats).encode(), re.I)


PRIVATE_RE = _private_re()
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "stack_info.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "chat_template.jinja")


def files_of(src: str, dst: str) -> list[tuple[str, str]]:
    if os.path.isdir(src):
        out = []
        for p in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)):
            if os.path.isfile(p) and "/cache/" not in p and "__pycache__" not in p:
                out.append((p, os.path.join(dst, os.path.relpath(p, src))))
        return out
    return [(src, dst)]


def scan(path: str, secrets: list[bytes]) -> str | None:
    if path.endswith(".safetensors"):
        return None   # binary weights: the token pattern is not meaningful there (and adapters never contain one)
    try:
        data = open(path, "rb").read()
    except OSError as e:
        return f"unreadable: {e}"
    if TOKEN_RE.search(data):
        return "contains an hf_ token pattern"
    m = PRIVATE_RE.search(data)
    if m:
        return f"contains a private detail ({m.group(0)[:3].decode(errors='replace')}…)"
    for s in secrets:
        if len(s) >= 12 and s in data:
            return "contains a value from .env"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--go", action="store_true")
    ns = ap.parse_args()
    plan = json.load(open(ns.plan))
    owner = plan["owner"]
    secrets = []
    for envf in (".env", "/workspace/.env"):
        if os.path.exists(envf):
            for line in open(envf, "rb"):
                if b"=" in line:
                    v = line.split(b"=", 1)[1].strip().strip(b"'\"")
                    if v:
                        secrets.append(v)
    problems, uploads = [], []
    for m in plan.get("models", []):
        if not os.path.isdir(m["src"]):
            problems.append(f"model {m['repo']}: missing {m['src']}")
            continue
        fl = [(os.path.join(m["src"], f), f) for f in ADAPTER_FILES if os.path.exists(os.path.join(m["src"], f))]
        if not any(f == "adapter_model.safetensors" for _, f in fl):
            problems.append(f"model {m['repo']}: no adapter_model.safetensors in {m['src']}")
        if m.get("card"):
            fl.append((m["card"], "README.md"))
        uploads.append(("model", f"{owner}/{m['repo']}", fl))
    ds = plan.get("dataset")
    if ds:
        fl = []
        for it in ds["items"]:
            if not os.path.exists(it["src"]):
                problems.append(f"dataset: missing {it['src']}")
                continue
            fl += files_of(it["src"], it["dst"])
        if ds.get("card"):
            fl.append((ds["card"], "README.md"))
        uploads.append(("dataset", f"{owner}/{ds['repo']}", fl))
    total = 0
    for kind, repo, fl in uploads:
        size = sum(os.path.getsize(s) for s, _ in fl)
        total += size
        print(f"{kind:7s} {repo}: {len(fl)} files, {size / 1e6:.1f} MB")
        for s, d in fl:
            why = scan(s, secrets)
            if why:
                problems.append(f"{repo}: {s} {why}")
    print(f"total {total / 1e6:.1f} MB")
    for c in plan.get("collections", []):
        print(f"collection '{c['title']}': {c['items']}")
    if problems:
        print("PUBLISH_PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    if not ns.go:
        print("PUBLISH_PLAN_OK (dry run; rerun with --go to publish)")
        return
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:   # the repo's .env (git-ignored) holds the token on a non-shell-safe line; read it here, never print it
        for envf in (".env", "/workspace/.env"):
            if os.path.exists(envf):
                for line in open(envf):
                    k, _, v = line.partition("=")
                    if k.strip().upper().replace("-", "_") in ("HF_TOKEN", "HUGGINGFACE_TOKEN") and v.strip():
                        token = v.strip().strip("'\"")
                        break
            if token:
                break
    api = HfApi(token=token)
    urls = []
    from huggingface_hub import CommitOperationAdd
    for kind, repo, fl in uploads:
        api.create_repo(repo, repo_type=kind, private=False, exist_ok=True)
        # ONE commit per repository: the Hub allows 128 commits per hour per account (2026-11-15: a file-per-commit upload of
        # the dataset hit that limit after ~128 files).
        ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=s) for s, d in fl]
        api.create_commit(repo_id=repo, repo_type=kind, operations=ops,
                          commit_message=f"What They Meant: {len(ops)} files")
        urls.append(f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")
        print(f"UPLOADED {repo} ({len(fl)} files)", flush=True)
    for c in plan.get("collections", []):
        col = api.create_collection(title=c["title"], description=c.get("description", ""), namespace=owner, private=False,
                                    exists_ok=True)
        for it in c["items"]:
            kind, repo = it.split(":", 1)
            api.add_collection_item(col.slug, item_id=f"{owner}/{repo}", item_type=kind, exists_ok=True)
        urls.append(f"https://huggingface.co/collections/{col.slug}")
        print(f"COLLECTION {col.slug}", flush=True)
    print("PUBLISH_DONE " + " ".join(urls))


if __name__ == "__main__":
    main()
