"""Build SFT / DPO / GRPO datasets from scenario specs (+ observed base-model failures for DPO rejected sides).

Two kinds of supervised turns per scenario rendering (both in the shared text protocol, wtm/protocol.py):
  lookup turns  : [system, member request, earlier look-ups + results]        → next canonical lookup (scenario.prefix_calls[i])
  decision turn : [system, request, <tool_call> lookup, <tool_result> raw]    → scenario.ideal_action
Base Tiny Aya skipped lookups entirely in smoke runs, so the lookup turn matters: it teaches "check before you move money".
Renderings: levels L0r / L1 / L2 / L3 × prompts P0 (no instruction, ~70 %) / P2 (a one-or-two-line intent hint, ~30 %). User,
2026-11-13: the model must act on what members mean with NO instructions or a 1-2 line hint, so the long golden block (P1) is never
used in training (it stays an evaluation reference). Targets come from the scenario specs only (never written by Claude).
Splits come from dataset/splits.json ({"train": [pair_ids], "val": [...], "test": [...]}); test pairs and eval-only (hard-set)
rows are never written here.

v1.2.1 consistency rules (training-pipeline review 2026-11-14), every drop is listed in <out>/build_report.json:
  one prompt, one target  SFT rows are grouped by a whitespace/Unicode-normalised prompt, and again with the P2 hint removed (the
                          hint does not change the right answer). A group with two different targets is dropped when all its rows
                          are look-up turns (S42: the twins differ in which action the agent previews, which the prompt cannot
                          predict) and fails the build when any row is a decision turn (a label contradiction).
  pushback rows           A_pushback rows (S07P, S49P, ...) render exactly like their A twin but carry other hidden labels (no
                          gold commit): left out of DPO and GRPO, and out of SFT where they duplicate the A twin's rows.
  GRPO labels             rows sharing a prompt must carry the same expected decision and gold_commits (the reward reads them).
  identifiers             every identifier in a twin-A message target (email/UPI id, handle, phone, ISO date, masked number,
                          order/invoice id) must appear in that row's prompt; otherwise the (scenario, level) leaves SFT and DPO,
                          because the target would teach the model to cite facts it was never shown. KNOWN_BAD_TARGETS adds
                          scenarios whose target misquotes an identifier that the rule cannot catch (the misquoted string occurs
                          elsewhere in the prompt, e.g. in member notes); the build fails once the spec is fixed (the target no
                          longer cites it, or a look-up result now shows it), so the entry goes away with the spec fix. V344A was
                          the only entry; dataset/fix_v121.py fixed its spec.
  DPO rejected            an observed base-model failure is used only when it was produced on the same scenario and level and
                          wtm.reward (the GRPO reward) scores it <= 0; otherwise the synthetic rejected (also required <= 0).
                          Failure files are read in sorted path order (the first usable failure per scenario and level wins), so
                          the build is identical on any filesystem. dpo_rejected_source in the report counts the written rows.

Reproduces v1.2.1 (both scenario globs are needed: the seed pairs and the synthetic variants; seed 20261112, hint share 0.30):
python3 training/build_datasets.py --scenarios 'dataset/scenarios/*.jsonl' 'dataset/scenarios_synth/*.jsonl' \
    --splits dataset/splits.json --failures 'experiments/replication/results/grid_*.jsonl' --out training/data
→ sft_train.jsonl / sft_val.jsonl (prompt-completion, conversational), dpo_train.jsonl, grpo_train.jsonl (prompt + scenario JSON),
  build_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.protocol import tool_result_message  # noqa: E402
from wtm.render import golden_text, render  # noqa: E402
from wtm.reward import reward  # noqa: E402
from wtm.sim import Simulator  # noqa: E402
from wtm.spec import load_env, load_jsonl, load_member  # noqa: E402

LEVELS_TRAIN = ["L0r", "L1", "L2", "L3"]
BASE_MODELS = ("hf:CohereLabs", "vllm:tiny")
IDENTIFIERS = [   # kind, pattern (group 1, when present, is the part that must appear in the prompt)
    ("email_or_upi", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*")),
    ("handle", re.compile(r"(?<![\w.@])@\w[\w.-]*\w")),
    ("phone", re.compile(r"(?<![\w-])(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -])?\d{3}-\d{4}(?![\w-])")),
    ("iso_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("masked_number", re.compile(r"(?:\*{2,}|[xX]{2,}|\.\.\.|…)\s?([0-9A-Za-z]{3,})")),
    ("order_id", re.compile(r"(?i)\b(?:order|invoice|ticket|case|ref|reference|transaction|txn)\s*(?:#|no\.?|number|id)?\s*:?\s*"
                            r"(#?[A-Z]{0,4}-?\d{3,}[A-Z0-9-]*)")),
]
KNOWN_BAD_TARGETS: dict[str, tuple[str, str]] = {}   # scenario id -> (identifier the target misquotes, why); out of SFT and DPO


def call_text(name: str, args: dict) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def action_text(a: dict) -> str:
    return a["text"] if a["type"] == "message" else call_text(a["name"], a.get("arguments", {}))


def prefix_messages(env, member, scn, level, prompt):
    system, user, tools = render(env, member, scn, level, prompt)
    sim = Simulator(env, member, scn, tools, level=level)
    msgs = [{"role": "user", "content": user}]
    for c in scn.get("prefix_calls", []):
        msgs.append({"role": "assistant", "content": call_text(c["tool"], c["arguments"])})
        msgs.append({"role": "user", "content": tool_result_message(c["tool"], sim.execute(c["tool"], c["arguments"]))})
    return system, user, msgs


def synthetic_rejected(scn: dict) -> str:
    """A realistic wrong decision when no usable observed base-model failure exists."""
    if scn["twin"].startswith("A"):
        g = (scn.get("gold_commits") or [{}])[0]
        return call_text("request_approval", {"summary": f"Proceed with your request: {scn['user_message']}"}) if g else \
            "Done — I've taken care of that for you."
    return "Before I go ahead, can you confirm this is definitely what you want to do?"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def prompt_key(prompt: list[dict], hint: str = "") -> str:
    """Normalised prompt; with `hint`, that text is removed from the system message first (P2 vs P0 of the same rendering)."""
    return json.dumps([[m["role"], norm(m["content"].replace(hint, "") if hint and m["role"] == "system" else m["content"])]
                       for m in prompt], ensure_ascii=False)


def target_of(r: dict) -> str:
    return (r.get("completion") or r.get("chosen"))[0]["content"]


def missing_identifiers(target: str, prompt: list[dict]) -> list[list[str]]:
    text = norm("\n".join(m["content"] for m in prompt))
    found = {(kind, (m.group(1) if rx.groups else m.group(0)).rstrip(".")) for kind, rx in IDENTIFIERS for m in rx.finditer(target)}
    return sorted([k, v] for k, v in found if norm(v) not in text)


def hidden_label(scn: dict) -> str:
    return json.dumps([scn["expected"].get("decision"), scn.get("gold_commits") or []], sort_keys=True)


def load_failures(patterns: list[str]) -> dict:
    """(scenario id, level) → last raw turns of failed base-model episodes, in a fixed file order."""
    fails = defaultdict(list)
    for p in patterns:
        for f in sorted(glob.glob(p)):
            for r in load_jsonl(f):
                if not r["decision_grade"]["success"] and r.get("events") and r["model"].startswith(BASE_MODELS):
                    last = r["events"][-1].get("raw")
                    if last:
                        fails[(r["id"], r["level"])].append(last)
    return fails


def comp(rows: list[dict]) -> dict:
    return {"n": len(rows), "twin": dict(sorted(Counter(r["twin"][:1] for r in rows).items())),
            "level": dict(sorted(Counter(r["level"] for r in rows).items())),
            "prompt_setting": dict(sorted(Counter(r["prompt_setting"] for r in rows).items())),
            "turn": dict(sorted(Counter(r.get("turn", "-") for r in rows).items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--failures", nargs="*", default=[])
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--hint-share", type=float, default=0.30, help="share of renderings with the 1-2 line hint (P2); rest P0")
    ns = ap.parse_args()
    rng = random.Random(20261112)
    scns = [s for p in ns.scenarios for f in sorted(glob.glob(p)) for s in load_jsonl(f)]
    splits = json.load(open(ns.splits))
    pair_split = {pid: k for k in ("train", "val", "test") for pid in splits[k]}
    eval_only = sorted(s["id"] for s in scns if s.get("eval_only"))
    if eval_only:
        raise SystemExit(f"BUILD_FAIL: eval-only (hard-set) rows given as training scenarios: {eval_only[:5]}")
    by_id = {s["id"]: s for s in scns}
    fails = load_failures(ns.failures)
    envs, members = {}, {}
    out = {k: [] for k in ("sft_train", "sft_val", "dpo_train", "grpo_train")}
    rejected_src = {}   # id(dpo row) -> where its rejected side came from
    skipped = 0
    for s in scns:
        split = pair_split.get(s["pair_id"])
        if split not in ("train", "val") or not s.get("ideal_action"):
            skipped += split in ("train", "val")
            continue
        env = envs.setdefault(s["env_id"], load_env(s["env_id"]))
        mk = (s["env_id"], s["member_id"])
        mem = members.setdefault(mk, load_member(*mk))
        kinds = {t["name"]: t["kind"] for t in env["tools"]}
        for lv in LEVELS_TRAIN:
            if lv == "L3" and not s.get("user_message_noisy"):
                continue                    # variants have no noisy rewrite: L3 would duplicate L2
            prompt = "P2" if rng.random() < ns.hint_share else "P0"
            system, user, msgs = prefix_messages(env, mem, s, lv, prompt)
            meta = {"scenario_id": s["id"], "pair_id": s["pair_id"], "twin": s["twin"], "env_id": s["env_id"], "level": lv,
                    "prompt_setting": prompt}
            sysmsg = [{"role": "system", "content": system}]
            target = action_text(s["ideal_action"])
            for i, c in enumerate(s.get("prefix_calls") or []):   # every look-up step (a few rows need 2–3 look-ups)
                out[f"sft_{split}"].append({"prompt": sysmsg + msgs[:1 + 2 * i], "completion": [{"role": "assistant",
                                            "content": call_text(c["tool"], c["arguments"])}], **meta, "turn": f"lookup{i + 1}"})
            out[f"sft_{split}"].append({"prompt": sysmsg + msgs, "completion": [{"role": "assistant", "content": target}],
                                        **meta, "turn": "decision"})
            if split == "train":
                observed = fails.get((s["id"], lv), [])
                rej = next((f for f in observed if reward(f, s, kinds)[0] <= 0), None)
                why = "observed_failure"
                if rej is None:
                    rej = synthetic_rejected(s)
                    why = "synthetic_failures_scored_above_0" if observed else "synthetic_no_failure_same_level"
                    if reward(rej, s, kinds)[0] > 0:
                        raise SystemExit(f"BUILD_FAIL: synthetic rejected for {s['id']} {lv} scores > 0: {rej!r}")
                out["dpo_train"].append({"prompt": sysmsg + msgs, "chosen": [{"role": "assistant", "content": target}],
                                         "rejected": [{"role": "assistant", "content": rej}], **meta})
                rejected_src[id(out["dpo_train"][-1])] = why
                out["grpo_train"].append({"prompt": sysmsg + msgs, "scenario": json.dumps(s, ensure_ascii=False),
                                          "tool_kinds": json.dumps(kinds), **meta})

    hint = golden_text("P2")
    dropped = []

    def drop(view: str, bad, reason: str, detail=None):
        keep = []
        for r in out[view]:
            if bad(r):
                dropped.append({"view": view, "scenario_id": r["scenario_id"], "level": r["level"],
                                "prompt_setting": r["prompt_setting"], "turn": r.get("turn"), "reason": reason}
                               | ({"detail": detail(r)} if detail else {}))
            else:
                keep.append(r)
        out[view] = keep

    # pushback rows: same rendering as the A twin, other hidden labels
    push = {s["id"] for s in scns if s["twin"].startswith("A_pushback")
            and hidden_label(s) != hidden_label(by_id.get(s["pair_id"] + "A", s))}
    a_rows = {(r["pair_id"], prompt_key(r["prompt"], hint), norm(target_of(r)))
              for v in ("sft_train", "sft_val") for r in out[v] if r["scenario_id"] == r["pair_id"] + "A"}
    for v in ("dpo_train", "grpo_train"):
        drop(v, lambda r: r["scenario_id"] in push, "pushback row: hidden labels differ from the A twin's")
    for v in ("sft_train", "sft_val"):
        drop(v, lambda r: r["scenario_id"] in push and (r["pair_id"], prompt_key(r["prompt"], hint), norm(target_of(r))) in a_rows,
             "pushback row duplicates the A twin's row (hint-stripped prompt and target)")

    # identifiers cited by twin-A message targets must be visible in the prompt
    unseen = {}
    for v in ("sft_train", "sft_val"):
        for r in out[v]:
            if r["turn"] == "decision" and r["twin"].startswith("A") and by_id[r["scenario_id"]]["ideal_action"]["type"] == "message":
                miss = missing_identifiers(target_of(r), r["prompt"])
                if r["scenario_id"] in KNOWN_BAD_TARGETS:
                    ident, _ = KNOWN_BAD_TARGETS[r["scenario_id"]]
                    results = norm("\n".join(m["content"] for m in r["prompt"][3::2]))   # look-up results after system, request
                    if ident not in target_of(r) or norm(ident) in results:
                        raise SystemExit(f"BUILD_FAIL: KNOWN_BAD_TARGETS[{r['scenario_id']!r}] is stale (at {r['level']} the target "
                                         f"no longer cites {ident} or a look-up result now shows it); remove the entry")
                    miss = miss + [["known_bad", ident]]
                if miss:
                    unseen[(r["scenario_id"], r["level"])] = miss
    for v in ("sft_train", "sft_val", "dpo_train"):
        drop(v, lambda r: (r["scenario_id"], r["level"]) in unseen, "target cites identifiers the prompt never shows, or misquotes one (KNOWN_BAD_TARGETS)",
             lambda r: unseen[(r["scenario_id"], r["level"])])

    # one prompt → one target (SFT, train and val together), exact and with the P2 hint removed
    conflicts, contradictions, conflicted, seen = [], [], set(), set()
    for h in ("", hint):
        groups = defaultdict(list)
        for v in ("sft_train", "sft_val"):
            for r in out[v]:
                groups[prompt_key(r["prompt"], h)].append(r)
        for rows in groups.values():
            ids = frozenset(id(r) for r in rows)
            if len({norm(target_of(r)) for r in rows}) > 1 and ids not in seen:
                seen.add(ids)
                desc = sorted([r["scenario_id"], r["level"], r["prompt_setting"], r["turn"], target_of(r)[:120]] for r in rows)
                lookups_only = all(r["turn"].startswith("lookup") for r in rows)
                (conflicts if lookups_only else contradictions).append({"hint_stripped": bool(h), "rows": desc})
                if lookups_only:
                    conflicted.update(id(r) for r in rows)
    if contradictions:
        raise SystemExit("BUILD_FAIL: one prompt maps to different targets and a decision turn is involved:\n"
                         + json.dumps(contradictions, indent=1, ensure_ascii=False))
    for v in ("sft_train", "sft_val"):
        drop(v, lambda r: id(r) in conflicted, "look-up turn whose prompt also leads to a different look-up (twin previews differ)")

    # GRPO: rows sharing a prompt must share the hidden label the reward reads
    for h in ("", hint):
        labels = defaultdict(set)
        for r in out["grpo_train"]:
            labels[prompt_key(r["prompt"], h)].add((r["scenario_id"], hidden_label(json.loads(r["scenario"]))))
        bad = [sorted(x) for x in labels.values() if len({lab for _, lab in x}) > 1]
        if bad:
            raise SystemExit(f"BUILD_FAIL: GRPO prompts with different hidden labels (hint stripped: {bool(h)}): {bad[:3]}")

    leaks = sorted({r["pair_id"] for k, rows in out.items() for r in rows
                    if pair_split.get(r["pair_id"]) != ("val" if k == "sft_val" else "train")})
    if leaks:
        raise SystemExit(f"BUILD_FAIL: pairs outside their split's view (test/val leak): {leaks[:10]}")

    os.makedirs(ns.out, exist_ok=True)
    for k, rows in out.items():
        rng.shuffle(rows)
        with open(os.path.join(ns.out, f"{k}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    src = Counter(rejected_src[id(r)] for r in out["dpo_train"])
    synthetic_rows = sorted([r["scenario_id"], r["level"], rejected_src[id(r)]] for r in out["dpo_train"]
                            if rejected_src[id(r)] != "observed_failure")
    report = {"args": vars(ns), "counts": {k: len(v) for k, v in out.items()}, "skipped_missing_ideal_action": skipped,
              "dpo_rejected_source": dict(sorted(src.items())), "dpo_synthetic_rejected": synthetic_rows,
              "composition": {k: comp(v) for k, v in out.items()},
              "dropped_by_reason": dict(Counter(f"{d['view']}: {d['reason']}" for d in dropped)),
              "identifier_skips": [{"scenario_id": k[0], "level": k[1], "missing": v} for k, v in sorted(unseen.items())],
              "known_bad_targets": KNOWN_BAD_TARGETS,
              "lookup_conflicts": conflicts, "pushback_rows_with_other_labels": sorted(push), "dropped": dropped}
    json.dump(report, open(os.path.join(ns.out, "build_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: len(v) for k, v in out.items()} | {"skipped_missing_ideal_action": skipped,
                                                            "dpo_rejected_source": dict(sorted(src.items())), "dropped": len(dropped)}))


if __name__ == "__main__":
    main()
