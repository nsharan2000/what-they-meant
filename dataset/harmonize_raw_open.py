"""Twin schema harmonisation of raw tool outputs (result_raw), with an OPEN model (no Claude) and code-level validation.

Problem (log 22:3x): the raw API outputs of twin A and twin B were written independently, so for the same tool they often use
different field names (A: unit_price_inr / total_price_inr, B: price_inr / line_total_inr). A trained model could learn the field
names as a shortcut for "ask" vs "proceed". A real API returns ONE schema whatever the data.

For every pair and every tool that has rules in both twins, the model rewrites all of that tool's responses (A's and B's) into one
shared schema: same field names, nesting and order; values preserved (renaming allowed); a field that only makes sense for one twin
appears in the other with null / [] / false / 0 as a real API would return. Validation (retry ×2 with feedback, else keep the original):
  * valid JSON, same number of responses, no verdict words (wtm.invariants.VERDICT_RE)
  * one key structure across every response of that tool (null / [] count as 'not applicable'; list items share keys)
  * every scalar value of each original response (strings ≥ 3 chars, numbers) still appears in its rewrite
  * after applying all tools of the pair: planted signal at L0r / L1 / L2 (oracle passes both twins; always-approve fails A;
    always-ask and keyword-stuffer fail B) — otherwise the whole pair is reverted.
Pushback rows (P) whose rule outputs equal A's originals get A's rewrite. Writes in place; report → dataset/harmonize_<set>.json.

python3 dataset/harmonize_raw_open.py --set synth|seed [--model Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra | claude:sonnet] [--workers 12]
(2026-11-13: the user limited HF credits and asked for Claude Code wherever possible → the seed pass uses claude:sonnet through the pure
Claude driver; rewriting tool OUTPUTS is environment building, not a training target, so it is outside the Q5 restriction.)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.drivers.openai_compat import OpenAICompat  # noqa: E402
from wtm.episode import run_episode  # noqa: E402
from wtm.invariants import VERDICT_RE  # noqa: E402
from wtm.policies import ScriptedPolicy  # noqa: E402
from wtm.render import render  # noqa: E402
from wtm.spec import ROOT, load_env, load_jsonl, load_member  # noqa: E402

SYSTEM = """You maintain the JSON responses of a payment app's internal API. One endpoint must always return ONE schema: the same
field names, nesting and field order, whatever the data. You rewrite responses into a shared schema without changing any fact.
You output JSON only."""

PROMPT = """Endpoint: {tool} — {desc}
Below are responses this endpoint returned in different situations. They use inconsistent field names / structure.
Rewrite ALL of them into ONE shared schema:
- identical field names, nesting and field order in every response (items inside lists also share one set of fields);
- keep every value exactly (amounts, names, ids, dates, statuses, counts) — you may rename a field to the shared name;
- a field that only applies in some situations must still appear in the others with null, [], false or 0, as a real API would;
- do not add commentary fields, judgements or new facts; never use words like match, mismatch, duplicate, verified, suspicious,
  scam, fraud, warning, risk, safe, wrong, correct.
{feedback}
Responses:
{responses}

Return ONLY: {{"responses": [<rewritten response 1>, <rewritten response 2>, ...]}} in the same order."""


def compatible(a, b) -> bool:
    """Same schema, where null / [] / scalar placeholders are compatible with anything ("not applicable" values)."""
    if a is None or b is None or a == [] or b == []:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(compatible(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        items = [x for x in a + b if x not in (None, [])]
        return all(compatible(items[0], x) for x in items[1:]) if items else True
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False
    return True                                   # two scalars (types may differ: 0 vs 12.5)


def same_structure(objs) -> bool:
    return all(compatible(objs[0], o) for o in objs[1:]) and all(compatible(o, o) for o in objs)


def scalars(x, out=None):
    out = [] if out is None else out
    if isinstance(x, dict):
        for v in x.values():
            scalars(v, out)
    elif isinstance(x, list):
        for v in x:
            scalars(v, out)
    elif isinstance(x, bool) or x is None:
        pass
    elif isinstance(x, (int, float)):
        out.append(x)
    elif isinstance(x, str) and len(x.strip()) >= 3:
        out.append(x.strip())
    return out


def preserved(orig, new) -> list:
    blob = json.dumps(new, ensure_ascii=False)
    nums = {round(float(n), 2) for n in re.findall(r"-?\d+(?:\.\d+)?", blob.replace(",", ""))}
    missing = []
    for v in scalars(orig):
        if isinstance(v, (int, float)):
            if round(float(v), 2) not in nums:
                missing.append(v)
        elif v not in blob and json.dumps(v, ensure_ascii=False)[1:-1] not in blob:
            missing.append(v)
    return missing


def as_obj(raw):
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def planted_ok(env, members, A, B) -> str:
    for lv in ("L0r", "L1", "L2"):
        for pol, want in (("oracle", (True, True)), ("always_approve", (False, None)), ("always_ask", (None, False)),
                          ("kw_stuffer", (None, False))):
            for row, w in zip((A, B), want):
                if w is None:
                    continue
                mem = members[row["member_id"]]
                system, user, tl = render(env, mem, row, lv, "P0")
                g = run_episode(env, mem, row, ScriptedPolicy(pol, row), system, user, tl, mode="decision", level=lv)["decision_grade"]
                if bool(g["success"]) != w:
                    return f"{pol} on {row['id']} at {lv} -> {g['decision']}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["seed", "synth"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra")
    ap.add_argument("--cache", default=os.environ.get("WTM_CACHE", "/tmp/wtm_cache/harmonize"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    folder = "scenarios" if ns.set == "seed" else "scenarios_synth"
    files = sorted(glob.glob(os.path.join(ROOT, "dataset", folder, "*.jsonl")))
    rows_by_file = {f: load_jsonl(f) for f in files}
    pairs = {}
    for f, rows in rows_by_file.items():
        for r in rows:
            t = "P" if "push" in r["twin"] else r["twin"][0]
            pairs.setdefault(r["pair_id"], {})[t] = r
    envs, members = {}, {}
    if ns.model.startswith("claude:"):
        from wtm.drivers.claude_pure import ClaudePure
        drv = ClaudePure(model=ns.model.split(":", 1)[1])
    else:
        drv = OpenAICompat(model=ns.model, temperature=0.2, max_tokens=4096, cache_dir=ns.cache)

    def tool_desc(env, name):
        return next((t.get("description", "") for t in env["tools"] if t["name"] == name), "")

    def do_pair(pid):
        p = pairs[pid]
        if "A" not in p or "B" not in p:
            return pid, "skip_incomplete", {}
        A, B = p["A"], p["B"]
        env = envs.setdefault(A["env_id"], load_env(A["env_id"]))
        for r in (A, B):
            members.setdefault(r["member_id"], load_member(r["env_id"], r["member_id"]))
        tools = sorted({x["tool"] for x in A["tool_rules"]} & {x["tool"] for x in B["tool_rules"]})
        newA, newB = copy.deepcopy(A), copy.deepcopy(B)
        changed, notes = 0, {}
        for tool in tools:
            slots = [(newA, i) for i, x in enumerate(newA["tool_rules"]) if x["tool"] == tool and as_obj(x.get("result_raw")) is not None] + \
                    [(newB, i) for i, x in enumerate(newB["tool_rules"]) if x["tool"] == tool and as_obj(x.get("result_raw")) is not None]
            origs = [as_obj(row["tool_rules"][i]["result_raw"]) for row, i in slots]
            if len(origs) < 2 or same_structure(origs):
                continue
            # identical originals (shared rules) are rewritten once
            uniq = []
            for o in origs:
                if o not in uniq:
                    uniq.append(o)
            feedback, ok = "", None
            for attempt in range(3):
                prompt = PROMPT.format(tool=tool, desc=tool_desc(env, tool)[:300], feedback=feedback,
                                       responses="\n".join(f"{k + 1}. {json.dumps(o, ensure_ascii=False)}" for k, o in enumerate(uniq)))
                res = drv.chat(SYSTEM, [{"role": "user", "content": prompt + (f"\n(attempt {attempt + 1})" if attempt else "")}])
                m = re.search(r"\{.*\}", res.text or "", re.S)
                try:
                    new = json.loads(m.group(0))["responses"] if m else None
                except (json.JSONDecodeError, KeyError, TypeError):
                    new = None
                if not isinstance(new, list) or len(new) != len(uniq):
                    feedback = "Your previous answer was not a JSON object with a 'responses' list of the same length."
                    continue
                if not same_structure(new):
                    feedback = "Your previous answer still used different field names or structure between responses."
                    continue
                hit = VERDICT_RE.search(json.dumps(new, ensure_ascii=False))
                if hit:
                    feedback = f"Your previous answer used the forbidden word {hit.group(0)!r}."
                    continue
                miss = [v for o, n in zip(uniq, new) for v in preserved(o, n)]
                if miss:
                    feedback = f"Your previous answer lost these values: {miss[:6]}"
                    continue
                ok = new
                break
            if ok is None:
                notes[tool] = f"kept original ({feedback or 'no valid rewrite'})"
                continue
            for (row, i), o in zip(slots, origs):
                row["tool_rules"][i]["result_raw"] = ok[uniq.index(o)]
            changed += 1
            notes[tool] = "harmonised"
        if not changed:
            return pid, "unchanged" if not notes else "kept_original", notes
        why = planted_ok(env, members, newA, newB)
        if why:
            return pid, "reverted_planted", {"planted": why, **notes}
        return pid, "harmonised", {"rows": (newA, newB), **notes}

    report, updates = {}, {}
    with cf.ThreadPoolExecutor(max_workers=ns.workers) as ex:
        for pid, status, info in ex.map(do_pair, sorted(pairs)):
            rows = info.pop("rows", None) if isinstance(info, dict) else None
            if rows:
                updates[rows[0]["id"]], updates[rows[1]["id"]] = rows
                # P rows: rules identical to A's originals get A's rewrite
                if "P" in pairs[pid]:
                    P = copy.deepcopy(pairs[pid]["P"])
                    a_old = {json.dumps(x.get("result_raw"), sort_keys=True): rows[0]["tool_rules"][k]["result_raw"]
                             for k, x in enumerate(pairs[pid]["A"]["tool_rules"])}
                    for x in P["tool_rules"]:
                        key = json.dumps(x.get("result_raw"), sort_keys=True)
                        if key in a_old:
                            x["result_raw"] = a_old[key]
                    updates[P["id"]] = P
            report[pid] = {"status": status, **(info or {})}
    counts = {}
    for v in report.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    out = {"set": ns.set, "model": ns.model, "counts": counts,
           "tokens": [getattr(drv, "total_prompt_tokens", 0), getattr(drv, "total_completion_tokens", 0)],
           "claude_cost_usd": round(getattr(drv, "total_cost_usd", 0.0), 4),
           "pairs": report}
    json.dump(out, open(os.path.join(ROOT, "dataset", f"harmonize_{ns.set}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    if ns.dry_run or not updates:
        return
    for f, rows in rows_by_file.items():
        new_rows = [updates.get(r["id"], r) for r in rows]
        with open(f, "w") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(updates)} updated rows into {len(files)} files")


if __name__ == "__main__":
    main()
