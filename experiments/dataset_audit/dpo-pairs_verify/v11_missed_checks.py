"""Verifier step 11: checks for problems in the DPO-pairs scope that the analyst did not report.

(a) Approval policy visible in the prompt: share of DPO prompts whose system text (minus the shared protocol block and the
    tool docs) states that money moves need an approval card / approval, by level. At L0r the env template is one line.
(b) Could the build have used a Tiny-Aya-Global failure generated under P0 (and after the canonical look-up) instead of the
    P1 / pre-look-up one it picked? Counts, per DPO row with an off-context rejected side, of usable alternatives in the
    existing grid files (same scenario + level, reward <= 0, Tiny Aya Global, grid prompt P0; and also same messages).
(c) The anti-commit prior vs the 18 B 'act' rows (commit is the chosen side): how many pairs push the commit type down vs up,
    by row class.
(d) The two 'lookup' rejected sides (a further look-up at the decision point gets reward -0.1; free in evaluation).
(e) P2 share by twin; P2 hint vs chosen agreement.
Output: results/v11_missed_checks.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.protocol import PROTOCOL_INSTRUCTIONS  # noqa: E402
from wtm.render import golden_text  # noqa: E402
from wtm.reward import reward  # noqa: E402
from wtm.spec import load_env  # noqa: E402

OUT = os.path.join(HERE, "results")
rows = [json.loads(l) for l in open(os.path.join(OUT, "v1_rows.jsonl"))]
dpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))]
HINT = golden_text("P2")
scns = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl"):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        for l in open(f):
            s = json.loads(l)
            scns[s["id"]] = s

# (a) approval policy in the prompt
APPROVAL_RE = re.compile(r"approv", re.I)
pol = defaultdict(Counter)
for r, d in zip(rows, dpo):
    sysm = d["prompt"][0]["content"].replace(HINT, "").replace(PROTOCOL_INSTRUCTIONS, "")
    sysm = re.sub(r"## Tools\n(?:- .*\n?)*", "", sysm)   # drop the tool docs block (request_approval's own description)
    pol[r["level"]][bool(APPROVAL_RE.search(sysm))] += 1
pol = {k: dict(v) for k, v in sorted(pol.items())}

# (b) alternatives in the grid
kinds = {}
alt = Counter()
alt_rows = []
grid_rows = defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "experiments/replication/results/grid_*.jsonl"))):
    for li, line in enumerate(open(f)):
        g = json.loads(line)
        m = g["model"].split("@")[0]
        if m not in ("hf:CohereLabs/tiny-aya-global:cohere", "vllm:tiny-aya-global-local") or g["decision_grade"]["success"]:
            continue
        if not g.get("events") or not g["events"][-1].get("raw"):
            continue
        grid_rows[(g["id"], g["level"])].append((os.path.basename(f), li, g))
for r, d in zip(rows, dpo):
    if r["rej_source"] != "observed" or (r["prov"]["grid_prompt"] == "P0" and r["prov"]["model"].startswith(("hf:CohereLabs/tiny-aya-global", "vllm:tiny"))):
        continue
    s = scns[r["scenario_id"]]
    if s["env_id"] not in kinds:
        kinds[s["env_id"]] = {t["name"]: t["kind"] for t in load_env(s["env_id"])["tools"]}
    k = kinds[s["env_id"]]
    rest = [(m["role"], m["content"]) for m in d["prompt"][1:]]
    p0 = [g for _, _, g in grid_rows.get((r["scenario_id"], r["level"]), []) if g["prompt"] == "P0"
          and reward(g["events"][-1]["raw"], s, k)[0] <= 0]
    same_ctx = []
    for g in p0:
        gm = [(m["role"], m["content"]) for m in g["messages"]]
        j = max(i for i, m in enumerate(gm) if m[0] == "assistant")
        if gm[:j] == rest:
            same_ctx.append(g)
    alt["rows_considered(non-P0 or non-TinyAyaGlobal rejected)"] += 1
    alt["have_P0_TinyAyaGlobal_failure_reward<=0"] += bool(p0)
    alt["have_one_generated_in_the_same_messages"] += bool(same_ctx)
    if p0 and len(alt_rows) < 5:
        alt_rows.append([r["idx"], r["scenario_id"], r["level"], len(p0), len(same_ctx)])

# (c) commit prior vs act rows
def is_commit_text(r, side):
    return r[f"{side}_parse_kind"] == "tool_call" and r[f"{side}_tool_kind"] == "commit"


cls = lambda r: "A" if r["twin"] == "A" else ("B_act" if r["expected"] == "act" else "B_approve")
commit_push = Counter()
for r in rows:
    if is_commit_text(r, "rej"):
        commit_push[(cls(r), "commit_rejected")] += 1
    if is_commit_text(r, "chosen"):
        commit_push[(cls(r), "commit_chosen")] += 1

# (d) lookup rejected
lk = [[r["idx"], r["scenario_id"], r["level"], r["rej_text"][:160]] for r in rows if r["rej_dec"] == "lookup"]

# (e) P2 shares and hint agreement
p2 = Counter((r["twin"], r["prompt_setting"]) for r in rows)
p2_rows = [r for r in rows if r["prompt_setting"] == "P2"]
hint_conflict = [[r["idx"], r["scenario_id"], r["level"]] for r in p2_rows if r["expected"] == "act"]

res = {
    "approval_policy_stated_in_system_prompt_by_level(True/False)": pol,
    "grid_alternatives": dict(alt), "grid_alternative_examples[idx,sid,level,n_P0,n_same_msgs]": alt_rows,
    "commit_type_push": {f"{a}:{b}": n for (a, b), n in sorted(commit_push.items())},
    "lookup_rejected": lk,
    "P2_share": {t: round(p2[(t, "P2")] / (p2[(t, "P2")] + p2[(t, "P0")]), 4) for t in ("A", "B")},
    "P2_rows": len(p2_rows), "P2_rows_hint_conflicts_with_act_target": hint_conflict,
}
json.dump(res, open(os.path.join(OUT, "v11_missed_checks.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
