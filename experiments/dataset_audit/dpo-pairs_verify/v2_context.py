"""Verifier step 2: was each observed rejected text generated in the same context as the DPO prompt it is paired with?

For every observed rejected side (results/v1_rows.jsonl provenance: grid file + line), compare the grid episode with the DPO prompt:
  sys_same      the grid episode's system prompt length equals the DPO system prompt length with the P2 hint removed (P0) or
                the DPO system length itself (when both are P2; never happens: the grid has no P2)
  msgs_same     the grid conversation before the rejected turn == the DPO prompt after the system message (user request,
                look-up calls and tool results), compared as exact strings
  canon_n       number of canonical look-ups in the DPO prompt
Also: identical rejected texts shared by A/B twins at the same level, and the twin-level structure.
Output: results/v2_context.json, results/v2_context_rows.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.render import golden_text  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = [json.loads(l) for l in open(os.path.join(OUT, "v1_rows.jsonl"))]
dpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset/curated_v121/dpo_train.jsonl"))]
HINT = golden_text("P2")
GRID = os.path.join(ROOT, "experiments/replication/results")

need = defaultdict(set)
for r in rows:
    if r["prov"]:
        need[r["prov"]["file"]].add(r["prov"]["line"])
grid = {}
for f, lines in need.items():
    for li, line in enumerate(open(os.path.join(GRID, f))):
        if li in lines:
            grid[(f, li)] = json.loads(line)

out_rows = []
for r, d in zip(rows, dpo):
    sysmsg = d["prompt"][0]["content"]
    sys_p0 = sysmsg.replace("\n\n" + HINT, "") if d["prompt_setting"] == "P2" else sysmsg
    rest = [(m["role"], m["content"]) for m in d["prompt"][1:]]
    canon_n = sum(1 for m in d["prompt"][1:] if m["role"] == "assistant")
    rec = {"idx": r["idx"], "scenario_id": r["scenario_id"], "twin": r["twin"], "level": r["level"],
           "prompt_setting": r["prompt_setting"], "canon_n": canon_n, "source": r["rej_source"]}
    if r["prov"]:
        g = grid[(r["prov"]["file"], r["prov"]["line"])]
        gm = [(m["role"], m["content"]) for m in g["messages"]]
        # the rejected text is the last assistant message of the grid episode (a tool result / "Okay." may follow it)
        j = max(i for i, m in enumerate(gm) if m[0] == "assistant")
        assert gm[j][1] == r["rej_text"], r["idx"]
        prefix = gm[:j]
        rec["grid_msgs_after_rejected"] = len(gm) - 1 - j
        rec.update({"grid_prompt": g["prompt"], "grid_model": g["model"].split("@")[0], "grid_sys_chars": g.get("system_chars"),
                    "dpo_sys_chars_p0": len(sys_p0), "sys_same_len": g.get("system_chars") == len(sys_p0),
                    "msgs_same": prefix == rest, "user_same": prefix[:1] == rest[:1], "grid_prior_turns": len(prefix) - 1,
                    "grid_tool_results": g.get("tool_results")})
        rec["fully_on_context"] = bool(rec["sys_same_len"] and rec["msgs_same"] and g["prompt"] == "P0"
                                       and d["prompt_setting"] == "P0")
    out_rows.append(rec)

obs = [x for x in out_rows if x["source"] == "observed"]
C = Counter
summary = {
    "n_observed": len(obs),
    "grid_prompt": dict(C(x["grid_prompt"] for x in obs)),
    "grid_prompt_x_dpo_prompt": dict(C(f"{x['grid_prompt']}->{x['prompt_setting']}" for x in obs)),
    "sys_same_len_given_grid_P0": dict(C(x["sys_same_len"] for x in obs if x["grid_prompt"] == "P0")),
    "msgs_same": dict(C(x["msgs_same"] for x in obs)),
    "msgs_same_by_canon_n": {str(k): dict(C(x["msgs_same"] for x in obs if x["canon_n"] == k)) for k in sorted({x["canon_n"] for x in obs})},
    "no_prior_turns_but_dpo_has_lookup": sum(1 for x in obs if x["grid_prior_turns"] == 0 and x["canon_n"] > 0),
    "no_prior_turns_and_dpo_has_no_lookup": sum(1 for x in obs if x["grid_prior_turns"] == 0 and x["canon_n"] == 0),
    "user_same": dict(C(x["user_same"] for x in obs)),
    "fully_on_context(P0 grid, P0 dpo, same sys length, same messages)": sum(x["fully_on_context"] for x in obs),
    "on_context_ignoring_prompt_setting(same messages, sys same len or grid P1)": sum(1 for x in obs if x["msgs_same"]),
    "fully_on_context_idx": [x["idx"] for x in obs if x["fully_on_context"]],
}

# identical rejected text shared by the two twins of the same pair at the same level
by_tl = defaultdict(dict)
for r in rows:
    by_tl[(r["pair_id"], r["level"])][r["twin"]] = r
both = {k: v for k, v in by_tl.items() if "A" in v and "B" in v}
same = {k: v for k, v in both.items() if v["A"]["rej_text"] == v["B"]["rej_text"]}
same_ws = {k: v for k, v in both.items() if " ".join(v["A"]["rej_text"].split()) == " ".join(v["B"]["rej_text"].split())}
rej_counts = C(r["rej_text"] for r in rows)
summary.update({
    "twin_levels_total": len(by_tl), "twin_levels_with_both": len(both),
    "unpaired": sorted([list(k) + list(v) for k, v in by_tl.items() if len(v) == 1]),
    "twin_levels_identical_rejected": len(same), "pairs_in_identical_twin_levels": 2 * len(same),
    "twin_levels_identical_rejected_ws_normalised": len(same_ws),
    "identical_twin_levels_both_first_action": sum(1 for v in same.values()
                                                   if v["A"]["prov"] and v["A"]["prov"]["n_events"] == 1
                                                   and v["B"]["prov"] and v["B"]["prov"]["n_events"] == 1),
    "identical_twin_levels_rej_dec_A_B": dict(C(f"{v['A']['rej_dec']}|{v['B']['rej_dec']}" for v in same.values())),
    "identical_twin_levels_synthetic": sum(1 for v in same.values() if not v["A"]["prov"]),
    "distinct_rejected_texts_used_more_than_once": sum(1 for t, n in rej_counts.items() if n > 1),
    "pairs_using_a_reused_rejected_text": sum(n for t, n in rej_counts.items() if n > 1),
    "examples_identical": [[v["A"]["idx"], v["B"]["idx"], k[0], k[1]] for k, v in list(same.items())[:6]],
})
json.dump(summary, open(os.path.join(OUT, "v2_context.json"), "w"), indent=1)
with open(os.path.join(OUT, "v2_context_rows.csv"), "w", newline="") as fh:
    keys = sorted({k for x in out_rows for k in x})
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for x in out_rows:
        w.writerow(x)
print(json.dumps(summary, indent=1))
