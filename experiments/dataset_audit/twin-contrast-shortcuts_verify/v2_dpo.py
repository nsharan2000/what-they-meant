"""V2 (verifier): what contrast do the DPO pairs encode? Both sides classified with the GRPO reward's own grader
(wtm.reward.decision_of / reward), using the scenario spec carried by the GRPO row with the same (scenario_id, level).
Also: where the rejected sides came from (build_report), length and strict-format cues, and what the BASE model's failures on
twin A look like (the DPO arm M3 starts from the base, not from M1), from the replication grid files.

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v2_dpo.py -> results/v2_dpo.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import CUR, ROOT, T, curated, jl, save  # noqa: E402

from wtm.protocol import parse_action  # noqa: E402
from wtm.reward import decision_of, reward  # noqa: E402

ASK = {"ask_informed", "ask_generic", "tell_informed", "statement"}


def coarse(d):
    if d.startswith("commit"):
        return "commit"
    if d.startswith("approve"):
        return "approve"
    if d in ASK:
        return "message"
    return d


def main():
    out = {}
    dpo = curated("dpo_train")
    grpo = {(r["scenario_id"], r["level"]): r for r in curated("grpo_train")}
    rows = []
    for r in dpo:
        g = grpo[(r["scenario_id"], r["level"])]
        assert g["prompt"] == r["prompt"], (r["scenario_id"], r["level"])
        scn = json.loads(g["scenario"])
        kinds = json.loads(g["tool_kinds"])
        ch, rj = r["chosen"][0]["content"], r["rejected"][0]["content"]
        dc, sc = decision_of(ch, scn, kinds)
        dr, sr = decision_of(rj, scn, kinds)
        rows.append({"i": r["_i"], "sid": r["scenario_id"], "level": r["level"], "t": T(r), "exp": scn["expected"]["decision"],
                     "chosen": dc, "rejected": dr, "chosen_c": coarse(dc), "rejected_c": coarse(dr),
                     "r_chosen": reward(ch, scn, kinds)[0], "r_rejected": reward(rj, scn, kinds)[0],
                     "len_c": len(ch.split()), "len_r": len(rj.split()), "chars_c": len(ch), "chars_r": len(rj),
                     "strict_c": sc, "strict_r": sr, "rej_text": rj[:160]})
    out["n"] = len(rows)
    for t in "AB":
        sel = [x for x in rows if x["t"] == t]
        out[f"twin_{t}"] = {"n": len(sel), "chosen_fine": dict(Counter(x["chosen"] for x in sel)),
                            "rejected_fine": dict(Counter(x["rejected"] for x in sel)),
                            "rejected_coarse": dict(Counter(x["rejected_c"] for x in sel)),
                            "expected": dict(Counter(x["exp"] for x in sel)),
                            "rejected_longer_words": round(sum(x["len_r"] > x["len_c"] for x in sel) / len(sel), 3),
                            "rejected_longer_chars": round(sum(x["chars_r"] > x["chars_c"] for x in sel) / len(sel), 3)}
    # the contrast that separates the twins: A ask > approve-card, B approve-card (or commit when expected act) > message
    twin_contrast = [x for x in rows if (x["t"] == "A" and x["chosen_c"] == "message" and x["rejected_c"] == "approve")
                     or (x["t"] == "B" and x["chosen_c"] in ("approve", "commit") and x["rejected_c"] == "message")]
    out["twin_decision_contrast_pairs"] = {"n": len(twin_contrast), "share": round(len(twin_contrast) / len(rows), 3),
                                           "A": sum(x["t"] == "A" for x in twin_contrast), "B": sum(x["t"] == "B" for x in twin_contrast)}
    commit_rej = [x for x in rows if x["rejected_c"] == "commit"]
    out["rejected_is_commit"] = {"n": len(commit_rej), "A": sum(x["t"] == "A" for x in commit_rej),
                                 "B": sum(x["t"] == "B" for x in commit_rej)}
    # B rows whose CHOSEN is itself a commit (expected act): chosen and rejected have the same coarse type
    out["B_rows_chosen_commit"] = dict(Counter(x["chosen"] + " > " + x["rejected"] for x in rows if x["t"] == "B" and x["chosen_c"] == "commit"))
    out["same_coarse_type_pairs"] = sum(x["chosen_c"] == x["rejected_c"] for x in rows)
    strict_sep = sum(x["strict_c"] != x["strict_r"] for x in rows)
    out["strict_format_differs"] = {"n": strict_sep, "share": round(strict_sep / len(rows), 3),
                                    "chosen_strict_rejected_not": sum(x["strict_c"] and not x["strict_r"] for x in rows)}
    # what a "never commit, always ask" vs "never commit, always approve" policy would look like on the preference data:
    # does any pair prefer an approve-card over a message on A, or a message over an approve-card on B?  (sanity of directions)
    out["A_rows_chosen_message_rejected_message"] = sum(x["t"] == "A" and x["chosen_c"] == "message" and x["rejected_c"] == "message" for x in rows)
    out["examples"] = {"A_rej_commit": [x for x in rows if x["t"] == "A" and x["rejected_c"] == "commit"][:3],
                       "B_rej_commit": [x for x in rows if x["t"] == "B" and x["rejected_c"] == "commit"][:3],
                       "A_rej_approve": [x for x in rows if x["t"] == "A" and x["rejected_c"] == "approve"],
                       "B_rej_message": [x for x in rows if x["t"] == "B" and x["rejected_c"] == "message"][:3]}
    br = json.load(open(os.path.join(CUR, "build_report.json")))
    out["build_report_dpo_rejected_source"] = br.get("dpo_rejected_source")

    # BASE Tiny Aya failure classes on twin A / B in the replication grid (the source of the observed rejected sides)
    base = Counter()
    base_b = Counter()
    nfiles = []
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "replication", "results", "grid_*.jsonl"))):
        n = 0
        for r in jl(f):
            if not str(r.get("model", "")).startswith(("hf:CohereLabs", "vllm:tiny")):
                continue
            n += 1
            d = r["decision_grade"]["decision"]
            if r["twin"].startswith("A_should"):
                base[d] += 1
            elif r["twin"].startswith("B"):
                base_b[d] += 1
        nfiles.append([os.path.basename(f), n])
    out["base_grid_files"] = nfiles
    out["base_tiny_aya_A_decisions_replication_grid"] = dict(base.most_common())
    out["base_tiny_aya_B_decisions_replication_grid"] = dict(base_b.most_common())
    # the base model's greedy decisions on the TRAINING pairs from the v1.2 base eval (all levels/prompts), twin A
    ev = os.path.join(ROOT, "experiments", "training_eval", "results", "eval_base_v12.jsonl")
    if os.path.exists(ev):
        spl = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))
        train = set(spl["train"])
        ca, cb = Counter(), Counter()
        for r in jl(ev):
            if r["pair_id"] in train:
                d = r["decision_grade"]["decision"]
                (ca if r["twin"].startswith("A_should") else cb if r["twin"].startswith("B") else Counter())[d] += 1
        out["base_eval_v12_train_pairs_A"] = dict(ca.most_common())
        out["base_eval_v12_train_pairs_B"] = dict(cb.most_common())
    save(out, "v2_dpo.json")
    for k, v in out.items():
        if k != "examples":
            print(k, v)
    for k, v in out["examples"].items():
        print(k)
        for x in v[:4]:
            print("   ", x["i"], x["sid"], x["level"], x["chosen"], "|", x["rejected"], "|", x["rej_text"][:120])


if __name__ == "__main__":
    main()
