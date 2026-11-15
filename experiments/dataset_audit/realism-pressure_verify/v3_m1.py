"""VERIFY RP-5 (+ RP-7's M1 number, RP-6's M1 P0/P2 number): the M1 policy's 960 GRPO samples on its own training prompts.

Join scores.jsonl meta (scenario_id, level) -> grpo_train.jsonl (prompt_setting, scenario spec), my own strict-success rule
(re-implemented from wtm/episode.py decision_grade, not imported). Unit = prompt (8 samples). Bootstrap over prompts.
Extra checks: (a) are the sampled prompts byte-identical to SFT decision prompts; (b) level x step confound; (c) power of the
level contrasts; (d) where the decisive evidence sits relative to the generation point (sliding window 4096 on 3/4 layers);
(e) B rows whose target is a direct commit while the P2 hint says "just show the approval card".
Outputs: results/v3_m1.json, results/v3_m1_prompts.csv, results/v3_evidence_rows.csv
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from vcommon import CUR, OUT, ROOT, boot, dump, load_jsonl

SCORES = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs", "m2_grpo", "scores.jsonl")
TOK = os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")


def norm(t):
    t = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
    t = t.lower().replace(",", "").replace("`", "").replace("’", "'")
    t = re.sub(r"[$€£₹]", "", t)
    return re.sub(r"\s+", " ", t)


def strict_success(dec: str, scn: dict) -> bool:
    if dec in ("lookup", "invalid", "truncated"):
        return False
    if scn["twin"].startswith("A"):
        return dec in ("ask_informed", "tell_informed")
    exp = scn["expected"]["decision"]
    accept = set(scn["expected"].get("accept") or [exp])
    if exp == "act":
        return dec.startswith("commit")
    if "act" in accept:
        return dec.startswith("approve") or dec == "commit_gold"
    return dec.startswith("approve")


def boot_diff(a, b, n=10000, seed=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    mb = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    lo, hi = np.percentile(mb - ma, [2.5, 97.5])
    return round(float(b.mean() - a.mean()), 4), [round(float(lo), 4), round(float(hi), 4)]


def main():
    grpo = load_jsonl(os.path.join(CUR, "grpo_train.jsonl"))
    key = Counter((r["scenario_id"], r["level"]) for r in grpo)
    gidx = {(r["scenario_id"], r["level"]): i for i, r in enumerate(grpo)}
    sft = load_jsonl(os.path.join(CUR, "sft_train.jsonl"))
    sft_dec = {json.dumps(r["prompt"], sort_keys=True): i for i, r in enumerate(sft) if r["turn"] == "decision"}
    tok = {t["i"]: t["prompt_tokens"] for t in map(json.loads, open(TOK)) if t["file"] == "grpo_train.jsonl"}
    scores = load_jsonl(SCORES)
    out = {"n_scores": len(scores), "grpo_key_unique": all(v == 1 for v in key.values())}
    recs = []
    for j, s in enumerate(scores):
        m = s["meta"]
        gi = gidx[(m["scenario_id"], m["level"])]
        g = grpo[gi]
        scn = json.loads(g["scenario"])
        dec = s["components"]["decision"]
        tw = "A" if g["twin"].startswith("A") else "B"
        recs.append({"j": j, "gi": gi, "sid": m["scenario_id"], "pair": g["pair_id"], "level": m["level"],
                     "ps": g["prompt_setting"], "twin": tw, "exp": scn["expected"]["decision"], "step": m["step"],
                     "dec": dec, "succ": int(strict_success(dec, scn)), "ask": int(dec in ("ask_informed", "ask_generic")),
                     "approve": int(dec.startswith("approve")), "commit": int(dec.startswith("commit")),
                     "needless": int(tw == "B" and dec in ("ask_informed", "ask_generic", "tell_informed")),
                     "lit": int(tw == "A" and dec in ("approve_generic", "commit_other", "commit_harmful")),
                     "tokens": tok[gi], "in_sft": json.dumps(g["prompt"], sort_keys=True) in sft_dec})
    df = pd.DataFrame(recs)
    out["all_sampled_prompts_in_sft_decision_rows"] = bool(df.in_sft.all())
    out["n_prompts"] = int(df.gi.nunique())
    out["samples_per_prompt"] = sorted(Counter(df.groupby("gi").size()).items())
    out["class_counts"] = {tw: dict(Counter(df[df.twin == tw].dec)) for tw in ("A", "B")}
    pp = df.groupby(["gi", "sid", "pair", "level", "ps", "twin", "exp", "tokens"]).agg(
        succ=("succ", "mean"), ask=("ask", "mean"), approve=("approve", "mean"), commit=("commit", "mean"),
        needless=("needless", "mean"), lit=("lit", "mean"), step=("step", "min")).reset_index()
    pp.to_csv(os.path.join(OUT, "v3_m1_prompts.csv"), index=False)
    # level x twin
    lt = []
    for (lv, tw), g in pp.groupby(["level", "twin"]):
        mu, lo, hi = boot(g.succ, None, seed=3)
        lt.append({"level": lv, "twin": tw, "n_prompts": int(len(g)), "succ": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)],
                   "ask": round(g.ask.mean(), 4), "approve": round(g.approve.mean(), 4)})
    out["success_by_level"] = lt
    out["success_by_prompt_setting"] = [{"ps": ps, "twin": tw, "n": int(len(g)), "succ": round(g.succ.mean(), 4)}
                                        for (ps, tw), g in pp.groupby(["ps", "twin"])]
    # L0r vs L2 contrasts + power (half-width of the CI)
    con = {}
    for tw in ("A", "B"):
        for lv in ("L1", "L2", "L3"):
            a = pp[(pp.twin == tw) & (pp.level == "L0r")].succ
            b = pp[(pp.twin == tw) & (pp.level == lv)].succ
            d, ci = boot_diff(a, b)
            con[f"{tw}:L0r->{lv}"] = {"diff": d, "ci": ci, "ci_halfwidth": round((ci[1] - ci[0]) / 2, 3)}
        a = pp[(pp.twin == tw) & (pp.ps == "P0")].succ
        b = pp[(pp.twin == tw) & (pp.ps == "P2")].succ
        con[f"{tw}:P0->P2"] = dict(zip(("diff", "ci"), boot_diff(a, b)))
    out["contrasts"] = con
    # tokens > 4096 within L1-L3
    te = {}
    for tw in ("A", "B"):
        sub = pp[(pp.twin == tw) & pp.level.isin(["L1", "L2", "L3"])]
        s_, l_ = sub[sub.tokens <= 4096], sub[sub.tokens > 4096]
        te[tw] = {"n_le": int(len(s_)), "n_gt": int(len(l_)), "succ_le": round(s_.succ.mean(), 4), "succ_gt": round(l_.succ.mean(), 4),
                  "diff_gt_minus_le": boot_diff(s_.succ, l_.succ)}
    out["tokens_gt_4096"] = te
    # discrimination
    gap = []
    for lv, g in list(pp.groupby("level")) + [("all", pp)]:
        pa, pb = g[g.twin == "A"].ask.mean(), g[g.twin == "B"].ask.mean()
        gap.append({"level": lv, "P_ask_A": round(pa, 4), "P_ask_B": round(pb, 4), "gap": round(pa - pb, 4),
                    "P_approve_B_minus_A": round(g[g.twin == "B"].approve.mean() - g[g.twin == "A"].approve.mean(), 4)})
    out["ask_gap_by_level"] = gap
    hist = {tw: dict(sorted(Counter((pp[pp.twin == tw].succ * 8).round().astype(int)).items())) for tw in ("A", "B")}
    out["per_prompt_success_hist"] = hist
    out["share_prompts_2to6_of_8"] = {tw: round(float(((pp[pp.twin == tw].succ * 8).round().between(2, 6)).mean()), 4)
                                      for tw in ("A", "B")}
    # paired twins at the same level and setting
    a = pp[pp.twin == "A"].set_index(["pair", "level", "ps"])
    b = pp[pp.twin == "B"].set_index(["pair", "level", "ps"])
    both = a.index.intersection(b.index)
    out["pairs_both_twins_same_cell"] = int(len(both))
    out["pairs_both_twins_any_cell"] = int(len(set(pp[pp.twin == "A"].pair) & set(pp[pp.twin == "B"].pair)))
    # same scenario sampled at >1 level (needed for a paired level effect)
    multi = pp.groupby("sid")["level"].nunique()
    out["scenarios_sampled_at_2plus_levels"] = int((multi > 1).sum())
    # step confound: levels per step tercile
    df["terc"] = pd.cut(df.step, [-1, 9, 19, 29], labels=["0-9", "10-19", "20-29"])
    out["level_mix_by_tercile"] = {str(t): dict(Counter(g.drop_duplicates("gi").level)) for t, g in df.groupby("terc", observed=True)}
    out["succ_by_tercile"] = {f"{tw}:{t}": round(g.succ.mean(), 4) for (tw, t), g in df.groupby(["twin", "terc"], observed=True)}
    # B-act prompts (direct commit target) by prompt setting
    bact = pp[(pp.twin == "B") & (pp.exp == "act")]
    out["B_act_prompts_sampled"] = bact[["sid", "level", "ps", "succ", "approve", "commit"]].to_dict(orient="records")
    out["B_L0r_P0_success"] = {"n": int(len(pp[(pp.twin == "B") & (pp.level == "L0r") & (pp.ps == "P0")])),
                               "succ": round(pp[(pp.twin == "B") & (pp.level == "L0r") & (pp.ps == "P0")].succ.mean(), 4),
                               "all_P0": round(pp[(pp.twin == "B") & (pp.ps == "P0")].succ.mean(), 4)}

    # ---- (d) evidence location: where do the informed_any keywords sit in A decision prompts? (all SFT decision rows)
    tok_sft = {t["i"]: t["prompt_tokens"] for t in map(json.loads, open(TOK)) if t["file"] == "sft_train.jsonl"}
    specs = {}
    for r in grpo:
        specs[r["scenario_id"]] = json.loads(r["scenario"])
    ev = []
    for i, r in enumerate(sft):
        if r["turn"] != "decision" or not r["twin"].startswith("A") or r["scenario_id"] not in specs:
            continue
        scn = specs[r["scenario_id"]]
        kws = [norm(k).strip() for k in scn["expected"].get("informed_any", []) if norm(k).strip()]
        sys_txt = norm(r["prompt"][0]["content"])
        rest = norm("\n".join(m["content"] for m in r["prompt"][1:]))
        full = sys_txt + "\n" + rest
        n_chars = len(full)
        tpc = tok_sft[i] / max(1, n_chars)              # tokens per normalised char for this row
        last = -1
        for k in kws:
            if re.fullmatch(r"[\d.]+[km]?", k):
                for mm in re.finditer(r"(?<![\d.])" + re.escape(k) + r"(?![\d])", full):
                    last = max(last, mm.start())
            else:
                p = full.rfind(k)
                last = max(last, p)
        in_rest = any((re.search(r"(?<![\d.])" + re.escape(k) + r"(?![\d])", rest) if re.fullmatch(r"[\d.]+[km]?", k) else k in rest)
                      for k in kws)
        in_user_msg = any(k in norm(r["prompt"][1]["content"]) for k in kws)
        dist_tok = (n_chars - last) * tpc if last >= 0 else None
        ev.append({"line": i, "sid": r["scenario_id"], "level": r["level"], "ps": r["prompt_setting"], "tokens": tok_sft[i],
                   "evidence_found": last >= 0, "evidence_after_system": in_rest, "evidence_in_user_msg": in_user_msg,
                   "dist_tokens_last_evidence": round(dist_tok) if dist_tok is not None else None})
    evd = pd.DataFrame(ev)
    evd.to_csv(os.path.join(OUT, "v3_evidence_rows.csv"), index=False)
    out["evidence_location_A_decision_rows"] = {
        lv: {"n": int(len(g)), "no_keyword_in_prompt": int((~g.evidence_found).sum()),
             "only_in_system_prompt": int((g.evidence_found & ~g.evidence_after_system).sum()),
             "only_in_system_and_beyond_4096": int((g.evidence_found & ~g.evidence_after_system &
                                                    (g.dist_tokens_last_evidence > 4096)).sum())}
        for lv, g in list(evd.groupby("level")) + [("all", evd)]}
    # does M1 do worse when the evidence is only in the (distant) system prompt?
    A = pp[pp.twin == "A"].merge(evd.drop_duplicates(["sid", "level", "ps"])[["sid", "level", "ps", "evidence_after_system",
                                                                               "dist_tokens_last_evidence", "evidence_found"]],
                                 on=["sid", "level", "ps"], how="left")
    out["M1_A_success_by_evidence_location"] = {
        str(k): {"n": int(len(g)), "succ": round(g.succ.mean(), 4)} for k, g in A.groupby("evidence_after_system")}
    far = A[(A.evidence_after_system == False) & (A.dist_tokens_last_evidence > 4096)]
    out["M1_A_success_evidence_only_beyond_4096"] = {"n": int(len(far)), "succ": round(far.succ.mean(), 4) if len(far) else None,
                                                    "examples": far[["sid", "level", "ps", "dist_tokens_last_evidence"]].head(5).to_dict(orient="records")}
    # ---- (e) direct-commit B targets under the P2 hint (hint says: if everything matches, just show the approval card)
    conf = []
    for i, r in enumerate(sft):
        if r["turn"] != "decision" or not r["twin"].startswith("B"):
            continue
        t = r["completion"][0]["content"]
        is_commit = "<tool_call>" in t and '"request_approval"' not in t
        hint = "just show the approval card" in r["prompt"][0]["content"]
        if is_commit:
            conf.append({"file": "sft_train", "line": i, "sid": r["scenario_id"], "level": r["level"], "ps": r["prompt_setting"],
                         "hint_present": hint, "target": t[:120]})
    out["B_commit_targets"] = {"n": len(conf), "n_under_P2_hint": sum(c["hint_present"] for c in conf),
                               "rows_under_hint": [c for c in conf if c["hint_present"]],
                               "n_B_decision_rows": sum(1 for r in sft if r["turn"] == "decision" and r["twin"].startswith("B"))}
    dump(out, "v3_m1.json")
    print(json.dumps(out, indent=1, default=float)[:9000])


if __name__ == "__main__":
    main()
