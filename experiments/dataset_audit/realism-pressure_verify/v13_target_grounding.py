"""Missed-issue check: the decision target text is identical at every level (v12), but the context differs by level (L0r: the
seed's short member card; L1: last 15 transactions; L2: as much history as fits). Does the target name values that the prompt
at that level does not contain (an ungrounded target -> the model is trained to state facts it cannot see)?

Per decision row: value-like tokens in the target (emails, @handles, phone numbers, IDs with digits, amounts with 2 decimals or
>= 3 digits, dates) and the informed_any keywords the target contains; a token is grounded when it occurs in the prompt
(system prompt + conversation, normalised). Reported by level, for twin A and twin B.
Outputs: results/v13_target_grounding.json, results/v13_rows.csv
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from vcommon import CUR, OUT, dump, load_jsonl

VAL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|@[\w-]{3,}|\b\d{4}-\d{2}-\d{2}\b|\b[A-Z]{2,}[-_]?\d{2,}[\w-]*\b|"
                 r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\b\d+\.\d{2}\b|\b\d{3,}\b")


def norm(t: str) -> str:
    t = t.lower().replace("’", "'")
    return re.sub(r"\s+", " ", t)


def num_variants(v: str):
    """'1,250.00' -> {'1,250.00', '1250.00', '1250', '1,250'}"""
    out = {v}
    if re.fullmatch(r"[\d,]+(\.\d{2})?", v):
        plain = v.replace(",", "")
        out |= {plain, plain.split(".")[0], v.split(".")[0]}
        if plain.endswith(".00"):
            out.add(plain[:-3])
        try:
            f = float(plain)
            out |= {f"{f:,.2f}", f"{f:.2f}", f"{f:,.0f}" if f == int(f) else f"{f:,.2f}"}
        except ValueError:
            pass
    return out


def main():
    specs = {json.loads(l)["scenario_id"]: json.loads(json.loads(l)["scenario"]) for l in open(os.path.join(CUR, "grpo_train.jsonl"))}
    rows = []
    for i, r in enumerate(load_jsonl(os.path.join(CUR, "sft_train.jsonl"))):
        if r["turn"] != "decision":
            continue
        tgt = r["completion"][0]["content"]
        prompt = norm("\n".join(m["content"] for m in r["prompt"]))
        vals = sorted({v.rstrip(".,;:") for v in VAL.findall(tgt)})
        ungrounded = [v for v in vals if not any(norm(x) in prompt for x in num_variants(v))]
        scn = specs.get(r["scenario_id"], {})
        kws = [k for k in scn.get("expected", {}).get("informed_any", []) if k.strip() and norm(k) in norm(tgt)]
        kw_ungr = [k for k in kws if norm(k) not in prompt]
        rows.append({"line": i, "sid": r["scenario_id"], "twin": r["twin"][0], "level": r["level"], "ps": r["prompt_setting"],
                     "n_values": len(vals), "n_ungrounded": len(ungrounded), "ungrounded": "|".join(ungrounded[:4]),
                     "n_kw_in_target": len(kws), "n_kw_ungrounded": len(kw_ungr), "kw_ungrounded": "|".join(kw_ungr[:3])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v13_rows.csv"), index=False)
    out = {}
    for (tw, lv), g in df.groupby(["twin", "level"]):
        out[f"{tw}:{lv}"] = {"rows": int(len(g)), "rows_with_value_tokens": int((g.n_values > 0).sum()),
                             "rows_with_ungrounded_value": int((g.n_ungrounded > 0).sum())}
    # NOTE: informed_any keywords are mostly descriptors ('different email', 'already paid'), not values, so their absence
    # from the prompt is expected; n_kw_ungrounded is kept in v13_rows.csv for reference only.
    # scenarios whose target is grounded at some levels and not at others
    a = df[df.twin == "A"].groupby(["sid"]).apply(lambda g: pd.Series({
        "levels_ungrounded": ",".join(sorted(g[g.n_ungrounded > 0].level.unique())),
        "levels_grounded": ",".join(sorted(g[g.n_ungrounded == 0].level.unique()))}), include_groups=False).reset_index()
    mixed = a[(a.levels_ungrounded != "") & (a.levels_grounded != "")]
    out["A_scenarios_grounded_only_at_some_levels"] = int(len(mixed))
    out["A_scenarios_examples"] = mixed.head(8).to_dict(orient="records")
    out["examples_ungrounded_rows"] = df[(df.n_ungrounded > 0) & (df.twin == "A")].head(10)[
        ["line", "sid", "level", "ps", "ungrounded"]].to_dict(orient="records")
    dump(out, "v13_target_grounding.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
