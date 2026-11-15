"""Missed-issue check: where does the decisive evidence sit in twin-A decision prompts, relative to the generation point?

Tiny Aya (Cohere2): 3 of every 4 layers use a 4,096-token sliding window. If the only copy of the decisive value (an
informed_any keyword = what the grader requires the question to name) is in the member block more than 4,096 tokens before
the end of the prompt, only the global layers can see it. Segments searched: member block (from '## Member' / the L0 member
card up to '## Tools') and the conversation after the system prompt (user request, look-up call, tool result). Policy text,
tool docs, protocol and the hint are excluded so generic words there do not count as evidence.
Token distance = characters after the last evidence hit x (row prompt tokens / row prompt characters) -- an approximation.
Outputs: results/v4_evidence.json, results/v4_evidence_rows.csv
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import pandas as pd

from vcommon import CUR, OUT, ROOT, dump, load_jsonl

TOK = os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")


def hits(text: str, kw: str):
    t, k = text.lower(), kw.lower().strip()
    if not k:
        return []
    if re.fullmatch(r"[\d.,]+[km]?", k):
        k2 = k.replace(",", "")
        pat = r"(?<![\d.])" + r",?".join(re.escape(c) for c in k2) + r"(?![\d])"
        return [m.start() for m in re.finditer(pat, t)]
    return [m.start() for m in re.finditer(re.escape(k), t)]


def main():
    tok = {(t["file"], t["i"]): t["prompt_tokens"] for t in map(json.loads, open(TOK))}
    specs = {}
    for r in load_jsonl(os.path.join(CUR, "grpo_train.jsonl")):
        specs[r["scenario_id"]] = json.loads(r["scenario"])
    rows = []
    for fn in ("sft_train", "sft_val"):
        for i, r in enumerate(load_jsonl(os.path.join(CUR, fn + ".jsonl"))):
            if r["turn"] != "decision" or not r["twin"].startswith("A"):
                continue
            scn = specs.get(r["scenario_id"])
            if scn is None:
                continue
            sys_txt = r["prompt"][0]["content"]
            conv = "\n".join(m["content"] for m in r["prompt"][1:])
            full = sys_txt + "\n" + conv
            ms = sys_txt.find("## Member") if "## Member" in sys_txt else sys_txt.find("Member:")
            me = sys_txt.find("## Tools")
            seg_member = (ms, me) if 0 <= ms < me else (None, None)
            kws = scn["expected"].get("informed_any", [])
            mem_hits, conv_hits = [], []
            for k in kws:
                for p in hits(full, k):
                    if seg_member[0] is not None and seg_member[0] <= p < seg_member[1]:
                        mem_hits.append(p)
                    elif p > len(sys_txt):
                        conv_hits.append(p)
            n_chars = len(full)
            tpc = tok[(fn + ".jsonl", i)] / n_chars
            last = max(mem_hits + conv_hits) if (mem_hits or conv_hits) else None
            rows.append({"file": fn, "line": i, "sid": r["scenario_id"], "level": r["level"], "ps": r["prompt_setting"],
                         "prompt_tokens": tok[(fn + ".jsonl", i)], "in_member_block": bool(mem_hits), "in_conversation": bool(conv_hits),
                         "dist_tokens": round((n_chars - last) * tpc) if last is not None else None,
                         "dist_tokens_member_last": round((n_chars - max(mem_hits)) * tpc) if mem_hits else None,
                         "dist_tokens_member_first": round((n_chars - min(mem_hits)) * tpc) if mem_hits else None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v4_evidence_rows.csv"), index=False)
    out = {}
    for (fn, lv), g in list(df.groupby(["file", "level"])) + [(("all", "all"), df)]:
        only_mem = g[g.in_member_block & ~g.in_conversation]
        out[f"{fn}:{lv}"] = {"n_A_decision_rows": int(len(g)), "no_evidence_in_member_or_conversation": int(g.dist_tokens.isna().sum()),
                             "evidence_in_conversation": int(g.in_conversation.sum()),
                             "evidence_only_in_member_block": int(len(only_mem)),
                             "only_member_and_beyond_4096": int((only_mem.dist_tokens > 4096).sum()),
                             "median_dist_only_member": float(only_mem.dist_tokens.median()) if len(only_mem) else None,
                             "max_dist_only_member": float(only_mem.dist_tokens.max()) if len(only_mem) else None}
    out["examples_only_member"] = df[df.in_member_block & ~df.in_conversation].sort_values("dist_tokens", ascending=False)\
        .head(8).to_dict(orient="records")
    out["examples_no_evidence"] = df[df.dist_tokens.isna()].head(6).to_dict(orient="records")
    dump(out, "v4_evidence.json")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
