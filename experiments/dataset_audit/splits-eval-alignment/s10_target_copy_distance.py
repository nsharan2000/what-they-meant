"""5 (continued). How far back must a twin-A target reach? Identifiers the target cites, located in its own prompt.

For every twin-A decision row with a message target (sft_train + sft_val), every identifier the target cites (same patterns as
training/build_datasets.py IDENTIFIERS: email/UPI id, @handle, phone, ISO date, masked number, order/invoice id; plus money amounts)
is located in the prompt: in the request or a look-up result (always near the end), in the system prompt within the last 4,096
tokens, or ONLY in the system prompt beyond 4,096 tokens (visible to 1 of every 4 layers at generation). Offsets use tokmodel.py,
rescaled to the exact prompt length from token_stats.jsonl.

python3 -B experiments/dataset_audit/splits-eval-alignment/s10_target_copy_distance.py -> results/s10_target_copy_distance.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

from common import ROOT, dump, jl, views
from tokmodel import FEATS, est_text, weights

sys.path.insert(0, os.path.join(ROOT, "training"))
from build_datasets import IDENTIFIERS, norm  # noqa: E402  (pure stdlib + wtm; nothing runs at import)

W = 4096
V = views()
w = weights()
PER_MSG, CONST = w[len(FEATS)], w[len(FEATS) + 1]
stats = {(s["file"], s["i"]): s["prompt_tokens"] for s in jl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))}
MONEY = re.compile(r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|INR))")
agg = defaultdict(Counter)
ex = defaultdict(list)
for v in ("sft_train", "sft_val"):
    for i, r in enumerate(V[v]):
        if r["turn"] != "decision" or not r["twin"].startswith("A"):
            continue
        t = r["completion"][0]["content"]
        if t.startswith("<tool_call>"):
            continue
        prompt = r["prompt"]
        sysm = prompt[0]["content"]
        msg_tok = [est_text(m["content"]) + PER_MSG for m in prompt]
        total_est = CONST + sum(msg_tok)
        scale = stats[(f"{v}.jsonl", i)] / total_est
        rest = norm("\n".join(m["content"] for m in prompt[1:])).lower()
        found = {(k, (m.group(1) if rx.groups else m.group(0)).rstrip(".")) for k, rx in IDENTIFIERS for m in rx.finditer(t)}
        found |= {("money", m.group(0).strip()) for m in MONEY.finditer(t)}
        for kind, val in sorted(found):
            key = f"{v}|{r['level']}"
            agg[key]["identifiers"] += 1
            vl = norm(val).lower()
            if vl in rest:
                agg[key]["in_request_or_lookup_result"] += 1
                continue
            occ = [m.start() for m in re.finditer(re.escape(vl), norm(sysm).lower())]
            if not occ:
                agg[key]["not_in_prompt_verbatim"] += 1
                agg[key][f"not_in_prompt_verbatim|{kind}"] += 1
                continue
            offs = [(total_est - (CONST + est_text(norm(sysm)[:o]))) * scale for o in occ]
            near = min(offs)
            if near > W:
                agg[key]["only_in_system_beyond_4096"] += 1
                if len(ex[key]) < 4:
                    ex[key].append({"i": i, "scenario_id": r["scenario_id"], "identifier": val, "kind": kind,
                                    "nearest_offset_from_end": round(near), "target": t[:150]})
            else:
                agg[key]["only_in_system_within_4096"] += 1
        agg[f"{v}|{r['level']}"]["rows"] += 1
out = {"by_view_level": {k: dict(v) for k, v in sorted(agg.items())}, "examples_beyond_window": ex,
       "note": "money amounts are matched verbatim (e.g. '$4,820'); an identifier counted 'not_in_prompt_verbatim' may still be "
               "present in another format"}
print(json.dumps(out["by_view_level"], indent=1))
print(json.dumps(ex, indent=0, ensure_ascii=False)[:2500])
print(dump("s10_target_copy_distance.json", out))
