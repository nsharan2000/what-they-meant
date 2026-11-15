"""SEA-6 sub-claims: (i) share of decisive scenario-record lines at L2 that sit > 4,096 tokens from the generation point;
(ii) identifiers cited by L2 twin-A message targets whose NEAREST occurrence in the prompt is > 4,096 tokens from the end;
(iii) M1 sampled accuracy on L2 prompts split by whether the A-target's comparator is beyond 4,096 (small n).
Token distances use the character-class model fitted in v06b.
Output: results/v06c_record_positions.json
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from vcommon import ROOT, curated, dump, specs, target_text
from wtm.render import _txn_line

w = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "results", "v06b_tokfit_positions.json")))["weights"])


def est(t):
    f = np.array([sum(c.isalpha() and c.isascii() for c in t), sum(c.isdigit() for c in t), sum(c.isspace() for c in t),
                  sum((not c.isalnum()) and (not c.isspace()) and c.isascii() for c in t), sum(not c.isascii() for c in t),
                  len(re.findall(r"[A-Za-z]+", t)), len(re.findall(r"\d+", t)), 0.0], dtype=float)
    return float(f @ w)


def specials_after_system(prompt):
    return 1 + sum(3 + (2 if m["role"] == "assistant" else 0) for m in prompt[1:]) + 3


by_id = {s["id"]: s for s in specs()}
sft = curated("sft_train")
IDR = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w.@])@[\w.-]*\w|\b[A-Z]{2,}-?\d[\w-]*\b|\*{2,}\d{3,}|\b\d{4}-\d{2}-\d{2}\b")
rec_c = Counter()
rec_ex = []
id_c = Counter()
id_far = []
row_far = {}
for i, r in enumerate(sft):
    if r["turn"] != "decision" or r["level"] != "L2":
        continue
    s = by_id[r["scenario_id"]]
    p = r["prompt"]
    sysm = p[0]["content"]
    tail = "".join(m["content"] for m in p[1:])
    tail_tok = est(tail) + specials_after_system(p)
    for x in s.get("scenario_records") or []:
        rec = x["record"]
        if x["field"] == "transactions" and isinstance(rec, dict):
            line = _txn_line(rec)
        elif isinstance(rec, str):
            line = rec
        else:
            line = json.dumps(rec, ensure_ascii=False)
        k = sysm.find(line)
        rec_c["records"] += 1
        if k < 0:
            rec_c["not_found_verbatim(capped_or_reformatted)"] += 1
            continue
        d = est(sysm[k:]) + tail_tok
        rec_c["found"] += 1
        rec_c["beyond_4096"] += d > 4096
        if d > 4096 and len(rec_ex) < 4:
            rec_ex.append([i, r["scenario_id"], round(d), line[:90]])
    if s["twin"] == "A_should_ask" and s["ideal_action"]["type"] == "message":
        t = target_text(r)
        far_any = False
        full = sysm + "\x00" + "\x00".join(m["content"] for m in p[1:])
        for ident in sorted(set(IDR.findall(t))):
            k = full.lower().rfind(ident.lower())
            id_c["identifiers"] += 1
            if k < 0:
                id_c["not_in_prompt"] += 1
                continue
            if k > len(sysm):
                id_c["nearest_in_tail"] += 1
                continue
            d = est(sysm[k:]) + tail_tok
            id_c["nearest_beyond_4096"] += d > 4096
            if d > 4096:
                far_any = True
                id_far.append([i, r["scenario_id"], ident, round(d)])
        row_far[(r["scenario_id"], "L2")] = far_any

sc = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
acc = defaultdict(lambda: [0, 0])
for x in sc:
    k = (x["meta"]["scenario_id"], x["meta"]["level"])
    if k in row_far:
        ok = x["components"]["decision"] in ("ask_informed", "tell_informed")
        acc["far" if row_far[k] else "near"][0] += ok
        acc["far" if row_far[k] else "near"][1] += 1
dump("v06c_record_positions.json", {"scenario_record_lines_L2": dict(rec_c), "record_examples_beyond": rec_ex,
                                    "A_target_identifiers_L2": dict(id_c), "identifiers_beyond_4096": id_far,
                                    "m1_A_L2_accuracy_by_comparator_depth": {k: v for k, v in acc.items()}})
print(dict(rec_c), rec_ex)
print(dict(id_c), id_far)
print(dict(acc))
