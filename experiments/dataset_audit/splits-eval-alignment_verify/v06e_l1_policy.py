"""SEA-6 sub-claim: share of L1 / L2 decision rows whose environment policy START lies > 4,096 tokens from the end
(char-class token model from v06b; exact total prompt tokens from token_stats.jsonl as a cross-check).
Output: results/v06e_l1_policy.json
"""
import json
import os
import re
from collections import Counter

import numpy as np

from vcommon import ROOT, curated, dump

w = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "results", "v06b_tokfit_positions.json")))["weights"])
TPL = open(os.path.join(ROOT, "third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja")).read()
PRE = TPL.split("<|SYSTEM_TOKEN|>", 1)[1].split("{% if ns.system_prompt")[0]
stats = {r["i"]: r["prompt_tokens"] for r in map(json.loads, open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl"))) if r["file"] == "sft_train.jsonl"}


def est(t):
    f = np.array([sum(c.isalpha() and c.isascii() for c in t), sum(c.isdigit() for c in t), sum(c.isspace() for c in t),
                  sum((not c.isalnum()) and (not c.isspace()) and c.isascii() for c in t), sum(not c.isascii() for c in t),
                  len(re.findall(r"[A-Za-z]+", t)), len(re.findall(r"\d+", t)), 0.0], dtype=float)
    return float(f @ w)


c = Counter()
for i, r in enumerate(curated("sft_train")):
    if r["turn"] != "decision" or r["level"] not in ("L1", "L2"):
        continue
    p = r["prompt"]
    specials = 1 + sum(3 + (2 if m["role"] == "assistant" else 0) for m in p[1:]) + 3
    d_policy_start = est(p[0]["content"] + "".join(m["content"] for m in p[1:])) + specials
    # cross-check: exact total minus the estimated preamble (= policy start distance via exact counts)
    d_exact = stats[i] - (est(PRE) + 4 + 40)      # ~40 tokens for the developer-preamble header
    c[(r["level"], "rows")] += 1
    c[(r["level"], "policy_start_beyond_4096(model)")] += d_policy_start > 4096
    c[(r["level"], "policy_start_beyond_4096(exact_total_minus_preamble)")] += d_exact > 4096
dump("v06e_l1_policy.json", {f"{a}|{b}": v for (a, b), v in sorted(c.items())})
print({f"{a}|{b}": v for (a, b), v in sorted(c.items())})
