"""SEA-10 verification: the forced Tiny Aya chat-template preamble — is its text available locally, how long is it, and does it
contain anything that biases toward asking?

Source: third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja (identical copy in AuditKIT); its length is
compared with research/aya-models.md ('Global, Fire, Water and Earth share one 3,239-character template'; 'forced preamble 389').
Token estimate: the character-class model fitted in v06b (results/v06b_tokfit_positions.json), plus the special tokens.
Output: results/v10_preamble.json
"""
import json
import os
import re

import numpy as np

from vcommon import ROOT, dump

p1 = os.path.join(ROOT, "third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja")
p2 = os.path.join(ROOT, "third_party/AuditKIT/tests/fixtures/chat_templates/tiny_aya_default.jinja")
tpl = open(p1).read()
same_copy = tpl == open(p2).read()
pre = tpl.split("<|SYSTEM_TOKEN|>", 1)[1].split("{% if ns.system_prompt")[0]
dev = tpl.split("{% if ns.system_prompt and ns.system_prompt != \"\" %}", 1)[1].split("{{ ns.system_prompt }}")[0]
lines = [l.strip() for l in (pre + dev).splitlines() if l.strip()]
ask_rx = re.compile(r"\b(ask|clarif\w*|follow[- ]?up|question\w*|confirm\w*|check)\b", re.I)
ask_lines = [l for l in lines if ask_rx.search(l)]
research = open(os.path.join(ROOT, "research/aya-models.md")).read()
claims = [l.strip() for l in research.splitlines() if re.search(r"3,239|389|Forced preamble|Developer Preamble", l)]

w = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "results", "v06b_tokfit_positions.json")))["weights"])


def feats(t):
    return np.array([sum(c.isalpha() and c.isascii() for c in t), sum(c.isdigit() for c in t), sum(c.isspace() for c in t),
                     sum((not c.isalnum()) and (not c.isspace()) and c.isascii() for c in t), sum(not c.isascii() for c in t),
                     len(re.findall(r"[A-Za-z]+", t)), len(re.findall(r"\d+", t)), 0.0], dtype=float)


est_tokens = float(feats(pre + dev) @ w) + 4      # BOS, START_OF_TURN, SYSTEM_TOKEN, END_OF_TURN
out = {"template_path": os.path.relpath(p1, ROOT), "template_chars": len(tpl), "auditkit_copy_identical": same_copy,
       "preamble_chars(system+default+developer header)": len(pre + dev), "estimated_preamble_tokens_incl_specials": round(est_tokens),
       "preamble_lines": lines, "lines_matching_ask_clarify_followup_question_confirm_check": ask_lines,
       "research_aya_models_md_lines": claims}
dump("v10_preamble.json", out)
print(json.dumps(out, indent=1)[:4000])
