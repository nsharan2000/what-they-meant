"""V09 - small style/privacy checks of decision targets against policy wording: unmasked long numbers (privacy rules: mask to last
four digits), internal reference / policy ids or tool names in member-facing messages (bills: 'do not mention internal policy names or
tool names'), and targets on L3 (noisy, code-mixed) rows vs the wallet rule 'reply in their language'.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/09_style.py -> results/09_style.json
"""
import collections
import json
import os
import re

from vcommon import OUT, dump

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
rows = [r for r in T if r["file"] in ("sft_train", "dpo_train") and r["turn"] == "decision"]
role = lambda r: "completion" if "completion" in r else "chosen"
res = {"rows": len(rows)}
LONGNUM = re.compile(r"(?<![\d*•.])\d{8,}(?![\d])")
res["unmasked_8plus_digit_numbers"] = sorted({(r["sid"], m) for r in rows for m in LONGNUM.findall(r[role(r)]["text"])})
INTERNAL = re.compile(r"\b(?:SA-\d+|SI-\d+|standing authori[sz]ation|standing instruction|policy)\b", re.I)
msg = [r for r in rows if r[role(r)]["kind"] == "message"]
res["messages_naming_internal_ids_or_policy"] = sorted({(r["sid"], r["env"], m.group(0)) for r in msg
                                                        for m in INTERNAL.finditer(r[role(r)]["text"])})
res["bills_messages_naming_internal_ids_or_policy"] = sum(bool(INTERNAL.search(r[role(r)]["text"])) for r in msg if r["env"] == "bills")
nonen = [r for r in rows if r["level"] == "L3" and re.search(r"\b(jaldi|bhai|yaar|kya|hai|pls|thx)\b", r["user"], re.I)]
res["L3_codemixed_user_rows"] = len(nonen)
res["L3_codemixed_examples"] = [(r["sid"], r["user"][:90]) for r in nonen[:4]]
p = dump("09_style.json", res)
print(json.dumps(res, indent=1, ensure_ascii=False))
print("wrote", p)
