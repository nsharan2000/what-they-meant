"""GA-4: format robustness of the keyword rule on the unique twin-A message targets (my own perturbations).

Perturbations (each applied alone): bold numbers, bold Capitalised words, bold every keyword-hit span's first word,
'1,234.00' -> '1234', integers -> 'N.00', hyphen -> space, "does not" <-> "doesn't", italics with underscores,
smart quotes. A flip = target stops being ask_informed.
-> results/v_perturb_summary.json
"""
from __future__ import annotations

import json
import re
from collections import Counter

from vcommon import EP, dump, rows, spec_files

specs = spec_files()
uniq = {}
for f in ("sft_train", "sft_val"):
    for r in rows(f):
        if r.get("turn") == "decision" and r["twin"].startswith("A"):
            t = r["completion"][0]["content"]
            if not t.startswith("<tool_call>"):
                uniq[(r["scenario_id"], t)] = specs[r["scenario_id"]]

P = {
    "bold_numbers": lambda t: re.sub(r"(\d[\d,.]*)", r"**\1**", t),
    "bold_capitalised": lambda t: re.sub(r"\b([A-Z][\w.@-]+)", r"**\1**", t),
    "italic_capitalised": lambda t: re.sub(r"\b([A-Z][\w.@-]+)", r"_\1_", t),
    "drop_decimals": lambda t: re.sub(r"(\d)\.\d\d\b", r"\1", t),
    "add_decimals": lambda t: re.sub(r"\b(\d+)\b(?![.,]\d)", r"\1.00", t),
    "hyphen_to_space": lambda t: t.replace("-", " "),
    "contract": lambda t: t.replace("does not", "doesn't").replace("is not", "isn't").replace("did not", "didn't"),
    "expand": lambda t: t.replace("doesn't", "does not").replace("isn't", "is not").replace("didn't", "did not").replace("don't", "do not"),
    "smart_quotes": lambda t: t.replace("'", "’"),
    "currency_word": lambda t: re.sub(r"\$(\d)", r"USD \1", t),
}
S = {"n_unique_A_message_targets": len(uniq), "base_ask_informed": 0, "flips": {}, "examples": {}}
base_ok = {k: EP.classify_decision("message", s, text=k[1]) == "ask_informed" for k, s in uniq.items()}
S["base_ask_informed"] = sum(base_ok.values())
for name, fn in P.items():
    changed = flips = 0
    ex = []
    for (sid, t), s in uniq.items():
        t2 = fn(t)
        if t2 == t:
            continue
        changed += 1
        c = EP.classify_decision("message", s, text=t2)
        if base_ok[(sid, t)] and c != "ask_informed":
            flips += 1
            if len(ex) < 4:
                ex.append([sid, c, t2[:220]])
    S["flips"][name] = [flips, changed]
    S["examples"][name] = ex
dump(S, "v_perturb_summary.json")
print(json.dumps(S, indent=1, ensure_ascii=False)[:5000])
