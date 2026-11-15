"""GA-6 / GA-9 spec-level checks (own code) over every spec (seed + variants + eval files incl. hard sets).

- literal-request echo on twin A: approval card {summary: request} and my own card text 'Please approve: <request>' -> class
- gold / harmful keys that json.dumps(args) of a numeric argument can never produce ('NN.00'), or a tool name used as key
- V541A: can a TEXT approval card ever reach approve_gold?  (card summary with '$40.00')
- B specs: is informed_any identical to the A twin's list?
- suspicious keywords: <=4 chars, truncated words, very generic phrases
-> results/v_specs_summary.json
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

from vcommon import EP, ROOT, SP, dump, jl, kinds

specs = {}
for pat in ("dataset/scenarios/*.jsonl", "dataset/scenarios_synth/*.jsonl", "dataset/eval/*.jsonl"):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        for s in jl(f):
            specs.setdefault(s["id"], (os.path.basename(f), s))
tool_names = set()
for e in ("wallet", "bills", "shopping", "bizops"):
    tool_names |= set(kinds(e))

S = {"n_specs": len(specs), "A_echo_card_classes": Counter(), "A_echo_gold": [], "A_mycard_gold": [], "unmatchable_keys": [],
     "B_informed_any_eq_A": Counter(), "short_or_truncated_keywords": []}
for sid, (fn, s) in sorted(specs.items()):
    if s["twin"].startswith("A_should"):
        c1 = EP.classify_decision("ui", s, args={"summary": s["user_message"]})
        c2 = EP.classify_decision("ui", s, args={"summary": f"Please approve: {s['user_message']}", "amount_note": "as requested"})
        S["A_echo_card_classes"][c1] += 1
        if c1 == "approve_gold":
            S["A_echo_gold"].append([sid, fn, s["user_message"][:90], s.get("gold_commits")])
        if c2 == "approve_gold":
            S["A_mycard_gold"].append([sid, fn])
    for which in ("gold_commits", "harmful_commits"):
        for sig in s.get(which) or []:
            for k in sig.get("match", []):
                if k in tool_names:
                    S["unmatchable_keys"].append([sid, fn, which, k, "tool name; _target/_commit_matches see only the arguments"])
                elif re.fullmatch(r"\d+\.\d*0", k):
                    S["unmatchable_keys"].append([sid, fn, which, k, "decimal ending in 0: a JSON number never serialises like this"])
    if s["twin"].startswith("B"):
        a = specs.get(s["pair_id"] + "A")
        if a:
            S["B_informed_any_eq_A"][str(a[1]["expected"].get("informed_any") == s["expected"].get("informed_any"))] += 1
    for k in s["expected"].get("informed_any") or []:
        n = SP.norm(k).strip()
        if (len(n) <= 4 and not re.fullmatch(r"[\d.]+[km]?", n)) or re.search(r"\b\w{1,3}$", n) and " " in n and n.split()[-1] in ("ava", "unav", "acc", "pay"):
            S["short_or_truncated_keywords"].append([sid, k])
# V541A text card / numeric card
v541 = specs["V541A"][1]
S["V541A_text_card"] = EP.classify_decision("ui", v541, args={"summary": "Send $40.00 to @green-thumb-james for the lawn service?"})
S["V541A_numeric_card"] = EP.classify_decision("ui", v541, args={"recipient": "@green-thumb-james", "amount": 40.0, "memo": "lawn service"})
S["V541A_numeric_commit"] = EP.classify_decision("commit", v541, args={"recipient": "@green-thumb-james", "amount": 40, "memo": "lawn service"})
S["V541A_harmful_numeric_commit"] = EP.classify_decision("commit", v541, args={"recipient": "@ethan-cole-lawns", "amount": 40})
# H221A harmful key '214.8' vs text '214.80'
h221 = specs["H221A"][1]
S["H221A_card_214.80_text"] = EP.classify_decision("ui", h221, args={"summary": "Pay LES bill $214.80?"})
S["H221A_commit_214.8_number"] = EP.classify_decision("commit", h221, args={"payee": "LES", "amount": 214.8})
S["H221A_commit_214.80_string"] = EP.classify_decision("commit", h221, args={"payee": "LES", "amount": "214.80"})
# V491A ideal message (self-contradiction check is a read): print it
S["V491A"] = {"user": specs["V491A"][1]["user_message"], "gold": specs["V491A"][1]["gold_commits"],
              "harmful": specs["V491A"][1]["harmful_commits"], "ideal": specs["V491A"][1]["expected"]["ideal_message"]}
S["A_echo_card_classes"] = dict(S["A_echo_card_classes"])
S["B_informed_any_eq_A"] = dict(S["B_informed_any_eq_A"])
dump(S, "v_specs_summary.json")
print(json.dumps(S, indent=1, ensure_ascii=False)[:6000])
