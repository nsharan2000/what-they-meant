"""V05 - PC4 (zero-look-up decision rows), PC5 (card-content rules vs twin-A informed cards; M1 evidence), PC6 (B card targets that
omit the rail / account / method the card rule asks for).
My own clash->card-field mapping (from tags + reading the targets): a twin-A clash is a CARD FIELD when the env's card rule makes the
approval card itself state the clashing fact (wallet: amount, recipient as shown, rail/type -> recipient/typo/near-miss, amount,
payment type, QR payee; bills: renewal terms / plan price, card used; shopping: quantities/units, payment method, terms; bizops: the
account/method (bank-detail change) and what will happen (S42A wrong action)). Duplicates, stale records, failing tests,
restricted accounts and institution errors are NOT card fields.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/05_pc4_pc5_pc6.py -> results/05_pc4_pc5_pc6.json
"""
import collections
import json
import os
import re

from vcommon import OUT, ROOT, bootstrap_diff, dump, rows, specs

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
SP = specs()
res = {}

# ---------------- PC4 ----------------
for f in ("sft_train", "dpo_train", "grpo_train", "sft_val"):
    d = [r for r in T if r["file"] == f and r["turn"] == "decision"]
    z = [r for r in d if r["n_lookups"] == 0]
    res.setdefault("pc4", {})[f] = {
        "decision_rows": len(d), "zero_lookup": len(z), "scenarios": dict(collections.Counter(r["sid"] for r in z)),
        "zero_lookup_B_L1plus": sum(r["t"] == "B" and r["level"] != "L0r" for r in z),
        "zero_lookup_P2": sum(r["ps"] == "P2" for r in z),
        "lines": sorted(f"{r['sid']}|{r['level']}|{r['ps']}|{r['i']}" for r in z)}
lk = [r for r in T if r["file"] == "sft_train" and r["turn"].startswith("lookup")]
res["pc4"]["sft_lookup_turns"] = len(lk)
res["pc4"]["sft_lookup1_turns_prompt_is_system_plus_request"] = sum(r["turn"] == "lookup1" and r["n_msgs"] == 2 for r in lk)
res["pc4"]["sft_decision_rows_prompt_is_system_plus_request"] = sum(
    r["n_msgs"] == 2 for r in T if r["file"] == "sft_train" and r["turn"] == "decision")
res["pc4"]["specs_with_empty_prefix_calls(train/val/test, any twin)"] = sorted(
    s["id"] for s in SP.values() if not s.get("prefix_calls") and s.get("ideal_action"))
W = [r for r in rows("sft_train") if r["env_id"] == "wallet" and r["level"] != "L0r"]
res["pc4"]["wallet_lookup_first_quote_rows"] = sum("1. Look it up first. Before you prepare a payment" in r["prompt"][0]["content"] for r in W)

# M1 samples
G = {(r["sid"], r["level"]): r for r in T if r["file"] == "grpo_train"}
by = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl")):
    s = json.loads(l)
    by[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"])


def grp(pred, cls):
    return {k: [int(d in cls) for d in v] for k, v in by.items() if pred(G[k])}


z = lambda g: g["n_lookups"] == 0
nz = lambda g: g["n_lookups"] > 0
res["pc4"]["m1_zero_lookup_prompts"] = {f"{k[0]}|{k[1]}|{G[k]['ps']}": dict(collections.Counter(v)) for k, v in by.items() if z(G[k])}
res["pc4"]["m1_lookup_rate_zero_vs_other"] = bootstrap_diff(grp(z, {"lookup"}), grp(nz, {"lookup"}))
res["pc4"]["m1_lookup_samples_other_prompts"] = sum(d == "lookup" for k, v in by.items() if nz(G[k]) for d in v)

# ---------------- PC5 ----------------
CARD_FIELD = {   # twin-A scenario -> clash is a field the env's card rule makes the card state (my reading)
    # wallet
    "S02A": 1, "S03A": 1, "S07A": 1, "S12A": 0, "S34A": 1, "S47A": 1, "S49A": 1, "S49P": 1, "S53A": 1, "S54A": 0, "S55A": 1,
    "S57A": None, "V031A": 1, "V071A": 1, "V072A": 1, "V073A": 1, "V074A": 1, "V341A": 1, "V344A": 1, "V471A": 1, "V472A": 1,
    "V473A": 1, "V474A": 1, "V491A": 1, "V492A": 1, "V531A": 1, "V532A": 1, "V534A": 1, "V541A": 1, "V551A": 1, "V552A": 1,
    "V553A": 1, "V554A": 1, "V571A": None, "V573A": 1,
    # bills
    "S13A": 1, "S18A": 1, "S22A": 0, "S25A": 1, "S32A": 1, "V133A": 1, "V181A": 1, "V182A": 1, "V183A": 1, "V184A": 1, "V223A": 0,
    "V253A": 1, "V254A": 1, "V321A": 1, "V322A": 1,
    # shopping
    "S04A": 1, "S10A": None, "S11A": None, "S35A": 0, "S58A": 1, "V042A": 1, "V043A": 1, "V044A": 1, "V104A": 1, "V113A": None,
    "V581A": 1, "V582A": 1, "V583A": 1, "V584A": 1,
    # bizops
    "S06A": 0, "S09A": 1, "S14A": 0, "S15A": None, "S26A": 0, "S42A": 1, "V061A": 0, "V062A": 0, "V091A": 1, "V093A": 1,
    "V094A": 1, "V152A": None, "V264A": 0,
}
A = [r for r in T if r["file"] == "sft_train" and r["turn"] == "decision" and r["t"] == "A"]
assert all(r["sid"] in CARD_FIELD for r in A), sorted({r["sid"] for r in A} - set(CARD_FIELD))
res["pc5"] = {"A_L1plus_P0_card_field_rows": sum(CARD_FIELD[r["sid"]] == 1 and r["level"] != "L0r" and r["ps"] == "P0" for r in A),
              "A_L1plus_P0_rows": sum(r["level"] != "L0r" and r["ps"] == "P0" for r in A)}
ai = {"approve_informed"}
At = lambda g: g["t"] == "A"
lv = collections.defaultdict(lambda: [0, 0])
for k, v in by.items():
    if At(G[k]):
        lv[G[k]["level"]][0] += sum(d == "approve_informed" for d in v)
        lv[G[k]["level"]][1] += len(v)
res["pc5"]["m1_A_approve_informed_by_level"] = {k: [a, n, round(a / n, 3)] for k, (a, n) in sorted(lv.items())}
res["pc5"]["m1_A_approve_informed_total"] = sum(a for a, n in lv.values())
cf = lambda g: At(g) and CARD_FIELD.get(g["sid"]) == 1
ncf = lambda g: At(g) and CARD_FIELD.get(g["sid"]) == 0
L1P0 = lambda g: g["level"] != "L0r" and g["ps"] == "P0"
L0P0 = lambda g: g["level"] == "L0r" and g["ps"] == "P0"
nonw = lambda g: g["env"] != "wallet"
res["pc5"]["m1_cardfield_vs_not_L1plus_P0"] = bootstrap_diff(grp(lambda g: cf(g) and L1P0(g), ai), grp(lambda g: ncf(g) and L1P0(g), ai))
res["pc5"]["m1_cardfield_vs_not_L0r_P0"] = bootstrap_diff(grp(lambda g: cf(g) and L0P0(g), ai), grp(lambda g: ncf(g) and L0P0(g), ai))
res["pc5"]["m1_cardfield_L1plus_vs_L0r_P0_all_envs"] = bootstrap_diff(grp(lambda g: cf(g) and L1P0(g), ai), grp(lambda g: cf(g) and L0P0(g), ai))
res["pc5"]["m1_cardfield_L1plus_vs_L0r_P0_non_wallet(wallet tool doc carries a card rule at L0r too)"] = bootstrap_diff(
    grp(lambda g: cf(g) and L1P0(g) and nonw(g), ai), grp(lambda g: cf(g) and L0P0(g) and nonw(g), ai))
res["pc5"]["m1_cardfield_vs_not_L1plus_all_settings"] = bootstrap_diff(grp(lambda g: cf(g) and g["level"] != "L0r", ai),
                                                                      grp(lambda g: ncf(g) and g["level"] != "L0r", ai))
res["pc5"]["m1_approve_informed_by_env_A"] = {e: [sum(d == "approve_informed" for k, v in by.items() if At(G[k]) and G[k]["env"] == e for d in v),
                                                  sum(len(v) for k, v in by.items() if At(G[k]) and G[k]["env"] == e)]
                                              for e in ("wallet", "bills", "shopping", "bizops")}

# ---------------- PC6 ----------------
RAIL = re.compile(r"PayPal|Venmo|Zelle|Cash App|\bUPI\b|Friends (?:&|and) Family|Goods (?:&|and) Services|\bpersonal\b|purchase payment|"
                  r"\bbank\b|ending (?:in )?\d|\*{2,}\d|••\d|x\d{4}\b|checking|savings|\bcard\b|Visa|Mastercard|Amex|debit|credit|"
                  r"\bACH\b|\bwire\b|on file|Pay in 3|wallet|Solana|Ethereum|Polygon|network|\bQR\b|e-mandate|from your|account",
                  re.I)
NONMONEY = re.compile(r"\bdeploy|cancel", re.I)
cards = []
for r in T:
    if r["file"] == "sft_train" and r["turn"] == "decision" and r["t"] == "B" and r["completion"]["decision"].startswith("approve"):
        tx = r["completion"]["text"]
        try:
            args = json.loads(re.search(r"<tool_call>(.*)</tool_call>", tx, re.S).group(1))["arguments"]
        except Exception:
            args = {}
        summ = " ".join(str(args.get(k, "")) for k in ("summary", "details"))
        rule = r["level"] != "L0r" or r["env"] == "wallet"
        cards.append({"sid": r["sid"], "env": r["env"], "level": r["level"], "i": r["i"], "summary": summ, "rule_in_prompt": rule,
                      "nonmoney": bool(NONMONEY.search(summ)), "states_method": bool(RAIL.search(summ))})
ruled = [c for c in cards if c["rule_in_prompt"] and not c["nonmoney"]]
res["pc6"] = {"B_card_targets": len(cards), "with_card_rule_in_prompt_excl_deploy_cancel": len(ruled),
              "omit_method_lenient_regex": sum(not c["states_method"] for c in ruled),
              "by_env": {e: [sum(not c["states_method"] for c in ruled if c["env"] == e), sum(c["env"] == e for c in ruled)]
                         for e in ("wallet", "bills", "shopping", "bizops")},
              "distinct_omitting_summaries": sorted({(c["sid"], c["summary"]) for c in ruled if not c["states_method"]})[:400],
              "distinct_stating_summaries": sorted({(c["sid"], c["summary"]) for c in ruled if c["states_method"]})}
p = dump("05_pc4_pc5_pc6.json", res)
print(json.dumps({k: (v if k != "pc6" else {kk: vv for kk, vv in v.items() if not kk.startswith("distinct")}) for k, v in res.items()},
                 indent=1, ensure_ascii=False)[:8000])
print("wrote", p)
