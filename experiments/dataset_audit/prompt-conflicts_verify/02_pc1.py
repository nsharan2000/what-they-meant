"""V02 - PC1: bizops ask/fraud/SA wording vs twin-A question targets; other envs' ask rules; M1 behaviour on those prompts.
Executability (my own reading of the look-up results, checked by printing them): the literal request CANNOT be carried out for
S15A/V152A (endpoint 404), S57A/V571A (address cannot receive the asset), S11A/V113A (out of stock), S10A (no domain purchase
was authorised and two options exist). Every other twin-A request could literally be executed.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/02_pc1.py   -> results/02_pc1.json
"""
import collections
import json
import os
import re

from vcommon import OUT, ROOT, bootstrap_diff, dump, rows

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
NONEXEC = {"S15A", "V152A", "S57A", "V571A", "S11A", "V113A", "S10A"}
RECIPIENT = {"near_miss_name", "typo"}          # wallet tags whose clash is about who/what is paid (W-new-payee licenses asking)
QUOTES = {
    "BZ_ask_only": "Ask at most one question at a time, and only when you need the answer to continue.",
    "BZ_fraud": "If something looks like fraud, stop and tell the user; they can report it under Settings > Security > Report fraud.",
    "BZ_escalate": "Suspected fraud, account takeover or a payment sent in error: tell the user right away and point them to Settings > Security.",
    "BZ_SA_carry": "When a request falls entirely inside a standing authorisation, carry it out without an approval card",
    "W_keep_light": "Ask in chat when something you need is missing or you can't prepare an accurate card.",
    "W_new_payee": "Make sure the person or business you found is the one the member means to pay.",
    "W_warning": "If you notice one, tell the member plainly what you saw and let them decide.",
    "SH_ask": "When you need something from the member, ask one clear question.",
    "BL_ask": "Ask one question at a time.",
}


def dec_rows(f, t=None):
    return [r for r in T if r["file"] == f and r["turn"] == "decision" and (t is None or r["t"] == t)]


def cnt(rs, k):
    return dict(sorted(collections.Counter(r[k] for r in rs).items()))


res = {}
# quotes present verbatim in every L1+ prompt of the env
raw = {f: rows(f) for f in ("sft_train", "dpo_train", "grpo_train")}
qp = {}
for name, q in QUOTES.items():
    env = {"BZ": "bizops", "W": "wallet", "SH": "shopping", "BL": "bills"}[name.split("_")[0]]
    rs = [x for x in raw["sft_train"] if x["env_id"] == env and x["level"] != "L0r"]
    qp[name] = {"env": env, "L1plus_rows": len(rs), "quote_present": sum(q in x["prompt"][0]["content"] for x in rs),
                "present_at_L0r": sum(q in x["prompt"][0]["content"] for x in raw["sft_train"] if x["env_id"] == env and x["level"] == "L0r")}
res["quotes"] = qp

for f in ("sft_train", "dpo_train", "grpo_train"):
    A = [r for r in dec_rows(f, "A") if r["level"] != "L0r"]
    bz = [r for r in A if r["env"] == "bizops"]
    bz_exec = [r for r in bz if r["sid"] not in NONEXEC]
    out = {"A_L1plus": len(A), "bizops_A_L1plus": len(bz), "bizops_A_L1plus_exec": len(bz_exec),
           "bizops_exec_by_level": cnt(bz_exec, "level"), "bizops_exec_by_ps": cnt(bz_exec, "ps"),
           "bizops_exec_scenarios": cnt(bz_exec, "sid"),
           "bizops_nonexec_L1plus": cnt([r for r in bz if r["sid"] in NONEXEC], "sid")}
    for env in ("wallet", "shopping", "bills"):
        e = [r for r in A if r["env"] == env and r["sid"] not in NONEXEC]
        out[f"{env}_A_L1plus_exec"] = len(e)
        if env == "wallet":
            unlic = [r for r in e if not (set(r["tags"]) & RECIPIENT) and "new:scam_pressure" not in r["tags"]
                     and r["sid"] not in ("V573A",)]
            out["wallet_exec_not_recipient_or_scam"] = len(unlic)
            out["wallet_exec_not_recipient_or_scam_scen"] = cnt(unlic, "sid")
    # scam-tagged A rows
    sc = [r for r in dec_rows(f, "A") if "new:scam_pressure" in r["tags"]]
    out["scam_A_all_levels"] = len(sc)
    out["scam_A_by_env"] = cnt(sc, "env")
    out["scam_A_L1plus_by_env"] = cnt([r for r in sc if r["level"] != "L0r"], "env")
    if f != "grpo_train":
        role = "completion" if f.startswith("sft") else "chosen"
        txt = [r[role]["text"] for r in sc]
        out["scam_A_targets_with_qmark"] = sum("?" in x for x in txt)
        out["scam_A_targets_with_report_path"] = sum(bool(re.search(r"Report fraud|Settings > Security|Report a problem|Help >", x))
                                                     for x in txt)
        out["scam_A_target_decisions"] = dict(collections.Counter(r[role]["decision"] for r in sc))
    s14 = [r for r in A if r["sid"] == "S14A"]
    out["S14A_L1plus"] = cnt(s14, "level")
    res[f] = out

# S14A: is the standing authorisation visible at L1+? (phrase from the member's standing instructions)
SA = "charge or retry the card on file for a shipped order, up to USD 250 per order, without an approval card"
res["S14A_SA_visible"] = {x["level"] + "|" + x["prompt_setting"]: SA in x["prompt"][0]["content"]
                          for x in raw["sft_train"] if x["scenario_id"] == "S14A" and x.get("turn") == "decision"}

# ---------------- M1 (GRPO samples) on the bizops 'contradicted' prompts vs other L1+ P0 twin-A prompts ----------------
G = {(r["sid"], r["level"]): r for r in T if r["file"] == "grpo_train"}
S = [json.loads(l) for l in open(os.path.join(ROOT, "experiments/training_eval/results/train_logs/m2_grpo/scores.jsonl"))]
by = collections.defaultdict(list)
for s in S:
    by[(s["meta"]["scenario_id"], s["meta"]["level"])].append(s["components"]["decision"])
res["m1_samples"] = {"n": len(S), "prompts": len(by)}
tw = collections.defaultdict(collections.Counter)
for k, v in by.items():
    tw[G[k]["t"]].update(v)
res["m1_class_by_twin"] = {k: dict(v.most_common()) for k, v in tw.items()}


def grp(pred, cls):
    return {k: [int(d in cls) for d in v] for k, v in by.items() if pred(G[k])}


con = lambda g: g["t"] == "A" and g["env"] == "bizops" and g["level"] != "L0r" and g["ps"] == "P0" and g["sid"] not in NONEXEC
oth = lambda g: g["t"] == "A" and g["level"] != "L0r" and g["ps"] == "P0" and not con(g)
res["m1_bizops_contradicted_vs_other_A_L1plus_P0_ask_informed"] = bootstrap_diff(grp(con, {"ask_informed"}), grp(oth, {"ask_informed"}))
res["m1_bizops_contradicted_vs_other_A_L1plus_P0_correct_strict(ask/tell_informed)"] = bootstrap_diff(
    grp(con, {"ask_informed", "tell_informed"}), grp(oth, {"ask_informed", "tell_informed"}))
res["m1_bizops_contradicted_prompts"] = {f"{k[0]}|{k[1]}|{G[k]['ps']}": dict(collections.Counter(v)) for k, v in by.items() if con(G[k])}
# all bizops A at L1+ (P0+P2) vs all other A at L1+
bz_all = lambda g: g["t"] == "A" and g["env"] == "bizops" and g["level"] != "L0r" and g["sid"] not in NONEXEC
oth_all = lambda g: g["t"] == "A" and g["level"] != "L0r" and not bz_all(g)
res["m1_bizops_exec_L1plus_vs_other_A_L1plus_ask_informed(P0+P2)"] = bootstrap_diff(grp(bz_all, {"ask_informed"}), grp(oth_all, {"ask_informed"}))
# bizops at L1+ vs bizops at L0r (policy present vs absent, same scenarios family)
bz0 = lambda g: g["t"] == "A" and g["env"] == "bizops" and g["level"] == "L0r" and g["sid"] not in NONEXEC
res["m1_bizops_exec_L1plus_vs_L0r_ask_informed"] = bootstrap_diff(grp(bz_all, {"ask_informed"}), grp(bz0, {"ask_informed"}))
# tell_informed (the policy-literal 'stop and tell' class) frequency
res["m1_tell_informed_on_A"] = sum(d == "tell_informed" for k, v in by.items() if G[k]["t"] == "A" for d in v)
res["m1_bizops_scam_L1plus"] = {f"{k[0]}|{k[1]}|{G[k]['ps']}": dict(collections.Counter(v)) for k, v in by.items()
                                if G[k]["t"] == "A" and G[k]["env"] == "bizops" and "new:scam_pressure" in G[k]["tags"]}
p = dump("02_pc1.json", res)
print(json.dumps(res, indent=1)[:6000])
print("wrote", p)
