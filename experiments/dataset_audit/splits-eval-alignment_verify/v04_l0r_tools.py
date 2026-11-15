"""SEA-4 verification: at L0r, do the twins of a pair see different tool lists?

For every pair (training pairs = pairs with rows in sft_train; eval sets from dataset/eval) compare l0_tools of A and B, and
the rendered L0r system prompts (P0): differ at all / differ only in the tool-docs block / member block differs.
Also: do the twins' CANONICAL look-ups use different tools (a policy cannot pick 'the' canonical look-up for both twins from an
identical first prompt)?
Output: results/v04_l0r_tools.json
"""
from collections import Counter

from vcommon import curated, dump, env, eval_sets, member, render, specs

by_id = {s["id"]: s for s in specs()}
sft = curated("sft_train")
train_ids = {r["scenario_id"] for r in sft}
sets = {"train": [by_id[i] for i in sorted(train_ids)]}
sets.update(eval_sets())

out = {}
for name, rows in sets.items():
    pairs = {}
    for s in rows:
        if s["twin"].startswith("A_pushback"):
            continue
        pairs.setdefault(s["pair_id"], {})[s["twin"][0]] = s
    c = Counter()
    diff_pairs, lookup_tool_diff = [], []
    for pid, t in sorted(pairs.items()):
        if len(t) != 2:
            c["incomplete_pairs"] += 1
            continue
        a, b = t["A"], t["B"]
        c["pairs"] += 1
        ta, tb = set(a["l0_tools"]), set(b["l0_tools"])
        sa = render(env(a["env_id"]), member(a["env_id"], a["member_id"]), a, "L0r", "P0")
        sb = render(env(b["env_id"]), member(b["env_id"], b["member_id"]), b, "L0r", "P0")
        if ta != tb:
            c["l0_tools_differ"] += 1
            c["extra_tool_only_in_A"] += bool(ta - tb) and not (tb - ta)
            c["extra_tool_only_in_B"] += bool(tb - ta) and not (ta - tb)
            c["both_have_extra"] += bool(ta - tb) and bool(tb - ta)
            same_block = a["l0_member_block"] == b["l0_member_block"]
            diff_pairs.append({"pair": pid, "only_A": sorted(ta - tb), "only_B": sorted(tb - ta),
                               "member_block_same": same_block, "same_request": a["user_message"] == b["user_message"],
                               "L0r_system_identical_otherwise": same_block})
        c["L0r_first_prompt_identical"] += (sa[0] == sb[0] and sa[1] == sb[1])
        pa = [(x["tool"]) for x in a.get("prefix_calls") or []]
        pb = [(x["tool"]) for x in b.get("prefix_calls") or []]
        if pa[:1] != pb[:1]:
            lookup_tool_diff.append({"pair": pid, "A_first_lookup": pa[:1], "B_first_lookup": pb[:1],
                                     "same_request": a["user_message"] == b["user_message"]})
        c["first_lookup_tool_differs"] += pa[:1] != pb[:1]
        c["first_lookup_tool_differs_and_same_request"] += (pa[:1] != pb[:1]) and a["user_message"] == b["user_message"]
        fa = [(x["tool"], str(sorted(x["arguments"].items()))) for x in (a.get("prefix_calls") or [])[:1]]
        fb = [(x["tool"], str(sorted(x["arguments"].items()))) for x in (b.get("prefix_calls") or [])[:1]]
        c["first_lookup_call_differs(tool or args)"] += fa != fb
        c["first_lookup_call_differs_and_same_request"] += (fa != fb) and a["user_message"] == b["user_message"]
    out[name] = {"counts": dict(c), "pairs_with_l0_tool_diff": diff_pairs, "pairs_with_first_lookup_tool_diff": lookup_tool_diff}

dump("v04_l0r_tools.json", out)
for k, v in out.items():
    print(k, v["counts"])
    print("   ", [(d["pair"], d["only_A"], d["only_B"]) for d in v["pairs_with_l0_tool_diff"]])
    print("    first-lookup tool diffs:", [(d["pair"], d["A_first_lookup"], d["B_first_lookup"], d["same_request"]) for d in v["pairs_with_first_lookup_tool_diff"]][:12])
