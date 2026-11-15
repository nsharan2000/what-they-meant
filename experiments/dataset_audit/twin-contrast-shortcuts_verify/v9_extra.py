"""V9 (verifier): extra checks.
  (a) DPO length cue in exact Tiny Aya TOKENS (token_stats chosen_tokens / rejected_tokens), by twin -- DPO's implicit reward
      sums token log-probs, so tokens (not chars) are the relevant length;
  (b) the first-token prior AT THE DECISION POINT: SFT rows whose prompt ends with a tool result (decision rows + lookup2), split
      by whether the target starts with a tool call -- the analyst's 0.726 '<' share mixes in look-up rows that follow a user turn;
  (c) P2 hint vs target: B rows whose target is a direct commit (expected 'act') rendered with the P2 hint, which says
      'if everything matches, just show the approval card';
  (d) twin-request kinds per family (source cluster) in train, to count how many independent families carry wording differences.

python3 -B experiments/dataset_audit/twin-contrast-shortcuts_verify/v9_extra.py -> results/v9_extra.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_twins import req_kind  # noqa: E402
from vcommon import HINT, T, all_specs, curated, save, splits, token_stats  # noqa: E402


def main():
    out = {}
    ts = token_stats()
    pair_split, cluster = splits()
    # (a)
    dpo = curated("dpo_train")
    tok = defaultdict(lambda: Counter())
    for r in dpo:
        s = ts[("dpo_train", r["_i"])]
        c, j = s["chosen_tokens"], s["rejected_tokens"]
        tok[T(r)]["rejected_longer" if j > c else ("chosen_longer" if c > j else "tie")] += 1
    out["dpo_length_tokens"] = {t: dict(v) | {"share_rejected_longer": round(v["rejected_longer"] / sum(v.values()), 3)} for t, v in tok.items()}
    # (b)
    sft = curated("sft_train")
    after_result = Counter()
    after_request = Counter()
    for r in sft:
        st = ts[("sft_train", r["_i"])]
        lt = st["completion_first5"][0] == "<"
        if r["prompt"][-1]["role"] == "user" and r["prompt"][-1]["content"].startswith("<tool_result"):
            after_result[("tool_call_first" if lt else "message_first", r["turn"])] += 1
        else:
            after_request[("tool_call_first" if lt else "message_first", r["turn"])] += 1
    out["first_token_after_tool_result"] = {f"{a}|{b}": n for (a, b), n in after_result.items()}
    out["first_token_after_user_request"] = {f"{a}|{b}": n for (a, b), n in after_request.items()}
    n_res = sum(after_result.values())
    out["share_tool_call_first_after_tool_result"] = round(sum(n for (a, _), n in after_result.items() if a == "tool_call_first") / n_res, 3)
    # (c)
    grpo = {(r["scenario_id"], r["level"]): r for r in curated("grpo_train")}
    act_rows = []
    for r in sft:
        if r["turn"] != "decision" or T(r) != "B":
            continue
        g = grpo.get((r["scenario_id"], r["level"]))
        exp = json.loads(g["scenario"])["expected"]["decision"] if g else None
        tgt = r["completion"][0]["content"]
        is_commit = tgt.startswith("<tool_call>") and '"request_approval"' not in tgt[:60]
        if is_commit:
            act_rows.append({"i": r["_i"], "sid": r["scenario_id"], "level": r["level"], "ps": r["prompt_setting"], "expected": exp,
                             "hint_in_prompt": r["prompt"][0]["content"].rstrip().endswith(HINT), "target": tgt[:110]})
    out["B_commit_target_rows"] = {"n": len(act_rows), "with_P2_hint": sum(x["hint_in_prompt"] for x in act_rows),
                                   "rows_with_hint": [x for x in act_rows if x["hint_in_prompt"]]}
    # (d) request kinds by family in train
    specs = all_specs()
    byp = defaultdict(dict)
    for s in specs.values():
        if T(s) in "AB":
            byp[s["pair_id"]][T(s)] = s
    fam = defaultdict(Counter)
    for p, sp in pair_split.items():
        if sp == "train" and "A" in byp.get(p, {}) and "B" in byp[p]:
            fam[cluster[p]][req_kind(byp[p]["A"]["user_message"], byp[p]["B"]["user_message"])] += 1
    out["train_families"] = len(fam)
    out["train_families_with_wording_pairs"] = sorted(c for c, v in fam.items() if v["wording"])
    out["train_families_with_identifier_only_pairs"] = sorted(c for c, v in fam.items() if v["identifier_only"])
    out["train_families_identical_only"] = sorted(c for c, v in fam.items() if set(v) == {"identical"})
    save(out, "v9_extra.json")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
