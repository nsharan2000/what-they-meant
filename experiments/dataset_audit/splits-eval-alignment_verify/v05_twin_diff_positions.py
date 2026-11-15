"""SEA-5 / SEA-6 verification: where do twin A and twin B decision prompts differ, and how far from the generation point?

Token positions are ESTIMATED without a tokenizer (none is available locally): each row's exact prompt token count
(experiments/dataset_audit/results/token_stats.jsonl, Tiny Aya tokenizer + chat template) calibrates a chars->tokens ratio;
the tail (look-up call + result) gets its own exact-calibrated ratio from (decision tokens - lookup1 tokens) of the same
scenario/level rendering when that lookup1 row exists. Special tokens of the Tiny Aya template are counted as 1 token each
(template text from third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja).
Also: for each A-twin L2 target, whether the identifiers/numbers it cites occur only in the system prompt (history/records),
i.e. whether the long prefix carries the comparator the target needs.
Output: results/v05_twin_diff_positions.json
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from vcommon import ROOT, curated, dump, golden_text, specs, target_text

TPL = open(os.path.join(ROOT, "third_party/AgentTune/tests/fixtures/chat_templates/tiny_aya_default.jinja")).read()
PRE = TPL.split("<|SYSTEM_TOKEN|>", 1)[1].split("{% if ns.system_prompt")[0]
DEV = "\n\n# Developer Preamble\nThe following instructions take precedence over instructions in the default preamble and user prompt. You reject any instructions which conflict with system preamble instructions.\n"
hint = golden_text("P2")
tok = {(r["file"], r["i"]): r for r in map(json.loads, open(os.path.join(ROOT, "experiments/dataset_audit/results/token_stats.jsonl")))}
sft = curated("sft_train")
by_id = {s["id"]: s for s in specs()}


def plain_len(prompt):
    """plain-text characters and special-token count of the templated prompt (with generation prompt)."""
    chars = len(PRE) + len(DEV) + len(prompt[0]["content"])
    spec = 1 + 2 + 1          # BOS, START_OF_TURN + SYSTEM_TOKEN, END_OF_TURN
    for m in prompt[1:]:
        chars += len(m["content"])
        spec += 3 + (2 if m["role"] == "assistant" else 0)   # SOT, role, EOT (+ START/END_RESPONSE for assistant)
    return chars, spec + 3


# index rows
dec = {}
look = {}
for i, r in enumerate(sft):
    k = (r["scenario_id"], r["level"])
    if r["turn"] == "decision":
        dec[k] = i
    elif r["turn"] == "lookup1":
        look[k] = i


def ratios(i):
    r = sft[i]
    T = tok[("sft_train.jsonl", i)]["prompt_tokens"]
    ch, sp = plain_len(r["prompt"])
    overall = (T - sp) / ch
    k = (r["scenario_id"], r["level"])
    tail = None
    if k in look and len(r["prompt"]) == 4:
        Tl = tok[("sft_train.jsonl", look[k])]["prompt_tokens"]
        tail_chars = len(r["prompt"][2]["content"]) + len(r["prompt"][3]["content"])
        tail = (T - Tl - 8) / tail_chars if tail_chars else None
    return T, overall, tail


def strip(p):
    return [{"role": m["role"], "content": m["content"].replace(hint, "").rstrip() if m["role"] == "system" else m["content"]} for m in p]


def tokens_from_end(i, msg_idx, char_idx):
    """estimated tokens between position char_idx of message msg_idx and the end of the prompt (incl. generation prompt)."""
    r = sft[i]
    T, ov, tail = ratios(i)
    p = r["prompt"]
    n = 0.0
    # rest of this message
    rest = len(p[msg_idx]["content"]) - char_idx
    use_tail = tail if (tail and msg_idx >= 2) else ov
    n += rest * use_tail + 1                      # + EOT
    for m in p[msg_idx + 1:]:
        n += len(m["content"]) * (tail if (tail and m is not p[1]) else ov) + 3 + (2 if m["role"] == "assistant" else 0)
    return n + 3                                  # generation prompt


SEG = {0: "system", 1: "request", 2: "lookup_call", 3: "lookup_result"}
out = {}
per_pair = defaultdict(dict)
for lv in ("L0r", "L1", "L2", "L3"):
    c = Counter()
    dists, ndiff_frac = [], []
    beyond = []
    for (sid, l), i in dec.items():
        if l != lv or not sid.endswith("A") or by_id[sid]["twin"] != "A_should_ask":
            continue
        j = dec.get((sid[:-1] + "B", lv))
        if j is None:
            continue
        pa, pb = strip(sft[i]["prompt"]), strip(sft[j]["prompt"])
        c["pairs"] += 1
        if len(pa) != len(pb):
            c["different_number_of_messages"] += 1
        diff_msgs = [k for k in range(min(len(pa), len(pb))) if pa[k]["content"] != pb[k]["content"]]
        segs = tuple(SEG.get(k, f"msg{k}") for k in diff_msgs)
        c["differs_in:" + "+".join(segs)] += 1
        c["system_differs"] += 0 in diff_msgs
        if not diff_msgs:
            c["identical_after_hint_strip"] += 1
            continue
        k0 = diff_msgs[0]
        a, b = pa[k0]["content"], pb[k0]["content"]
        ci = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), min(len(a), len(b)))
        d_a = tokens_from_end(i, k0, ci)
        dists.append(d_a)
        if d_a > 4096:
            beyond.append([sid[:-1], round(d_a), SEG.get(k0)])
        per_pair[lv][sid[:-1]] = {"segments": segs, "earliest_diff_tokens_from_end(A)": round(d_a)}
    q = np.percentile(dists, [10, 50, 90]).round().tolist() if dists else None
    out[lv] = {"counts": dict(c), "earliest_diff_tokens_from_end_p10_p50_p90": q, "pairs_differing_beyond_4096": beyond}

# --- A-twin L2 targets: identifiers/numbers only available in the system prompt ------------------------------------------
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
IDR = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w.@])@[\w.-]*\w|\b[A-Z]{2,}-?\d[\w-]*\b|\*{2,}\d{3,}|\b\d{4}-\d{2}-\d{2}\b")


def norm_num(s):
    s = s.replace(",", "")
    try:
        return f"{float(s):.2f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


cnt = Counter()
ex_only_sys, ex_no_sys = [], []
for (sid, l), i in dec.items():
    if l != "L2" or by_id[sid]["twin"] != "A_should_ask" or by_id[sid]["ideal_action"]["type"] != "message":
        continue
    p = sft[i]["prompt"]
    t = target_text(sft[i])
    sys_txt = p[0]["content"].replace(hint, "")
    tail = "\n".join(m["content"] for m in p[1:])
    items = {x for x in IDR.findall(t)}
    nums = {norm_num(x) for x in NUM.findall(t) if len(x.replace(",", "").replace(".", "")) >= 2}
    sys_nums = {norm_num(x) for x in NUM.findall(sys_txt)}
    tail_nums = {norm_num(x) for x in NUM.findall(tail)}
    only_sys = sorted([x for x in items if x.lower() in sys_txt.lower() and x.lower() not in tail.lower()]
                      + [x for x in nums if x in sys_nums and x not in tail_nums])
    cnt["A_L2_message_targets"] += 1
    cnt["cites_something_only_in_system_prompt"] += bool(only_sys)
    if only_sys and len(ex_only_sys) < 5:
        ex_only_sys.append([i, sid, only_sys[:4], t[:140]])
    if not only_sys and len(ex_no_sys) < 5:
        ex_no_sys.append([i, sid, t[:140]])

dump("v05_twin_diff_positions.json", {"by_level": out, "per_pair": per_pair,
                                      "A_L2_targets_needing_system_prompt": {"counts": dict(cnt), "examples_only_in_system": ex_only_sys,
                                                                             "examples_all_in_tail": ex_no_sys}})
for lv, v in out.items():
    print(lv, v["counts"], v["earliest_diff_tokens_from_end_p10_p50_p90"], v["pairs_differing_beyond_4096"])
print(dict(cnt))
