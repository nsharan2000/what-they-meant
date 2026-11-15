"""V06 - PC7 (policy absent at L0r; policy outside the 4,096-token sliding window at L2/L3) and PC10 (L0r seed member cards naming
tools absent from the row's tool docs).
Token positions: the exact prompt length per row comes from experiments/dataset_audit/results/token_stats.jsonl (Tiny Aya tokenizer
+ chat template). The policy's own token count is not stored, so I estimate it two ways and report the count of rows whose policy
ends > 4,096 tokens before the generation point under a grid of assumptions:
  (1) OLS per-segment fit over all 2,120 rows: tokens ~ 1 + policy_chars + member_chars[L0r] + member_chars[L1] + member_chars[L2/L3]
      + tools_chars + post_chars (messages after the system prompt) + n_msgs + P2  (coefficients = tokens per char of each segment)
  (2) bounds: policy chars / {3.6, 4.0, 4.4} chars per token, template preamble {450, 550, 650} tokens.
The chat template (third_party/AuditKIT/tests/fixtures/chat_templates/tiny_aya_default.jinja) puts a fixed System/Default preamble
before our system prompt, so policy end = preamble + policy tokens.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/06_pc7_pc10.py -> results/06_pc7_pc10.json
"""
import collections
import json
import os
import re

import numpy as np

from vcommon import OUT, ROOT, dump, envs, listed_tools, rows, segments

T = [json.loads(l) for l in open(os.path.join(OUT, "table.jsonl"))]
E = envs()
res = {}
# ---------------- PC7: policy lengths ----------------
words = lambda s: len(re.sub(r"\{\{[a-z_]+\}\}", "", s).split())
res["policy_words"] = {e: {"L0_role_line": words(E[e]["system_prompt"]["L0"]), "L1_policy": words(E[e]["system_prompt"]["L1"].split("{{member_block}}")[0])}
                       for e in E}
sd = [r for r in T if r["file"] == "sft_train" and r["turn"] == "decision"]
res["sft_decision_by_level"] = dict(collections.Counter(r["level"] for r in sd))
res["sft_decision_L0r_P0"] = sum(r["level"] == "L0r" and r["ps"] == "P0" for r in sd)
res["sft_decision_L0r_P0_by_twin"] = dict(collections.Counter(r["t"] for r in sd if r["level"] == "L0r" and r["ps"] == "P0"))

# ---------------- PC7: regression ----------------
X, y = [], []
for r in T:
    if r["prompt_tokens"] is None:
        continue
    lv = r["level"]
    X.append([1, r["policy_chars"], r["member_chars"] * (lv == "L0r"), r["member_chars"] * (lv == "L1"),
              r["member_chars"] * (lv in ("L2", "L3")), r["tools_chars"], r["post_chars"], r["n_msgs"], int(r["ps"] == "P2")])
    y.append(r["prompt_tokens"])
X, y = np.array(X, float), np.array(y, float)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
resid = y - pred
names = ["const", "policy_chars", "member_L0r", "member_L1", "member_L2L3", "tools_chars", "post_chars", "n_msgs", "P2"]
res["fit"] = {"coef": dict(zip(names, [round(float(b), 5) for b in beta])), "r2": round(float(1 - resid.var() / y.var()), 5),
              "resid_abs_p95": round(float(np.percentile(np.abs(resid), 95)), 1), "n": len(y),
              "chars_per_token_policy": round(1 / beta[1], 2), "hint_chars": len(open(os.path.join(ROOT, "prompts/golden_short.md")).read().strip())}
tmpl = open(os.path.join(ROOT, "third_party/AuditKIT/tests/fixtures/chat_templates/tiny_aya_default.jinja")).read()
pre = tmpl.split("{{ ns.system_prompt }}")[0]
pre_text = re.sub(r"\{%.*?%\}|\{\{.*?\}\}", "", pre, flags=re.S)
res["template_preamble_chars"] = len(pre_text)

late = [r for r in T if r["file"] == "sft_train" and r["turn"] == "decision" and r["level"] in ("L2", "L3")]
grid = {}
for ratio in (3.6, 4.0, 4.4):
    for pre_tok in (450, 550, 650):
        n = collections.Counter()
        for r in late:
            pol_end = pre_tok + r["policy_chars"] / ratio
            if r["prompt_tokens"] - pol_end > 4096:
                n[r["level"]] += 1
        grid[f"ratio{ratio}_pre{pre_tok}"] = dict(n)
res["late_rows"] = dict(collections.Counter(r["level"] for r in late))
res["policy_beyond_window_grid"] = grid
fitpol = collections.Counter()
for r in late:
    pol_end = 550 + beta[1] * r["policy_chars"]
    fitpol[r["level"]] += int((r["prompt_tokens"] - pol_end) > 4096)
res["policy_beyond_window_fit_policy_coef_pre550"] = dict(fitpol)
res["policy_tokens_fit_by_env"] = {e: round(float(beta[1] * len(E[e]["system_prompt"]["L1"].split("{{member_block}}")[0])), 0) for e in E}
# hint distance: tokens after the hint = messages after the system prompt (+ template tokens)
hint_rows = [r for r in T if r["file"] == "sft_train" and r["turn"] == "decision" and r["ps"] == "P2"]
d = [beta[6] * r["post_chars"] + beta[7] * r["n_msgs"] for r in hint_rows]
res["hint_to_generation_tokens_est"] = {"median": round(float(np.median(d)), 0), "p95": round(float(np.percentile(d, 95)), 0), "n": len(d)}
# prompt length distribution at L2/L3
res["late_prompt_tokens"] = {lv: {"min": min(r["prompt_tokens"] for r in late if r["level"] == lv),
                                  "p50": int(np.median([r["prompt_tokens"] for r in late if r["level"] == lv])),
                                  "max": max(r["prompt_tokens"] for r in late if r["level"] == lv)} for lv in ("L2", "L3")}
ex = sorted(late, key=lambda r: -(r["prompt_tokens"] - 550 - beta[1] * r["policy_chars"]))[:3]
res["examples_far"] = [{"line": r["i"], "sid": r["sid"], "level": r["level"], "ps": r["ps"], "prompt_tokens": r["prompt_tokens"],
                        "policy_end_to_generation_est": int(r["prompt_tokens"] - 550 - beta[1] * r["policy_chars"])} for r in ex]

# authorisation visibility at L0r for 'act' rows (S14B/S15B/V152B) is in 03_pc2_pc3.json

# ---------------- PC10 ----------------
bad = []
for f in ("sft_train", "dpo_train", "grpo_train", "sft_val"):
    for i, x in enumerate(rows(f)):
        if x["level"] != "L0r":
            continue
        seg = segments(x["prompt"][0]["content"], E[x["env_id"]], "L0r")
        m = re.search(r"Tools:\s*([^\n]*)", seg["member"])
        if not m:
            continue
        named = [re.sub(r"\(.*?\)", "", t).strip(" .") for t in m.group(1).split(",")]
        listed = set(listed_tools(seg["tools"]))
        env_tools = {t["name"] for t in E[x["env_id"]]["tools"]}
        miss = [n for n in named if n and n not in listed]
        if miss:
            bad.append({"file": f, "i": i, "sid": x["scenario_id"], "turn": x.get("turn", "decision"), "ps": x["prompt_setting"],
                        "missing": miss, "missing_not_env_tool": [n for n in miss if n not in env_tools],
                        "note": m.group(1)[:120]})
res["pc10"] = {
    "rows_by_file_turn": dict(collections.Counter(f"{b['file']}:{b['turn']}" for b in bad)),
    "decision_rows_sft_plus_dpo": sum(b["turn"] == "decision" and b["file"] in ("sft_train", "dpo_train") for b in bad),
    "names_in_decision_rows_sft_plus_dpo": dict(collections.Counter(n for b in bad if b["turn"] == "decision" and b["file"] in ("sft_train", "dpo_train") for n in b["missing"])),
    "not_env_tools_at_all": sorted({n for b in bad for n in b["missing_not_env_tool"]}),
    "scenarios": sorted({b["sid"] for b in bad}), "examples": [b for b in bad if b["file"] == "sft_train"][:6]}
# all L0r 'Tools:' notes (whether or not they name missing tools) and other instruction-like seed notes ("Policy:", "Type 2 agent")
notes = collections.Counter()
for x in rows("sft_train"):
    if x["level"] == "L0r" and x.get("turn") == "decision":
        mb = segments(x["prompt"][0]["content"], E[x["env_id"]], "L0r")["member"]
        for k in ("Tools:", "Policy:", "Type 2 agent", "Type 1"):
            notes[k] += k in mb
res["L0r_seed_card_instruction_like_notes_sft_decision"] = dict(notes)
p = dump("06_pc7_pc10.json", res)
print(json.dumps(res, indent=1, ensure_ascii=False)[:7000])
print("wrote", p)
