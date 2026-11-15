"""DPO-pairs audit, part 2: what each pair contrasts, contradictions, twin mirrors, hint conflicts, teaching balance.

Reads results/pairs.jsonl (pairs_audit.py) + the DPO / SFT files (read-only). Writes results/contrasts.json, results/contrast_rows.csv.

Contrast type per pair (chosen action vs rejected action, action = message | ui (approval card) | commit | lookup | invalid):
  * the divergence point: the first character where chosen and rejected differ (DPO's log-ratio only depends on tokens from the
    first differing token on: the shared prefix cancels), classified as start / tool name / arguments / message text
  * tool-call-like rejected texts that the parser reads as chat messages (malformed '<tool_call {' etc.): their decision class
    (statement / ask_generic) is a parser artifact
  * near-duplicate pairs: chosen and rejected name the same tool and differ only in formatting (e.g. one extra brace)
Contradiction checks: a text chosen in one pair and rejected in another (exact, whitespace-normalised); identical rejected text in
several pairs; A/B twins at the same level (prompt line diff, mirror texts); hint (P2) vs chosen; rejected sides the evaluation
grader accepts.

    python3 experiments/dataset_audit/dpo-pairs/contrasts.py
"""
from __future__ import annotations

import collections
import difflib
import json
import os
import re
import sys
import unicodedata

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.render import golden_text  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = os.path.join(ROOT, "dataset", "curated_v121")
HINT = golden_text("P2")


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def nrm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


pairs = load(os.path.join(OUT, "pairs.jsonl"))
dpo = load(os.path.join(D, "dpo_train.jsonl"))
sft = load(os.path.join(D, "sft_train.jsonl"))
scn_by_id, KINDS = {}, {}
for r in load(os.path.join(D, "grpo_train.jsonl")):
    s = json.loads(r["scenario"])
    scn_by_id[s["id"]] = s
    KINDS[(r["scenario_id"], r["level"])] = json.loads(r["tool_kinds"])

NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][\w]*)"')


def action(p, side):
    k = p[f"{side}_kind"]
    if k == "tool_call":
        tk = p["chosen_tool"] if side == "chosen" else p["rejected_tool_kind"]
        if side == "chosen":
            tk = "ui" if p["chosen_tool"] == "request_approval" else ("lookup" if p["chosen_decision"] == "lookup" else "commit")
        return {"ui": "ui", "commit": "commit", "lookup": "lookup", "verify": "lookup"}.get(tk, "invalid")
    return k   # message | invalid


def tool_like_message(text: str) -> bool:
    return bool(re.search(r"<tool_call|\*\*Tool Call|\"arguments\"", text))


def divergence(ch: str, rj: str, cp: int, ch_kind: str) -> str:
    if cp == 0:
        return "first character (different action type or leading whitespace)"
    pre = ch[:cp]
    if ch_kind == "message":
        return "inside message text"
    m = re.match(r'<tool_call>\{"name": "([^"]*)$', pre)
    if m is not None or pre in ("<", "<tool_call>", "<tool_call>{", '<tool_call>{"', '<tool_call>{"name": "') or len(pre) < 22:
        return "at or before the tool name"
    if '"arguments"' not in pre:
        return "at or before the tool name"
    return "inside arguments (same tool)"


rows = []
for p in pairs:
    ch, rj = p["chosen_text"], p["rejected_text"]
    ca, ra = action(p, "chosen"), action(p, "rejected")
    rname = NAME_RE.search(rj)
    if p["rejected_kind"] == "message" and tool_like_message(rj):
        mk = KINDS[(p["scenario_id"], p["level"])].get(rname.group(1) if rname else "", "")
        ra_true = "malformed " + {"ui": "ui", "commit": "commit", "lookup": "lookup", "verify": "lookup"}.get(mk, "unknown-tool") \
            + " call read as message"
    else:
        ra_true = ra
    same_tool_text = bool(rname) and p["chosen_kind"] == "tool_call" and rname.group(1) == p["chosen_tool"]
    sim = difflib.SequenceMatcher(None, ch, rj[:2000], autojunk=False).ratio()
    rows.append({
        "idx": p["idx"], "scenario_id": p["scenario_id"], "pair_id": p["pair_id"], "twin": p["twin"], "level": p["level"],
        "prompt_setting": p["prompt_setting"], "env": p["env"], "expected": p["expected"],
        "chosen_action": ca, "rejected_action": ra, "rejected_action_true": ra_true,
        "chosen_decision": p["chosen_decision"], "rejected_decision": p["rejected_decision"],
        "contrast": f"{ca} > {ra_true}", "decision_class_differs": p["chosen_decision"] != p["rejected_decision"],
        "action_differs": ca != ra_true,
        "same_tool_name_in_text": same_tool_text, "char_similarity": round(sim, 3),
        "divergence": divergence(ch, rj, p["common_prefix_chars"], p["chosen_kind"]),
        "common_prefix_chars": p["common_prefix_chars"],
        "rejected_is_question": p["rejected_is_question"],
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "contrast_rows.csv"), index=False)


def ct(a, b, d=None):
    t = pd.crosstab((df if d is None else d)[a], (df if d is None else d)[b], margins=True)
    return {str(i): {str(c): int(v) for c, v in r.items()} for i, r in t.iterrows()}


def vc(c, d=None):
    return {str(k): int(v) for k, v in (df if d is None else d)[c].value_counts().items()}


def ex(mask, n=3, cols=("idx", "scenario_id", "level", "prompt_setting")):
    return df[mask].head(n)[list(cols)].to_dict("records")


# ---- 1/4: contrast types, wording vs decision ---------------------------------------------------------------------------
near_dup = df[(df.char_similarity >= 0.9)]
contrast = {
    "contrast_by_twin": ct("contrast", "twin"),
    "divergence_by_twin": ct("divergence", "twin"),
    "action_differs": vc("action_differs"),
    "decision_class_differs": vc("decision_class_differs"),
    "same_action_same_tool_or_both_messages": int((~df.action_differs).sum()),
    "malformed_tool_calls_read_as_messages": ex(df.rejected_action_true.str.startswith("malformed"), 20,
                                                 ("idx", "scenario_id", "level", "rejected_decision", "rejected_action_true")),
    "near_duplicates_char_similarity_ge_0.9": near_dup[["idx", "scenario_id", "level", "prompt_setting", "chosen_decision",
                                                        "rejected_decision", "char_similarity"]].to_dict("records"),
    "char_similarity_quantiles": {str(q): round(float(df.char_similarity.quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
}

# teaching axes
def axis(r):
    ca, ra = r["chosen_action"], r["rejected_action_true"]
    if ca == "commit" and ra in ("commit", "malformed commit call read as message"):
        return "commit > wrong or malformed commit"
    if ra == "commit" or ra == "malformed commit call read as message":
        return "X > direct commit (incl. malformed)"
    if ra.startswith("malformed"):
        return f"{ca} > {ra}"
    if ca == "message" and ra == "ui":
        return "ask > approval card"
    if ca in ("ui", "commit") and ra == "message":
        return "proceed > " + ("question" if r["rejected_is_question"] else "non-question message")
    if ca == "message" and ra == "message":
        return "informed question > " + ("generic question" if r["rejected_is_question"] else "statement")
    return f"{ca} > {ra}"


df["teaching_axis"] = df.apply(axis, axis=1)
teaching = {
    "axis_by_twin": ct("teaching_axis", "twin"),
    "axis_by_level": ct("teaching_axis", "level"),
    "axis_by_prompt_setting": ct("teaching_axis", "prompt_setting"),
    "ask_vs_dontask_by_level": ct("teaching_axis", "level", df[df.teaching_axis.isin(
        ["ask > approval card", "proceed > question", "informed question > generic question", "informed question > statement"])]),
    "examples": {a: ex(df.teaching_axis == a, 3) for a in sorted(df.teaching_axis.unique())},
}

# ---- 5: contradictions ---------------------------------------------------------------------------------------------------
chosen_texts = collections.defaultdict(list)
rej_texts = collections.defaultdict(list)
for p in pairs:
    chosen_texts[nrm(p["chosen_text"])].append(p["idx"])
    rej_texts[nrm(p["rejected_text"])].append(p["idx"])
both = sorted(set(chosen_texts) & set(rej_texts))
cross = [{"text": t[:160], "chosen_in": chosen_texts[t], "rejected_in": rej_texts[t]} for t in both]
dup_rej = {t: ix for t, ix in rej_texts.items() if len(ix) > 1}
dup_rows = []
for t, ix in sorted(dup_rej.items(), key=lambda kv: -len(kv[1])):
    sub = [pairs[i] for i in ix]
    dup_rows.append({"n": len(ix), "text": t[:140], "idx": ix, "scenarios": sorted({s["scenario_id"] for s in sub}),
                     "levels": sorted({s["level"] for s in sub}), "twins": sorted({s["twin"] for s in sub}),
                     "sources": sorted({s["rejected_source"] for s in sub})})
# identical rejected for A and B twins of one pair at one level
by_pl = collections.defaultdict(dict)
for p in pairs:
    by_pl[(p["pair_id"], p["level"])][p["twin"]] = p
twin_same_rej = [{"pair_id": k[0], "level": k[1], "A_idx": v["A"]["idx"], "B_idx": v["B"]["idx"],
                  "text": v["A"]["rejected_text"][:120]}
                 for k, v in sorted(by_pl.items()) if "A" in v and "B" in v
                 and nrm(v["A"]["rejected_text"]) == nrm(v["B"]["rejected_text"])]


# A/B twin prompts at the same level: line diff (P2 hint removed)
def lines(prompt):
    out = []
    for m in prompt:
        c = m["content"].replace("\n\n" + HINT, "") if m["role"] == "system" else m["content"]
        out += [f"{m['role']}|{x.strip()}" for x in c.splitlines() if x.strip()]
    return out


twin_rows = []
for (pid, lv), v in sorted(by_pl.items()):
    if "A" not in v or "B" not in v:
        continue
    a, b = v["A"], v["B"]
    la, lb = lines(dpo[a["idx"]]["prompt"]), lines(dpo[b["idx"]]["prompt"])
    ca, cb = collections.Counter(la), collections.Counter(lb)
    only_a, only_b = sum((ca - cb).values()), sum((cb - ca).values())
    chars_a = sum(len(x) for x in (ca - cb).elements())
    total_chars = sum(len(m["content"]) for m in dpo[a["idx"]]["prompt"])
    a_rej_act = df.loc[a["idx"], "rejected_action_true"]
    b_ch_act = df.loc[b["idx"], "chosen_action"]
    twin_rows.append({"pair_id": pid, "level": lv, "A_idx": a["idx"], "B_idx": b["idx"],
                      "A_setting": a["prompt_setting"], "B_setting": b["prompt_setting"],
                      "prompt_lines_only_in_A": only_a, "prompt_lines_only_in_B": only_b, "A_lines": len(la),
                      "A_diff_chars": chars_a, "A_prompt_chars": total_chars,
                      "A_rejected_action": a_rej_act, "B_chosen_action": b_ch_act,
                      "mirror_action": a_rej_act == b_ch_act,
                      "A_rejected_vs_B_chosen_sim": round(difflib.SequenceMatcher(None, a["rejected_text"][:2000],
                                                                                  b["chosen_text"]).ratio(), 3),
                      "A_chosen_vs_B_rejected_sim": round(difflib.SequenceMatcher(None, a["chosen_text"],
                                                                                  b["rejected_text"][:2000]).ratio(), 3)})
tw = pd.DataFrame(twin_rows)
tw.to_csv(os.path.join(OUT, "twin_contrast.csv"), index=False)
pairs_by_pl = collections.Counter((p["pair_id"], p["level"]) for p in pairs)
unpaired = [{"pair_id": k[0], "level": k[1], "twin": list(v)[0], "idx": list(v.values())[0]["idx"]}
            for k, v in sorted(by_pl.items()) if len(v) == 1]

# hint vs chosen: P2 says "if everything matches, just show the approval card"; B 'act' targets commit directly
hint_conflict_dpo = [{"idx": p["idx"], "scenario_id": p["scenario_id"], "level": p["level"], "chosen": p["chosen_text"][:90]}
                     for p in pairs if p["prompt_setting"] == "P2" and p["expected"] == "act"]
hint_conflict_sft = []
for i, r in enumerate(sft):
    s = scn_by_id.get(r["scenario_id"])
    if r["turn"] == "decision" and r["prompt_setting"] == "P2" and s and s["expected"]["decision"] == "act":
        hint_conflict_sft.append({"i": i, "scenario_id": r["scenario_id"], "level": r["level"]})
act_total_sft = sum(1 for r in sft if r["turn"] == "decision" and scn_by_id.get(r["scenario_id"], {}).get("expected", {})
                    .get("decision") == "act")

contradictions = {
    "text_chosen_in_one_pair_and_rejected_in_another": cross,
    "duplicate_rejected_texts": {"n_texts": len(dup_rows), "n_pairs": sum(d["n"] for d in dup_rows), "top": dup_rows[:12]},
    "identical_rejected_for_A_and_B_twin_same_level": {"n": len(twin_same_rej), "rows": twin_same_rej[:12]},
    "twin_pairs_same_level": {
        "n_matched_pair_levels": len(tw), "unpaired_rows": unpaired,
        "different_prompt_setting_between_twins": int((tw.A_setting != tw.B_setting).sum()),
        "prompt_lines_only_in_A": {str(q): float(tw.prompt_lines_only_in_A.quantile(q)) for q in (0, 0.25, 0.5, 0.75, 1)},
        "A_diff_chars_share_of_prompt": {str(q): round(float((tw.A_diff_chars / tw.A_prompt_chars).quantile(q)), 4)
                                         for q in (0, 0.25, 0.5, 0.75, 1)},
        "mirror_A_rejected_action_eq_B_chosen_action": vc("mirror_action", tw),
        "mirror_examples": tw[tw.mirror_action][["pair_id", "level", "A_idx", "B_idx", "A_rejected_vs_B_chosen_sim",
                                                  "prompt_lines_only_in_A"]].head(5).to_dict("records"),
        "A_rejected_vs_B_chosen_sim_median": float(tw.A_rejected_vs_B_chosen_sim.median()),
    },
    "p2_hint_vs_act_target": {"dpo_pairs": hint_conflict_dpo, "n_dpo": len(hint_conflict_dpo),
                              "n_dpo_act_total": int((df.expected == "act").sum()),
                              "n_sft_decision_rows": len(hint_conflict_sft), "n_sft_act_decision_rows": act_total_sft,
                              "sft_examples": hint_conflict_sft[:4]},
    "rejected_accepted_by_eval_grader_lenient": [{"idx": p["idx"], "scenario_id": p["scenario_id"], "level": p["level"],
                                                  "accept": p["adjudicated_accept"], "rejected_decision": p["rejected_decision"],
                                                  "rejected": p["rejected_text"][:110]}
                                                 for p in pairs if p["rejected_eval_lenient"]],
    "B_rejected_informed": [{"idx": p["idx"], "scenario_id": p["scenario_id"], "level": p["level"],
                             "rejected_decision": p["rejected_decision"], "rejected": p["rejected_text"][:200]}
                            for p in pairs if p["twin"] == "B" and p["rejected_decision"] in ("ask_informed", "tell_informed")],
    "A_rejected_commit_gold": {"n": int(((df.twin == "A") & (df.rejected_decision == "commit_gold")).sum()),
                               "examples": ex((df.twin == "A") & (df.rejected_decision == "commit_gold"), 3)},
}
res = {"contrast": contrast, "teaching": teaching, "contradictions": contradictions}
json.dump(res, open(os.path.join(OUT, "contrasts.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({"contrast_by_twin": contrast["contrast_by_twin"], "axis_by_twin": teaching["axis_by_twin"]}, indent=1))
print("CONTRASTS_OK")
