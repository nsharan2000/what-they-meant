"""DPO-pairs audit (lens: dpo-pairs), part 1: one row per pair + summary tables. READ-ONLY on every repo file.

For every pair of dataset/curated_v121/dpo_train.jsonl (0-based line index = `idx`):
  * decision class + reward of chosen and rejected with wtm.reward (exactly what training/build_datasets.py used to accept a
    rejected side), plus the evaluation grader's verdict (wtm.episode.decision_grade on the ADJUDICATED scenario, as wtm.run does)
  * where the rejected side came from: the build's own rule is replayed (training/build_datasets.py load_failures + reward <= 0,
    files in sorted order, first usable failure per (scenario, level)) while keeping provenance (grid file, line, model, prompt
    setting, earlier turns of that episode); synthetic rejected sides are recomputed with the build's synthetic_rejected()
  * format / degeneracy flags of the rejected text (wtm.protocol.parse_action flags, whitespace, repetition, zlib ratio)
  * exact Tiny Aya token counts from experiments/dataset_audit/results/token_stats.jsonl (index-aligned, alignment verified)
  * character common prefix of chosen vs rejected (DPO gradients only see tokens after the shared prefix)
  * overlap with SFT (same prompt / same target) and with evaluation scenario files
Outputs (experiments/dataset_audit/dpo-pairs/results/): pairs.jsonl, pairs.csv, summary_pairs.json

    python3 experiments/dataset_audit/dpo-pairs/pairs_audit.py
"""
from __future__ import annotations

import collections
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
import unicodedata
import zlib

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.episode import decision_grade, is_question  # noqa: E402
from wtm.labels import adjudicate  # noqa: E402
from wtm.protocol import parse_action  # noqa: E402
from wtm.render import golden_text  # noqa: E402
from wtm.reward import decision_of, reward  # noqa: E402

spec = importlib.util.spec_from_file_location("build_datasets", os.path.join(ROOT, "training", "build_datasets.py"))
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)   # main() is guarded; we reuse synthetic_rejected, BASE_MODELS, load_failures, norm, prompt_key

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
D = os.path.join(ROOT, "dataset", "curated_v121")
GRID = sorted(glob.glob(os.path.join(ROOT, "experiments", "replication", "results", "grid_*.jsonl")))
HINT = golden_text("P2")


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def nrm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


dpo = load(os.path.join(D, "dpo_train.jsonl"))
grpo = load(os.path.join(D, "grpo_train.jsonl"))
sft = load(os.path.join(D, "sft_train.jsonl"))
sft_val = load(os.path.join(D, "sft_val.jsonl"))
grpo_by = {(r["scenario_id"], r["level"]): r for r in grpo}
assert len(grpo_by) == len(grpo) == len(dpo)

# --- provenance: same pairs hash as training/train_dpo.py computes (to compare with runs/m3_dpo/args.json later) ----------
pairs_sha = hashlib.sha256(json.dumps([[r["prompt"], r["chosen"], r["rejected"]] for r in dpo], sort_keys=True).encode()).hexdigest()
file_sha = hashlib.sha256(open(os.path.join(D, "dpo_train.jsonl"), "rb").read()).hexdigest()

# --- token stats (index-aligned) ------------------------------------------------------------------------------------------
tok = {}
for r in load(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl")):
    if r["file"] == "dpo_train.jsonl":
        tok[r["i"]] = r
tok_misaligned = [i for i, r in enumerate(dpo) if (tok[i]["scenario_id"], tok[i]["level"], tok[i]["prompt_setting"])
                  != (r["scenario_id"], r["level"], r["prompt_setting"])]

# --- replay of the build's failure selection, with provenance ------------------------------------------------------------
fails = collections.defaultdict(list)   # (id, level) -> [(raw, provenance)] in the build's order
for f in GRID:
    for ln, line in enumerate(open(f)):
        r = json.loads(line)
        if not r["decision_grade"]["success"] and r.get("events") and r["model"].startswith(bd.BASE_MODELS):
            last = r["events"][-1].get("raw")
            if last:
                prior = r["events"][:-1]
                fails[(r["id"], r["level"])].append((last, {
                    "grid_file": os.path.basename(f), "grid_line": ln, "model": r["model"], "grid_prompt": r.get("prompt"),
                    "rep": r.get("rep"), "grid_decision": r["decision_grade"]["decision"], "grid_n_events": len(r["events"]),
                    "grid_system_chars": r.get("system_chars"),
                    "grid_prior_calls": [[e.get("tool"), e.get("args")] for e in prior if e.get("kind") == "tool_call"],
                    "grid_prior_kinds": [e.get("kind") + (":narr" if e.get("narration") else "") for e in prior],
                    "grid_prior_errors": sum(1 for e in prior if isinstance(e.get("result"), dict) and "error" in e["result"]),
                    "grid_last_kind": r["events"][-1].get("kind")}))
# sanity: same lists as the build's own loader
orig = bd.load_failures([os.path.join(ROOT, "experiments", "replication", "results", "grid_*.jsonl")])
assert {k: [x for x, _ in v] for k, v in fails.items()} == {k: v for k, v in orig.items()}, "failure replay differs from build"

# --- evaluation scenario files (leak check) ---------------------------------------------------------------------------------
eval_pairs = collections.defaultdict(set)
for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "eval", "*.jsonl"))):
    for r in load(f):
        eval_pairs[os.path.basename(f).rsplit("_", 1)[0]].add(r.get("pair_id"))
splits = json.load(open(os.path.join(ROOT, "dataset", "splits.json")))

# --- SFT index ----------------------------------------------------------------------------------------------------------------
sft_dec = collections.defaultdict(list)
for i, r in enumerate(sft):
    if r["turn"] == "decision":
        sft_dec[(r["scenario_id"], r["level"])].append(i)
sft_prompt_keys = collections.defaultdict(list)
for i, r in enumerate(sft):
    sft_prompt_keys[bd.prompt_key(r["prompt"], HINT)].append(i)
sft_targets = collections.defaultdict(list)
for i, r in enumerate(sft):
    sft_targets[nrm(r["completion"][0]["content"])].append((i, r["turn"], r["scenario_id"]))
val_prompt_keys = {bd.prompt_key(r["prompt"], HINT) for r in sft_val}


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def action_info(text: str, kinds: dict) -> dict:
    act = parse_action(text, set(kinds))
    return {"kind": act.kind, "tool": act.name if act.kind == "tool_call" else "", "tool_kind": kinds.get(act.name, "")
            if act.kind == "tool_call" else "", "strict": act.strict_ok, "flags": act.flags}


def repetition(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cnt = collections.Counter(lines)
    words = text.split()
    tri = collections.Counter(zip(words, words[1:], words[2:]))
    b = text.encode()
    return {"max_line_repeat": max(cnt.values()) if cnt else 0,
            "trigram_repeat_share": round(sum(v - 1 for v in tri.values() if v > 1) / max(1, len(words) - 2), 3),
            "zlib_ratio": round(len(zlib.compress(b)) / max(1, len(b)), 3)}


rows = []
for idx, r in enumerate(dpo):
    g = grpo_by[(r["scenario_id"], r["level"])]
    scn = json.loads(g["scenario"])
    kinds = json.loads(g["tool_kinds"])
    adj = adjudicate(scn)
    ch, rj = r["chosen"][0]["content"], r["rejected"][0]["content"]
    rc, dc = reward(ch, scn, kinds)
    rr, dr = reward(rj, scn, kinds)
    ai_c, ai_r = action_info(ch, kinds), action_info(rj, kinds)
    # source of the rejected side (replay of the build rule)
    observed = fails.get((r["scenario_id"], r["level"]), [])
    pick = next(((t, p) for t, p in observed if reward(t, scn, kinds)[0] <= 0), None)
    if pick is not None:
        src = "observed_failure"
        prov = pick[1]
        replay_match = pick[0] == rj
    else:
        src = "synthetic_failures_scored_above_0" if observed else "synthetic_no_failure_same_level"
        prov = {}
        replay_match = bd.synthetic_rejected(scn) == rj
    canon = [[c["tool"], c["arguments"]] for c in scn.get("prefix_calls") or []]
    sys_len = len(r["prompt"][0]["content"])
    sys_len_nohint = len(r["prompt"][0]["content"].replace("\n\n" + HINT, "")) if r["prompt_setting"] == "P2" else sys_len
    t = tok[idx]
    cp = common_prefix(ch, rj)
    sft_rows = sft_dec.get((r["scenario_id"], r["level"]), [])
    pk = bd.prompt_key(r["prompt"], HINT)
    row = {
        "idx": idx, "scenario_id": r["scenario_id"], "pair_id": r["pair_id"], "twin": r["twin"][:1], "env": r["env_id"],
        "level": r["level"], "prompt_setting": r["prompt_setting"], "expected": scn["expected"]["decision"],
        "adjudicated_accept": ",".join(sorted(adj["expected"].get("accept") or [])),
        "seed_or_variant": "seed" if r["scenario_id"].startswith("S") else "variant",
        # chosen
        "chosen_text": ch, "chosen_decision": dc["decision"], "chosen_reward": rc, "chosen_kind": ai_c["kind"],
        "chosen_tool": ai_c["tool"], "chosen_strict": ai_c["strict"],
        "chosen_eval_strict": decision_grade(adj, dc["decision"])["success"],
        "chosen_eval_lenient": decision_grade(adj, dc["decision"])["success_lenient"],
        # rejected
        "rejected_text": rj, "rejected_decision": dr["decision"], "rejected_reward": rr, "rejected_kind": ai_r["kind"],
        "rejected_tool": ai_r["tool"], "rejected_tool_kind": ai_r["tool_kind"], "rejected_strict": ai_r["strict"],
        "rejected_flags": "|".join(ai_r["flags"]),
        "rejected_eval_strict": decision_grade(adj, dr["decision"])["success"],
        "rejected_eval_lenient": decision_grade(adj, dr["decision"])["success_lenient"],
        "rejected_is_question": is_question(rj) if ai_r["kind"] == "message" else False,
        "rejected_lead_ws": rj[:1].isspace(), "rejected_trail_ws": rj[-1:].isspace(),
        "chosen_lead_ws": ch[:1].isspace(), "chosen_trail_ws": ch[-1:].isspace(),
        "rejected_newlines": rj.count("\n"), "chosen_newlines": ch.count("\n"),
        "rejected_chars": len(rj), "chosen_chars": len(ch), "rejected_words": len(rj.split()), "chosen_words": len(ch.split()),
        **{"rejected_" + k: v for k, v in repetition(rj).items()},
        "rejected_nonascii_share": round(sum(ord(c) > 127 for c in rj) / max(1, len(rj)), 3),
        "rejected_has_think": "<think>" in rj, "rejected_has_action_md": "Action:" in rj or "```" in rj,
        # tokens (exact, token_stats.jsonl)
        "chosen_tokens": t["chosen_tokens"], "rejected_tokens": t["rejected_tokens"], "prompt_tokens": t["prompt_tokens"],
        "chosen_first5": " ".join(t["chosen_first5"]), "rejected_first5": " ".join(t["rejected_first5"]),
        "first_token_differs": t["chosen_first5"][:1] != t["rejected_first5"][:1],
        # shared prefix
        "common_prefix_chars": cp, "common_prefix_text": ch[:cp],
        # source
        "rejected_source": src, "replay_matches_file": replay_match, **{k: prov.get(k) for k in (
            "grid_file", "grid_line", "model", "grid_prompt", "rep", "grid_decision", "grid_n_events", "grid_system_chars",
            "grid_last_kind", "grid_prior_errors")},
        "grid_prior_kinds": "|".join(prov.get("grid_prior_kinds", [])) if prov else None,
        "grid_prior_calls_match_canonical": (prov.get("grid_prior_calls") == canon) if prov else None,
        "grid_prior_tools_match_canonical": ([c[0] for c in prov.get("grid_prior_calls", [])] == [c[0] for c in canon])
        if prov else None,
        "grid_n_prior_calls": len(prov.get("grid_prior_calls", [])) if prov else None,
        "n_canonical_lookups": len(canon),
        "dpo_system_chars": sys_len, "dpo_system_chars_nohint": sys_len_nohint,
        # overlap
        "sft_decision_rows": ",".join(map(str, sft_rows)),
        "sft_same_prompt": any(sft[i]["prompt"] == r["prompt"] for i in sft_rows),
        "sft_same_prompt_and_target": any(sft[i]["prompt"] == r["prompt"] and sft[i]["completion"][0]["content"] == ch
                                          for i in sft_rows),
        "sft_hintstripped_prompt_rows": len(sft_prompt_keys.get(pk, [])),
        "in_sft_val_prompt": pk in val_prompt_keys,
        "rejected_equals_some_sft_target": [list(x) for x in sft_targets.get(nrm(rj), [])][:3],
        "pair_split": next((k for k in ("train", "val", "test") if r["pair_id"] in splits[k]), None),
        "pair_in_eval_files": ",".join(sorted(k for k, v in eval_pairs.items() if r["pair_id"] in v)),
    }
    if row["rejected_source"] == "observed_failure":
        row["grid_context_same_system_len"] = (row["grid_prompt"] == "P0" and row["grid_system_chars"] == sys_len_nohint)
    else:
        row["grid_context_same_system_len"] = None
    rows.append(row)

df = pd.DataFrame(rows)
with open(os.path.join(OUT, "pairs.jsonl"), "w") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
df.drop(columns=["chosen_text", "rejected_text", "common_prefix_text"]).to_csv(os.path.join(OUT, "pairs.csv"), index=False)


def ct(a, b, data=None):
    d = df if data is None else data
    t = pd.crosstab(d[a], d[b], margins=True)
    return {str(i): {str(c): int(v) for c, v in row.items()} for i, row in t.iterrows()}


def vc(col, data=None):
    d = df if data is None else data
    return {str(k): int(v) for k, v in d[col].value_counts(dropna=False).items()}


obs = df[df.rejected_source == "observed_failure"]
summary = {
    "inputs": {"dpo_file_sha256": file_sha, "pairs_sha256_as_train_dpo": pairs_sha, "n_pairs": len(df),
               "token_stats_misaligned_rows": tok_misaligned, "grid_files_in_build_order": [os.path.basename(f) for f in GRID]},
    "replay_matches_file": vc("replay_matches_file"),
    "decision_classes": {
        "chosen_by_twin": ct("chosen_decision", "twin"), "rejected_by_twin": ct("rejected_decision", "twin"),
        "rejected_by_expected": ct("rejected_decision", "expected"), "chosen_by_expected": ct("chosen_decision", "expected"),
        "chosen_reward": vc("chosen_reward"), "rejected_reward": vc("rejected_reward"),
        "chosen_reward_le_0": int((df.chosen_reward <= 0).sum()),
        "margin_le_0": int((df.chosen_reward - df.rejected_reward <= 0).sum()),
        "chosen_eval_strict_false": int((~df.chosen_eval_strict).sum()),
        "rejected_eval_strict_true": int(df.rejected_eval_strict.sum()),
        "rejected_eval_lenient_true": int(df.rejected_eval_lenient.sum()),
    },
    "rejected_source": {"all": vc("rejected_source"), "by_twin": ct("rejected_source", "twin"),
                        "by_level": ct("rejected_source", "level"), "by_env": ct("rejected_source", "env"),
                        "by_prompt_setting": ct("rejected_source", "prompt_setting"),
                        "by_seed_or_variant": ct("rejected_source", "seed_or_variant")},
    "observed_failure_provenance": {
        "model": vc("model", obs), "model_by_twin": ct("model", "twin", obs), "grid_file": vc("grid_file", obs),
        "grid_prompt_vs_dpo_prompt": ct("grid_prompt", "prompt_setting", obs),
        "grid_n_prior_calls": vc("grid_n_prior_calls", obs),
        "grid_prior_calls_match_canonical": vc("grid_prior_calls_match_canonical", obs),
        "grid_prior_tools_match_canonical": vc("grid_prior_tools_match_canonical", obs),
        "grid_prior_errors_gt0": int((obs.grid_prior_errors > 0).sum()),
        "grid_p0_same_system_len": vc("grid_context_same_system_len", obs[obs.grid_prompt == "P0"]),
        "grid_last_kind": vc("grid_last_kind", obs), "grid_decision": vc("grid_decision", obs),
    },
    "tokens": {
        "chosen_mean": round(df.chosen_tokens.mean(), 1), "rejected_mean": round(df.rejected_tokens.mean(), 1),
        "chosen_median": float(df.chosen_tokens.median()), "rejected_median": float(df.rejected_tokens.median()),
        "chosen_max": int(df.chosen_tokens.max()), "rejected_max": int(df.rejected_tokens.max()),
        "share_rejected_longer": round(float((df.rejected_tokens > df.chosen_tokens).mean()), 3),
        "share_rejected_shorter": round(float((df.rejected_tokens < df.chosen_tokens).mean()), 3),
        "by_twin": {tw: {"chosen_mean": round(d.chosen_tokens.mean(), 1), "rejected_mean": round(d.rejected_tokens.mean(), 1),
                         "share_rejected_longer": round(float((d.rejected_tokens > d.chosen_tokens).mean()), 3)}
                    for tw, d in df.groupby("twin")},
        "by_source": {s: {"n": len(d), "chosen_mean": round(d.chosen_tokens.mean(), 1),
                          "rejected_mean": round(d.rejected_tokens.mean(), 1),
                          "share_rejected_longer": round(float((d.rejected_tokens > d.chosen_tokens).mean()), 3)}
                      for s, d in df.groupby("rejected_source")},
        "rejected_gt_200": df[df.rejected_tokens > 200][["idx", "scenario_id", "level", "prompt_setting", "rejected_tokens",
                                                          "rejected_decision", "model"]].to_dict("records"),
        "longest_rejected": df.sort_values("rejected_tokens", ascending=False).head(1)[
            ["idx", "scenario_id", "level", "prompt_setting", "rejected_tokens", "rejected_decision", "model", "grid_prompt",
             "rejected_max_line_repeat", "rejected_trigram_repeat_share", "rejected_zlib_ratio"]].to_dict("records"),
    },
    "format": {
        "rejected_kind": vc("rejected_kind"), "chosen_kind": vc("chosen_kind"),
        "rejected_strict": vc("rejected_strict"), "chosen_strict": vc("chosen_strict"),
        "rejected_flags": vc("rejected_flags"),
        "rejected_lead_ws": int(df.rejected_lead_ws.sum()), "rejected_trail_ws": int(df.rejected_trail_ws.sum()),
        "chosen_lead_ws": int(df.chosen_lead_ws.sum()), "chosen_trail_ws": int(df.chosen_trail_ws.sum()),
        "rejected_tool_call_multiline": int(((df.rejected_kind == "tool_call") & (df.rejected_newlines > 0)).sum()),
        "rejected_tool_calls": int((df.rejected_kind == "tool_call").sum()),
        "chosen_tool_call_multiline": int(((df.chosen_kind == "tool_call") & (df.chosen_newlines > 0)).sum()),
        "rejected_has_action_md": int(df.rejected_has_action_md.sum()), "rejected_has_think": int(df.rejected_has_think.sum()),
        "rejected_max_line_repeat_ge3": int((df.rejected_max_line_repeat >= 3).sum()),
        "rejected_trigram_repeat_share_ge_0.2": int((df.rejected_trigram_repeat_share >= 0.2).sum()),
        "rejected_nonascii_share_ge_0.2": int((df.rejected_nonascii_share >= 0.2).sum()),
        "first_token_differs": vc("first_token_differs"),
    },
    "overlap": {
        "sft_same_prompt": vc("sft_same_prompt"), "sft_same_prompt_and_target": vc("sft_same_prompt_and_target"),
        "has_sft_decision_row": int((df.sft_decision_rows != "").sum()),
        "in_sft_val_prompt": vc("in_sft_val_prompt"), "pair_split": vc("pair_split"),
        "pair_in_eval_files": vc("pair_in_eval_files"),
        "rejected_equals_some_sft_target": int(df.rejected_equals_some_sft_target.map(bool).sum()),
    },
    "balance": {"twin_by_level": ct("twin", "level"), "twin_by_prompt_setting": ct("twin", "prompt_setting"),
                "twin_by_env": ct("twin", "env"), "expected_by_prompt_setting": ct("expected", "prompt_setting")},
}
json.dump(summary, open(os.path.join(OUT, "summary_pairs.json"), "w"), indent=1, default=str)
print(json.dumps({k: summary[k] for k in ("inputs", "replay_matches_file", "rejected_source")}, indent=1, default=str))
print("PAIRS_AUDIT_OK")
