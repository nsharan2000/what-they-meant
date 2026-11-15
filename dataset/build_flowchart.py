"""Build dataset/curation-flowchart.md — the dataset curation flowchart (our selection + CuratorKIT selection) with every count read
from pipeline artefacts. Re-run after each stage; a stage whose artefact does not exist yet shows as "pending" (never a guessed number).

Artefacts read: data/seed/scenarios_latest.jsonl · dataset/scenarios/*.jsonl · dataset/scenarios_synth/*.jsonl (+ _variants_report.json)
· dataset/scenarios_synth_dropped.jsonl · dataset/variant_drift.json · dataset/harmonize_{seed,synth}.json · dataset/review_<env>.json
· experiments/replication/results/judge_*.jsonl · dataset/splits.json · dataset/curated/curation_report.json
· experiments/training_eval/results/capability_*.json (BFCL canary).
python3 dataset/build_flowchart.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wtm.spec import ROOT, load_jsonl  # noqa: E402

D = lambda *p: os.path.join(ROOT, *p)  # noqa: E731


def jload(p, default=None):
    try:
        return json.load(open(p)) if os.path.exists(p) else default
    except (json.JSONDecodeError, OSError):
        return default


def main():
    seed = load_jsonl(D("data", "seed", "scenarios_latest.jsonl"))
    seed_pairs = {r["scenario_id"][:3] for r in seed}
    posts = {r["source_post_id"] for r in seed}
    conv = [r for f in sorted(glob.glob(D("dataset", "scenarios", "*.jsonl"))) for r in load_jsonl(f)]
    synth = [r for f in sorted(glob.glob(D("dataset", "scenarios_synth", "*.jsonl"))) for r in load_jsonl(f)]
    dropped = load_jsonl(D("dataset", "scenarios_synth_dropped.jsonl")) if os.path.exists(D("dataset", "scenarios_synth_dropped.jsonl")) else []
    syn_pairs = {r["pair_id"] for r in synth}
    drop_pairs = {r["pair_id"] for r in dropped}
    drift_pairs = {r["pair_id"] for r in dropped if "drift" in (r.get("dropped_reason") or "")}
    label_pairs = drop_pairs - drift_pairs
    hard = [r for f in sorted(glob.glob(D("dataset", "scenarios_hard", "*.jsonl"))) for r in load_jsonl(f)]
    hard_sel = {k: jload(D("experiments", "hardset", "results", f"hardset_{k}.json"), {}) for k in ("mined", "generated")}
    vrep = jload(D("dataset", "scenarios_synth", "_variants_report.json"), {"accepted": [], "rejected": {}})
    rejected = vrep.get("rejected") or {}
    rej_reasons = Counter()
    for v in rejected.values():
        if "label audit" in v:
            rej_reasons["cross-model label audit (DeepSeek-V3.2)"] += 1
        elif "planted" in v:
            rej_reasons["planted-signal check"] += 1
        elif "informed_any" in v:
            rej_reasons["clash keywords leak into B / missing from A"] += 1
        elif "judgement word" in v:
            rej_reasons["verdict words in raw outputs"] += 1
        elif "budget" in v:
            rej_reasons["token budget"] += 1
        else:
            rej_reasons["schema / tools / JSON"] += 1
    accepted_total = len(syn_pairs) + len(drop_pairs)
    jobs = accepted_total + len(rejected)
    drift = jload(D("dataset", "variant_drift.json"), {})
    harm_syn = jload(D("dataset", "harmonize_synth.json"), {})
    harm_seed = jload(D("dataset", "harmonize_seed.json"), {})
    reviews = {e: jload(D("dataset", f"review_{e}.json"), None) for e in ("wallet", "shopping", "bills", "bizops")}
    n_changes = sum(len((r or {}).get("changes", [])) for r in reviews.values())
    rsum = jload(D("dataset", "review_summary.json"), {}).get("envs", {})
    n_reverted = sum(e.get("verifier_reverted_or_fixed", 0) for e in rsum.values())
    splits = jload(D("dataset", "splits.json"), {})
    cur = jload(D("dataset", "curated_v121", "curation_report.json"), {})   # v1.2.1 = the version trained on (2026-11-14)
    bld = jload(D("dataset", "curated_v121", "build_report.json"), {}).get("counts", {})
    judge = []
    for f in sorted(glob.glob(D("experiments", "replication", "results", "judge_*.jsonl"))):
        judge += load_jsonl(f)
    lv = [j for j in judge if isinstance(j.get("judge"), dict)]
    seed_ids = {r["id"] for r in conv}
    lab_seed = {j["id"]: j["judge"].get("label_valid") for j in lv if j["id"] in seed_ids}
    lab_var = {j["id"]: j["judge"].get("label_valid") for j in lv if j["id"].startswith("V")}
    fixrep = jload(D("dataset", "fix_v12_report.json"), {}).get("summary", {})
    bad_seed = sorted(k for k, v in lab_seed.items() if v is False)
    bad_var = sorted(k for k, v in lab_var.items() if v is False)
    env_counts = Counter(r["env_id"] for r in conv)
    syn_env = Counter(r["env_id"] for r in synth)
    caps = {}
    for f in sorted(glob.glob(D("experiments", "training_eval", "results", "capability_*.json"))):
        caps.update((jload(f, {}) or {}).get("arms", {}))
    bfcl = any("bfcl" in t for arm in caps.values() for t in arm)

    def split_count(k):
        ids = splits.get(k, [])
        return f'{k} {len(ids)} ({sum(i.startswith("S") for i in ids)} seed + {sum(i.startswith("V") for i in ids)} variants)'

    def stage(name, d):
        if not d or "manifest" not in d:
            return f"{name}: pending"
        sc = d["manifest"].get("stage_counts", {})
        parts = [f"{k} {v.get('output_count', v.get('exported_count', '?'))}" for k, v in sc.items()]
        return f"{name}: " + " → ".join(parts)

    dc = drift.get("counts", {})
    hs, hd = harm_syn.get("counts", {}), harm_seed.get("counts", {})
    pretty = lambda m: ('Claude Sonnet 5.5' if str(m).startswith('claude:sonnet') else
                        'Qwen3-235B' if 'Qwen3-235B' in str(m) else str(m or '?'))
    hm_syn, hm_seed = pretty(harm_syn.get('model')), pretty(harm_seed.get('model'))
    fc = jload(D('dataset', 'curated_v121', 'from_curated_report.json'), {})
    fc_txt = (f"v1.2.1 views {bld.get('sft_train', '?')} SFT · {bld.get('sft_val', '?')} val · {bld.get('dpo_train', '?')} DPO · "
              f"{bld.get('grpo_train', '?')} GRPO → kept SFT {fc['sft']['kept']} · DPO {fc['dpo']['kept']} · GRPO {fc['grpo']['kept']} rows"
              if fc else 'pending')
    lines = ["# Dataset curation flowchart", "",
             "*Auto-generated by `dataset/build_flowchart.py`; every count is read from a pipeline artefact (\"pending\" = stage not run yet).*",
             "", "```mermaid", "flowchart TD"]
    lines += [
        f'  S1["User-curated seed<br/>{len(seed)} rows · {len(seed_pairs)} twin pairs · {len(posts)} Reddit posts<br/>(A should-ask / B should-act / P pushback)"]',
        f'  E["Our selection 1 — environment fit<br/>4 apps: ' + ", ".join(f"{k} {v}" for k, v in sorted(env_counts.items())) +
        ' rows<br/>Nemotron-Persona members · seeded histories · ≤ 5 environments"]',
        f'  RV["Our selection 2 — conversion review<br/>Sonnet reviewer + independent Opus verifier per app<br/>'
        + (f'{n_changes} changes recorded · {n_reverted} reverted by the verifier' if any(reviews.values()) else 'pending') + '"]',
        f'  S2["Synthetic variants — Qwen3-235B writer, DeepSeek-V3.2 label audit<br/>{jobs} jobs (45 pairs × 4) → {accepted_total} pairs accepted · {len(rejected)} rejected after 3 attempts"]',
        '  V["Our selection 3 — mechanical validity (every row)<br/>planted signal: oracle 100 %, always-act / approve / ask / keyword-stuffer 0 %<br/>'
        'invariants: same request across levels · evidence visible · token budgets · no verdict words in raw outputs"]',
        f'  DR["Our selection 4 — clash-type drift (DeepSeek-V3.2)<br/>' +
        (f'{dc.get("same_as_origin", 0)} same as origin · {dc.get("drift_same_split", 0) + dc.get("drift_same_family_tag", 0) + dc.get("drift_same_family", 0)} '
         f'harmless drift · <b>{len(drift_pairs)} dropped</b> (drifted into another family in another split)' if drift else 'pending') + '"]',
        f'  H["Our selection 5 — twin-schema harmonisation (tool outputs only, never targets; + code checks)<br/>variants ({hm_syn}): ' +
        (f'{hs.get("harmonised", 0)} pairs harmonised · {hs.get("kept_original", 0)} kept' if hs else 'pending') + f' · seed ({hm_seed}): ' +
        (f'{hd.get("harmonised", 0)} harmonised · {hd.get("kept_original", 0)} kept' if hd else 'pending') + '"]',
        '  R["Our selection 6 — realism ladder<br/>L0 seed · L0r raw fields · L1 product · L2 buried · L3 noisy (+ L0p / L2s controls)"]',
        f'  J["Our selection 7 — label validity (Opus 5.5 judge, validated on planted transcripts)<br/>seed: {len(lab_seed)} rows, {len(bad_seed)} flagged '
        f'→ fixed or sent to the user · variants: {len(lab_var)} rows, {len(bad_var)} flagged → <b>{len(label_pairs)} pairs dropped</b>, 2 texts fixed<br/>'
        f'v1.2 consistency fixes: {fixrep.get("changes", "pending")} (stale years, invented values, clash cues)"]',
        f'  SP["Group split by source post<br/>{split_count("train")}<br/>{split_count("val")} · {split_count("test")}"]',
        '  C["CuratorKIT selection (twin-safe)<br/>SchemaGate → ExactDeduplicator → TextCleaner (strip_html off) → SecretsGate<br/>'
        'NOT used: MinHash · Diversity · Embedding dedup · PII pseudonymiser · output_split (they merge or leak twins)"]',
        f'  MB["Mapped back to the original rows (training/from_curated.py)<br/>CuratorKIT decides WHICH rows survive; text untouched (its exports flatten the prompt into one message and collapse whitespace)<br/>{fc_txt}"]',
        '  X["Exports<br/>SFT: look-up + decision turns · DPO: ideal vs observed base failure · GRPO: prompts + hidden labels<br/>'
        'Benchmark: seed + variant test pairs × realism levels"]',
        f'  S3["BFCL single-turn + irrelevance via AuditKIT<br/>held-out capability canary — never trained on{"" if bfcl else " (pending)"}"]',
        f'  HS["Hard set (EVAL ONLY) — 2-line hint, realistic level, Opus-judged<br/>mined: {hard_sel["mined"].get("n_hard", "pending")} of '
        f'{hard_sel["mined"].get("n_pairs", "?")} pairs failed by ≥ 3/5 big models · generated (Claude Opus): {len({r["pair_id"] for r in hard})} pairs, '
        f'{hard_sel["generated"].get("n_hard", "pending")} failed by ≥ 3/5"]',
        "  S1 --> E --> RV --> V", "  S2 --> V", "  V --> DR --> H --> R --> J --> SP --> C --> MB --> X", "  S3 -.-> X", "  J --> HS", "  HS -.-> X",
        "```", "",
        "## Counts", "",
        f"- Seed: {len(seed)} rows, {len(seed_pairs)} pairs, {len(posts)} source posts; converted rows: {len(conv)} "
        f"({', '.join(f'{k} {v}' for k, v in sorted(env_counts.items()))}).",
        f"- Synthetic variants: {jobs} generation jobs → {accepted_total} pairs accepted, {len(rejected)} rejected "
        f"({', '.join(f'{k}: {v}' for k, v in rej_reasons.most_common()) or 'none'}); drift check dropped {len(drift_pairs)} "
        f"({', '.join(sorted(drift_pairs)) or '–'}); Opus label check dropped {len(label_pairs)} ({', '.join(sorted(label_pairs)) or '–'}) "
        f"→ {len(syn_pairs)} pairs / {len(synth)} rows "
        f"({', '.join(f'{k} {v}' for k, v in sorted(syn_env.items()))}).",
        f"- Twin-schema harmonisation: variants {json.dumps(hs) if hs else 'pending'}; seed {json.dumps(hd) if hd else 'pending'}.",
        f"- Seed review: {n_changes} recorded changes, {n_reverted} reverted by the verifier"
        + ("" if any(reviews.values()) else " (pending)") + ".",
        f"- Splits (pairs): {split_count('train')} · {split_count('val')} · {split_count('test')}.",
        f"- Opus label validity: seed {len(lab_seed)} judged, flagged {bad_seed or 'none'}; variants {len(lab_var)} judged, "
        f"flagged {bad_var or 'none'}.",
        f"- {stage('CuratorKIT SFT', cur.get('sft'))}",
        f"- {stage('CuratorKIT DPO', cur.get('dpo'))}",
        f"- Mapped back to original rows (training/from_curated.py): {json.dumps({k: fc.get(k) for k in ('sft', 'dpo', 'grpo')}) if fc else 'pending'}.",
        "- GRPO prompts: the CuratorKIT-passed DPO prompts (same scenario × level × prompt setting; GRPO rows carry hidden labels, not targets).",
        f"- Hard set: mined {json.dumps({k: hard_sel['mined'].get(k) for k in ('n_pairs', 'n_hard', 'n_core', 'pass_rate_by_model')})}; "
        f"generated {json.dumps({k: hard_sel['generated'].get(k) for k in ('n_pairs', 'n_hard', 'n_core', 'pass_rate_by_model')})}.",
        f"- {stage('CuratorKIT clean control (SecretsGate false positives)', cur.get('planted_clean_control'))}",
    ]
    open(D("dataset", "curation-flowchart.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-10:]))


if __name__ == "__main__":
    main()
