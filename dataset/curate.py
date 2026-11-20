"""CuratorKIT pass over the training views (run inside lx-exp, where curatorkit is installed editable).

Twin-safe CuratorKIT settings:
  * ingest through `preprocessing_fn` returning DataSample (skips the nondeterministic alias detection)
  * clean_transforms strip_html=False (it would delete <tool_call> tags), max_tokens=32768
  * dedup="exact" only — MinHash / DiversityGate / embedding dedup would merge twin A and twin B (they differ by one fact)
  * secrets_gate=True (false positives on synthetic e-mails/handles are measured on a planted clean set and reported)
  * NO PIIPseudonymizer (rewrites near-miss names independently per sample), NO output_split (random split leaks twins)
  * optional judge gates (HallucinationGate / RewardGate) use an OPEN model through the HF router — training-data provenance
    policy (questions-for-user Q5): Claude never gates or labels training rows
Splits are ours (by source post); CuratorKIT's manifest.json stage counts + rejected.jsonl reasons feed the curation flowchart.

python3 dataset/curate.py --data training/data --out dataset/curated [--judge openai/Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="training/data")
ap.add_argument("--out", default="dataset/curated")
ap.add_argument("--judge", default=None, help="LiteLLM model id for HallucinationGate/RewardGate (open model via HF router)")
ap.add_argument("--planted-clean", type=int, default=20, help="clean control rows to measure SecretsGate false positives")
ns = ap.parse_args()

from curatorkit import Curator, CuratorConfig, DataSample  # noqa: E402


def rows(name):
    p = os.path.join(ns.data, name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def pair_sample(r):
    return DataSample(source_uri=f"wtm://{r['scenario_id']}/{r['level']}/{r.get('prompt_setting', 'P0')}", task_type="preference",
                      instruction=json.dumps(r["prompt"], ensure_ascii=False),
                      chosen=json.dumps(r["chosen"], ensure_ascii=False), rejected=json.dumps(r["rejected"], ensure_ascii=False),
                      input=r["prompt"][-1]["content"][:4000],
                      metadata={"wtm_scenario": r["scenario_id"], "wtm_twin": r["twin"], "wtm_level": r["level"]})


def sft_sample(r):
    return DataSample(source_uri=f"wtm://{r['scenario_id']}/{r['level']}/{r.get('turn')}", task_type="conversational",
                      instruction=json.dumps(r["prompt"], ensure_ascii=False), output=r["completion"][0]["content"],
                      metadata={"wtm_scenario": r["scenario_id"], "wtm_twin": r["twin"], "wtm_turn": r.get("turn")})


def run(kind, data_rows, fn, fmt):
    os.makedirs(ns.out, exist_ok=True)
    src = os.path.join(ns.out, f"_in_{kind}.jsonl")
    with open(src, "w") as fh:
        for r in data_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cfg = dict(dataset=src, preprocessing_fn=fn, clean_transforms={"strip_html": False}, max_tokens=32768, dedup="exact",
               secrets_gate=True, export_formats=[fmt], output_dir=os.path.join(ns.out, kind))
    if ns.judge:
        cfg.update(judge_llm=ns.judge, judge_llm_api_base="https://router.huggingface.co/v1")
    res = Curator(CuratorConfig(**cfg)).run()
    man_p = os.path.join(ns.out, kind, "manifest.json")
    manifest = json.load(open(man_p)) if os.path.exists(man_p) else {}
    return {"kind": kind, "in": len(data_rows), "passed": len(getattr(res, "passed", []) or []), "manifest": manifest}


report = {}
pairs, sft = rows("dpo_train.jsonl"), rows("sft_train.jsonl")
if pairs:
    report["dpo"] = run("dpo", pairs, pair_sample, "dpo")
if sft:
    report["sft"] = run("sft", sft, sft_sample, "messages")
# planted clean control for SecretsGate false positives: plain-English rows with no identifiers at all
clean = [{"scenario_id": f"CLEAN{i}", "level": "L0", "twin": "B_should_act", "turn": "decision",
          "prompt": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": f"Please summarise item number {i} of the reading list in one sentence."}],
          "completion": [{"role": "assistant", "content": "It explains how to plan a weekly budget."}]} for i in range(ns.planted_clean)]
report["planted_clean_control"] = run("clean_control", clean, sft_sample, "messages")
json.dump(report, open(os.path.join(ns.out, "curation_report.json"), "w"), indent=1, default=str)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "manifest"} for k, v in report.items()}, indent=1))
