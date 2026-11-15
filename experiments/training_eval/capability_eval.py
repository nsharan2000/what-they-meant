"""Capability retention through AuditKIT's lm-eval front door (`ak.run_lmeval`), base vs each trained adapter on ONE vLLM server.

Tasks (Tiny Aya is multilingual, so retention is checked beyond English):
  ifeval (instruction following, chat template) · gsm8k (5-shot maths) · global_mmlu_full_{en,hi,de,es,ja} (knowledge, loglikelihood)
  · mgsm_direct_{en,hi,de,es,ja} (multilingual maths). Same limits/seeds for every arm; results → JSON per arm + ak.compare deltas.
An arm is label=served_name, or a bare label (m1) resolved through --adapters-txt: run_eval.sh serves each adapter as
<label>-<content hash> and appends a block per READY server to <out>.adapters.txt; the LAST block names what the (KEEP_VLLM=1)
server is serving now, so an older adapter version can never be measured under the new label. The tokenizer path is the base's.

python3 experiments/training_eval/capability_eval.py --arms base=tiny-aya-global-local m1 m2 m3 \
    --adapters-txt experiments/training_eval/results/eval_test.adapters.txt --tokenizer <mirror path> \
    --out experiments/training_eval/results/capability.json [--limit-scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import auditkit as ak  # noqa: E402
from wtm.run import served_adapters  # noqa: E402

TASKS = [  # (task list, num_fewshot, limit, chat_template)
    (["ifeval"], 0, 200, True),
    (["gsm8k"], 5, 250, True),
    (["global_mmlu_full_en", "global_mmlu_full_hi", "global_mmlu_full_de", "global_mmlu_full_es", "global_mmlu_full_ja"], 0, 8, False),
    (["mgsm_direct_en", "mgsm_direct_bn", "mgsm_direct_de", "mgsm_direct_es", "mgsm_direct_ja"], 0, 100, True),   # MGSM has no Hindi
]


def resolve_arms(arms: list[str], adapters_txt: str | None) -> list[tuple[str, str]]:
    """[(label, served name)]; exits with ARM_UNRESOLVED before any request when a bare label has no served name."""
    mapping = {}
    if adapters_txt:
        try:
            mapping = served_adapters(adapters_txt)
        except (OSError, ValueError) as e:
            sys.exit(f"ARM_UNRESOLVED cannot read --adapters-txt: {e}")
    out = []
    for arm in arms:
        label, eq, served = arm.partition("=")
        if not eq:
            if label not in mapping:
                sys.exit(f"ARM_UNRESOLVED {label}: " + (f"not in the last run_eval block of {adapters_txt} "
                                                         f"(it has: {' '.join(sorted(mapping)) or 'no adapters'})"
                                                         if adapters_txt else "give label=served_name or --adapters-txt"))
            served = mapping[label]
        out.append((label, served))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="label=served_model_name, or a bare label with --adapters-txt")
    ap.add_argument("--adapters-txt", default=None, help="run_eval.sh's <out>.adapters.txt (resolves bare labels)")
    ap.add_argument("--base-url", default="http://localhost:8001/v1/completions")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-scale", type=float, default=1.0, help="0.1 for a smoke test")
    ap.add_argument("--only", nargs="*", default=None, help="run only task groups whose first task starts with one of these")
    ns = ap.parse_args()
    arms = resolve_arms(ns.arms, ns.adapters_txt)
    report = json.load(open(ns.out)) if os.path.exists(ns.out) else {"tasks": [t[0] for t in TASKS], "arms": {}, "compare": {}}
    runs = {}
    for label, served in arms:
        report["arms"].setdefault(label, {})
        for tasks, shots, limit, chat in TASKS:
            if ns.only and not any(tasks[0].startswith(o) for o in ns.only):
                continue
            t0 = time.time()
            lim = max(2, int(limit * ns.limit_scale))
            try:
                # concurrency / retries / tokenizer backend are lm-eval MODEL args: AuditKIT forwards only a whitelist of
                # top-level options, so they go through model_args (otherwise requests run one at a time)
                r = ak.run_lmeval(tasks, model=f"api:{served}", base_url=ns.base_url, tokenizer=ns.tokenizer,
                                  model_args={"num_concurrent": 16, "max_retries": 3, "tokenizer_backend": "huggingface"},
                                  num_fewshot=shots, limit=lim, apply_chat_template=chat,
                                  gen_kwargs="temperature=0,max_gen_toks=768",
                                  experiment_name=f"wtm-capability-{label}-{tasks[0]}")
                runs[(label, tasks[0])] = r
                report["arms"][label][tasks[0]] = {"headline": r.headline, "limit": lim, "shots": shots,
                                                   "minutes": round((time.time() - t0) / 60, 1)}
            except Exception as e:  # noqa: BLE001 — record and continue: one broken task must not hide the others
                report["arms"][label][tasks[0]] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(label, tasks[0], json.dumps(report["arms"][label][tasks[0]])[:300], flush=True)
            json.dump(report, open(ns.out, "w"), indent=1, default=str)
    base = arms[0][0]
    for (label, t), r in runs.items():
        if label != base and (base, t) in runs:
            cmp = ak.compare(runs[(base, t)], r)
            report["compare"][f"{label}|{t}"] = {"summary": cmp.summary(), "retention": str(cmp.retention())}
    json.dump(report, open(ns.out, "w"), indent=1, default=str)
    print("CAPABILITY_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
