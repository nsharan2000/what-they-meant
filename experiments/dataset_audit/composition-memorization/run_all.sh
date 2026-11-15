#!/usr/bin/env bash
# Reproduces every number of the composition-memorization audit (CPU only, stdlib + numpy + pandas, no network).
# Inputs are read-only: dataset/curated_v121/*, dataset/scenarios*/, envs/, experiments/dataset_audit/results/token_stats.jsonl,
# experiments/training_eval/results/train_logs/{m1_sft/result.json, m2_grpo/scores.jsonl}. Outputs: ./results/, ./rebuild_v121/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
# pre-curation v1.2.1 build (deterministic; byte-identical to the build CuratorKIT consumed -- compared by sha256 below)
python3 -I training/build_datasets.py --scenarios 'dataset/scenarios/*.jsonl' 'dataset/scenarios_synth/*.jsonl' \
    --splits dataset/splits.json --failures 'experiments/replication/results/grid_*.jsonl' --out "$HERE/rebuild_v121" > /dev/null
shasum -a 256 "$HERE/rebuild_v121/sft_val.jsonl" dataset/curated_v121/sft_val.jsonl
cd "$HERE"
python3 composition.py
python3 twin_contrast.py
python3 grpo_decisions.py > /dev/null
python3 token_composition.py > /dev/null
python3 sft_log.py > /dev/null
python3 wording_overlap.py > /dev/null
python3 curation_diff.py > /dev/null
python3 regurgitation.py > /dev/null
python3 dpo_pivot.py > /dev/null
python3 selector.py > /dev/null
echo COMPOSITION_MEMORIZATION_OK
