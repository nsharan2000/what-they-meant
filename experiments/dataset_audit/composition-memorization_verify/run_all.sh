#!/bin/sh
# Verify pass for the composition-memorization lens. Read-only on dataset/ and experiments/; writes only here.
set -e
cd "$(dirname "$0")"
ROOT=../../..
# independent pre-curation rebuild (training/build_datasets.py writes only to --out)
(cd $ROOT && python3 training/build_datasets.py --scenarios 'dataset/scenarios/*.jsonl' 'dataset/scenarios_synth/*.jsonl' \
   --splits dataset/splits.json --failures 'experiments/replication/results/grid_*.jsonl' \
   --out experiments/dataset_audit/composition-memorization_verify/rebuild)
for s in v1_tokens v2_grpo v3_dupes v4_sftlog v5_val v6_extra v7_curation v8_dpo v9_regurg v10_oracle v11_misc v12_order v13_act v14_actreq v15_power; do
  echo "== $s"; python3 $s.py > /dev/null
done
echo VERIFY_ALL_OK
