#!/bin/bash
# E3 (audit): closed-model arms on the FROZEN eval specs (dataset/eval test + hard_test, sha 66e755e8fc09334f = the Spark's),
# same decision mode as the Tiny Aya eval, levels L0r and L2 x prompts P0 and P2. Pure Claude Code agents (wtm/drivers/claude_pure.py).
# Output outside ~/Desktop (long local runs lose write access there); copied into experiments/training_eval/results/ afterwards.
cd "$(dirname "$0")/../.." || exit 1
OUT=scratch/e3
for spec in "haiku 8" "opus 6"; do
  set -- $spec
  for split in test hard_test; do
    python3 -m wtm.run --scenarios dataset/eval/${split}_*.jsonl --models claude:$1 --levels L0r L2 --prompts P0 P2 \
      --mode decision --workers $2 --out "$OUT/eval_${split}_claude.jsonl"
    echo "E3_DONE $1 $split $(TZ=Asia/Kolkata date '+%H:%M IST')"
  done
done
echo "E3_ALL_DONE $(TZ=Asia/Kolkata date '+%H:%M IST')"
