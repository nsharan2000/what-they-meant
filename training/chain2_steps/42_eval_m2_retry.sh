# Eval of the stacked GRPO v2 arm (m2) after the stack was verified exactly in weight space (training/verify_stack_exact.py,
# 2026-11-15 15:5x: relative error 1.05e-7) — step 30's probe check was inconclusive (GRPO's probe effect < M1's bf16 merge
# rounding). Base/m1/m3 episodes and capability are cached; only m2 runs.
source "$WS/training/chain2_lib.sh"
python3 -c 'import json, sys
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("ok") is True and d.get("verified") is True else 1)' \
  "$WS/runs/m2_grpo/stacked/stack_info.json" 2>/dev/null || skip "no verified stacked m2"
pipeline_v2 EVAL_BUDGET=92 START_AT=eval STOP_AFTER=eval \
  || { attn "eval with the stacked GRPO v2 (m2) failed: logs/pipeline.log"; exit 1; }
attn "CHECK-IN (not a failure): GRPO v2 (m2) evaluated — experiments/training_eval/results/eval_{test,hard_test}.jsonl; J4 next"
exit 0
