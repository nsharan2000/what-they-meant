# Held-out eval (test + hard_test, L0r/L1/L2/L2s x P0/P2) + capability for every arm incl. the stacked GRPO v2 (m2);
# base/m1/m3 episodes already done are cached by run_eval.sh / capability_eval.py.
source "$WS/training/chain2_lib.sh"
pipeline_v2 EVAL_BUDGET=92 START_AT=eval STOP_AFTER=eval \
  || { attn "eval (with m2) failed: logs/pipeline.log, experiments/training_eval/results/eval_*.jsonl.log"; exit 1; }
