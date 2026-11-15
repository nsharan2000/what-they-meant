# Check-in: GRPO v2 (m2) has been evaluated (or its eval failed). Raises an informational attention entry so the session wakes up
# to analyse it while J4 trains (the session is otherwise woken only by failures, idle or the end).
source "$WS/training/chain2_lib.sh"
if [ -f "$CDIR/markers/40_eval_m2.done" ]; then
  attn "CHECK-IN (not a failure): eval with GRPO v2 (m2) finished — experiments/training_eval/results/eval_{test,hard_test}.jsonl; J4 starts next"
else
  attn "CHECK-IN: the eval with m2 did not finish (see 40_eval_m2); J4 starts next"
fi
exit 0
