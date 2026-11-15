# Held-out eval + capability for J4: m5 = epoch-2 (final); m5e1 = epoch-1 too when >= 5 h remain (answers: did epoch 2 help?).
source "$WS/training/chain2_lib.sh"
arms=""
[ -f "$WS/runs/m5_sft_dw/final/adapter_config.json" ] && arms="m5=$WS/runs/m5_sft_dw/final"
left=$(mins_until_ist "$GPU_CUTOFF_IST")
if [ -f "$WS/runs/m5_sft_dw/epoch-1/adapter_config.json" ] && { [ -z "$arms" ] || [ "$left" -ge 300 ]; }; then
  arms="$arms m5e1=$WS/runs/m5_sft_dw/epoch-1"
fi
[ -n "$arms" ] || skip "no J4 adapter"
echo "$arms" > "$CDIR/extra_arms.txt"
pipeline_v2 EXTRA_ARMS="$arms" EVAL_BUDGET=92 START_AT=eval STOP_AFTER=eval \
  || { attn "eval with J4 ($arms) failed: logs/pipeline.log"; exit 1; }
