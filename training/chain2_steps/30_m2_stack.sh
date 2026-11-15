# Stack M1 + the GRPO v2 adapter into one LoRA on the base (stack_lora.py via run_pipeline_v2.sh stage m2_stack), then measure
# every adapter's weight-space size (CPU).
source "$WS/training/chain2_lib.sh"
[ -s "$CDIR/m2_adapter.txt" ] || skip "no GRPO v2 adapter (step 20)"
A=$(cat "$CDIR/m2_adapter.txt")
pipeline_v2 M2_ADAPTER="$A" START_AT=m2_stack STOP_AFTER=m2_stack \
  || { attn "m2_stack failed for $A (runs/pipeline/logs/m2_stack.log)"; exit 1; }
G1="$WS/runs/m2_grpo_v1_lr5e-6/final"
CUDA_VISIBLE_DEVICES= python3 experiments/training_eval/adapter_effect.py --adapter "m1=$M1" --adapter "m2_grpo_v2=$A" \
    --adapter "m2_grpo_v1=$G1" --adapter "m3=$M3" --out experiments/training_eval/results/train_logs/adapter_effect_grpo_v2.json \
  || echo "adapter_effect failed (non-fatal)"
exit 0
