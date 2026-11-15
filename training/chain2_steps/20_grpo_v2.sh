# J3 = GRPO v2 from merged M1: lr 5e-5, 36 steps, pair-balanced order, reward v2, completions logged, a checkpoint every 6 steps,
# through run_pipeline_v2.sh (stage m2_grpo under memguard). The lr-5e-6 run is archived first. Killed or stopped early: the
# newest checkpoint is used (attention entry).
source "$WS/training/chain2_lib.sh"
[ -f "$CDIR/markers/15_grpo_v2_smoke.done" ] || skip "the GRPO v2 smoke did not pass"
D=$(cat "$CDIR/grpo_v2_data_dir.txt")
if [ -d "$WS/runs/m2_grpo" ] && [ ! -e "$WS/runs/m2_grpo_v1_lr5e-6" ] && python3 -c 'import json, sys
sys.exit(0 if float(json.load(open(sys.argv[1]))["lr"]) == 5e-06 else 1)' "$WS/runs/m2_grpo/args.json" 2>/dev/null; then
  mv "$WS/runs/m2_grpo" "$WS/runs/m2_grpo_v1_lr5e-6" && echo "archived the lr-5e-6 GRPO run -> runs/m2_grpo_v1_lr5e-6"
fi
left=$(mins_until_ist "$GPU_CUTOFF_IST")
maxmin=$((left - 300)); [ "$maxmin" -gt 480 ] && maxmin=480
[ "$maxmin" -ge 120 ] || skip "too late for GRPO v2 ($left min to the GPU cutoff)"
echo "GRPO v2: wall limit $maxmin min ($left min to the GPU cutoff)"
pipeline_v2 GRPO_STEPS=36 GRPO_LR=5e-5 GRPO_DATA_DIR="$D" \
  GRPO_EXTRA="--no-shuffle --reward-variant v2 --log-completions --save-steps 6" GRPO_MAX_MIN="$maxmin" \
  START_AT=m2_grpo STOP_AFTER=m2_grpo
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$WS/runs/m2_grpo/final/adapter_model.safetensors" ]; then
  echo "$WS/runs/m2_grpo/final" > "$CDIR/m2_adapter.txt"; exit 0
fi
ck=$(ls -d "$WS"/runs/m2_grpo/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1)
if [ -n "$ck" ] && [ -f "$ck/adapter_model.safetensors" ]; then
  echo "$ck" > "$CDIR/m2_adapter.txt"; attn "GRPO v2 ended rc=$rc before its last step; continuing with $ck"; exit 0
fi
attn "GRPO v2 failed rc=$rc and left no checkpoint (logs/pipeline.log, runs/pipeline/logs/m2_grpo.log)"; exit 1
