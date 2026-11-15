# J4 = decision-weighted SFT from the base (same data and hyper-parameters as M1; W=10 on the decision tokens = 16.6 % of the
# loss weight), both epoch ends kept. Needs ~5.2 h + eval; wall limit leaves room for the eval before the GPU cutoff.
source "$WS/training/chain2_lib.sh"
[ -f "$CDIR/markers/45_j4_smoke.done" ] || skip "the J4 smoke did not pass"
left=$(mins_until_ist "$GPU_CUTOFF_IST")
[ "$left" -ge 400 ] || skip "too late for J4 ($left min to the GPU cutoff; it needs ~5.2 h + ~1 h eval)"
maxmin=$((left - 90)); [ "$maxmin" -gt 420 ] && maxmin=420
echo "J4: wall limit $maxmin min ($left min to the GPU cutoff)"
guarded j4_sft 30 26 "$maxmin" expandable_segments:True offline -- python3 training/train_sft.py --model "$BASE" \
    --data "$WS/training/data_v121_curated" --out "$WS/runs/m5_sft_dw" --decision-weight 10 --keep-epoch-ends
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$WS/runs/m5_sft_dw/final/adapter_model.safetensors" ]; then exit 0; fi
if [ -f "$WS/runs/m5_sft_dw/epoch-1/adapter_model.safetensors" ]; then
  attn "J4 ended rc=$rc before epoch 2 finished; the epoch-1 adapter is kept as the J4 arm"; exit 0
fi
attn "J4 failed rc=$rc (logs/chain2/50_j4_sft.log)"; exit 1
