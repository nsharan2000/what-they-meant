# Safety + alignment audit of the M3 DPO (lr 5e-6) arm (m3), after base/m1 (step 70). ~1 h; skipped when < 2 h remain to the GPU cutoff.
source "$WS/training/chain2_lib.sh"
R=experiments/training_eval/results
A="$M3"
[ -f "$A/adapter_config.json" ] || skip "no m3 adapter at $A"
if [ -f "$A/stack_info.json" ] && ! python3 -c 'import json, sys
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("ok") is True and d.get("verified") is True else 1)' "$A/stack_info.json"; then
  skip "m3 stack not verified"
fi
left=$(mins_until_ist "$GPU_CUTOFF_IST")
[ "$left" -ge 120 ] || skip "too late for the m3 audit ($left min to the GPU cutoff)"
guarded audit_safety_m3 44 32 180 expandable_segments:True online -- python3 experiments/training_eval/safety_eval.py \
    --base "$BASE" --arms base "m3=$A" --out "$R/safety.json" --gen-dir "$R/cache/safety.gen" \
  || { attn "safety audit (m3) failed"; exit 1; }
guarded audit_align_m3 24 16 120 expandable_segments:True online -- python3 experiments/training_eval/alignment_audit.py \
    --base "$BASE" --arms "m3=$A" --out "$R/alignment_tuned.json" --baseline-arm "$R/alignment_base.arms/base.json" \
  || { attn "alignment audit (m3) failed"; exit 1; }
