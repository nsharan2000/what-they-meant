# Safety + alignment audit of the GRPO v2 stacked arm (m2), after base/m1 (step 70). ~1 h; skipped when < 2 h remain to the GPU cutoff.
source "$WS/training/chain2_lib.sh"
R=experiments/training_eval/results
A="$WS/runs/m2_grpo/stacked"
[ -f "$A/adapter_config.json" ] || skip "no m2 adapter at $A"
if [ -f "$A/stack_info.json" ] && ! python3 -c 'import json, sys
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("ok") is True and d.get("verified") is True else 1)' "$A/stack_info.json"; then
  skip "m2 stack not verified"
fi
left=$(mins_until_ist "$GPU_CUTOFF_IST")
[ "$left" -ge 120 ] || skip "too late for the m2 audit ($left min to the GPU cutoff)"
guarded audit_safety_m2 44 32 180 expandable_segments:True online -- python3 experiments/training_eval/safety_eval.py \
    --base "$BASE" --arms base "m2=$A" --out "$R/safety.json" --gen-dir "$R/cache/safety.gen" \
  || { attn "safety audit (m2) failed"; exit 1; }
guarded audit_align_m2 24 16 120 expandable_segments:True online -- python3 experiments/training_eval/alignment_audit.py \
    --base "$BASE" --arms "m2=$A" --out "$R/alignment_tuned.json" --baseline-arm "$R/alignment_base.arms/base.json" \
  || { attn "alignment audit (m2) failed"; exit 1; }
