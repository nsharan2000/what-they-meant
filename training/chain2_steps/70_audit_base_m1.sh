# SafeTune drift (HarmBench + XSTest) and AlignTune alignment audit for the core before/after pair: base and M1 (same commands
# and memguard budgets as run_pipeline.sh audit_pair; both scripts cache per arm, so later steps only add arms).
source "$WS/training/chain2_lib.sh"
R=experiments/training_eval/results
guarded audit_safety_m1 44 32 240 expandable_segments:True online -- python3 experiments/training_eval/safety_eval.py \
    --base "$BASE" --arms base "m1=$M1" --out "$R/safety.json" --gen-dir "$R/cache/safety.gen" \
  || { attn "safety audit (base, m1) failed (logs/chain2/70_audit_base_m1.log)"; exit 1; }
guarded audit_align_base 24 16 120 expandable_segments:True online -- python3 experiments/training_eval/alignment_audit.py \
    --base "$BASE" --arms base --out "$R/alignment_base.json" \
  || { attn "alignment audit (base) failed"; exit 1; }
guarded audit_align_m1 24 16 120 expandable_segments:True online -- python3 experiments/training_eval/alignment_audit.py \
    --base "$BASE" --arms "m1=$M1" --out "$R/alignment_tuned.json" --baseline-arm "$R/alignment_base.arms/base.json" \
  || { attn "alignment audit (m1) failed"; exit 1; }
