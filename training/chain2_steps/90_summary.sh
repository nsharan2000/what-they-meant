# Pipeline summary over every arm (runs/pipeline/summary.json) — CPU. The chain then goes idle (the watcher's end notice).
source "$WS/training/chain2_lib.sh"
arms=$(cat "$CDIR/extra_arms.txt" 2>/dev/null || true)
env SOFT_STAGES="" EXTRA_ARMS="$arms" START_AT=summary STOP_AFTER=summary bash training/run_pipeline_v2.sh || true
ls -la experiments/training_eval/results/ | tail -n 30
exit 0
