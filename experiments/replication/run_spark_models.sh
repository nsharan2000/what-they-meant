#!/bin/bash
# Chain on the Spark (inside lx-exp): serve each cached local model with vLLM, run the replication grid against it, stop it.
# ONE GPU job at a time; resumable (wtm.run skips finished keys). Ends with SPARK_GRID_DONE.
# Run it under memguard (python3 training/memguard.py --name spark_grid --budget-gb 80 -- bash
# experiments/replication/run_spark_models.sh; budget: 0.55 pool ~67 GB of 121 plus engine overhead); it then proceeds as part
# of that guarded job (SPARK_GRID_UNDER_MEMGUARD) because the lock file names a memguard.py that is its own ancestor. Run bare,
# it holds memguard's lock (/workspace/.gpu_job.lock, flock(2) on fd 9) for the whole chain. Any other holder:
# SPARK_GRID_REFUSE, exit 75; checked before the EXIT trap is set, so a refused run never stops the holder's server. Servers
# are stopped by PID file (stop_vllm.sh, waits until the memory is free), never by name; the EXIT trap stops the last one.
cd /workspace
GPU_LOCK=/workspace/.gpu_job.lock
under_memguard() {   # the lock's holder is the memguard.py named in the lock file AND an ancestor of this script
  local holder p=$PPID
  holder=$(sed -n 's/^.* pid=\([0-9][0-9]*\) since=.*$/\1/p' "$GPU_LOCK" | head -1)
  while [ -n "$holder" ] && [ "${p:-1}" -gt 1 ]; do
    [ "$p" = "$holder" ] && { ps -o args= -p "$p" | grep -q 'memguard\.py'; return; }
    p=$(ps -o ppid= -p "$p" | tr -d ' ')
  done
  return 1
}
exec 9>>"$GPU_LOCK" || exit 1
python3 -c 'import errno, fcntl, sys
try:
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as e:
    sys.exit(75 if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES) else f"GPU lock: {e}")'
case $? in
  0) echo "run_spark_models pid=$$ since=$(date '+%Y-%m-%d %H:%M:%S')" > "$GPU_LOCK" ;;   # by path: the flock stays on fd 9
  75) if under_memguard; then echo "SPARK_GRID_UNDER_MEMGUARD $(cat "$GPU_LOCK")"
      else echo "SPARK_GRID_REFUSE another GPU job holds $GPU_LOCK: $(cat "$GPU_LOCK" 2>/dev/null)"; exit 75; fi ;;
  *) exit 1 ;;
esac
trap 'bash experiments/infra/stop_vllm.sh' EXIT
MODELS=("Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct" "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct" "mistralai/Mistral-7B-Instruct-v0.3|mistral-7b-instruct")
LEVELS="${LEVELS:-L0 L0r L1 L2}"; PROMPTS="${PROMPTS:-P0 P1}"
for spec in "${MODELS[@]}"; do
  hf="${spec%%|*}"; name="${spec##*|}"
  echo "=== $name $(date)"
  bash experiments/infra/serve_vllm.sh "$hf" "$name" 8001 16384 0.55 || { echo "SERVE_FAILED $name"; continue; }
  python3 -m wtm.run --scenarios dataset/scenarios/*.jsonl --models "vllm:$name@http://localhost:8001/v1" \
    --levels $LEVELS --prompts $PROMPTS --mode decision --workers 24 \
    --out experiments/replication/results/grid_spark.jsonl 2>&1 | grep -E "done|jobs|cost|Error|Traceback" | tail -5
  bash experiments/infra/stop_vllm.sh || { echo "SPARK_GRID_STOP_FAILED $name"; exit 1; }
done
echo SPARK_GRID_DONE
