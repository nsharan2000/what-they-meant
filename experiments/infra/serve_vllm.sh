#!/bin/bash
# serve_vllm.sh <hf_model_id> <served_name> [port] [max_model_len] [gpu_mem_util]
# Extra vLLM flags via $VLLM_EXTRA_ARGS (e.g. "--enable-lora --max-lora-rank 16 --lora-modules m1=/workspace/runs/m1_sft/final").
# Starts an OpenAI-compatible vLLM server INSIDE lx-exp and blocks until it is ready (prints READY <name> <port>).
# Stop it with experiments/infra/stop_vllm.sh [name]; the PID is in /workspace/logs/vllm-<name>.pid (port in vllm-<name>.port).
# Why the care on a SHARED box (ONE GPU job at a time): every earlier server of ours is stopped by PID and waited for — its
# VLLM::EngineCore child included, port closed — before the new one allocates its pool, so two pools never overlap; liveness is
# `kill -0` on our own PID (a `pgrep -f` pattern would also see other projects' servers); READY means /v1/models lists the served
# name AND every --lora-modules name, so an adapter that failed to load can never be queried as if it were there.
# The server is started with fd 9 closed: the eval scripts, run bare, hold memguard's GPU lock (/workspace/.gpu_job.lock) on fd 9
# for their whole run, and a server kept alive (KEEP_VLLM=1) must not keep that lock after its script exits. A kept server is
# then visible to the next memguard preflight only through MemAvailable (its budget + floor check); stop it with stop_vllm.sh
# first. Under memguard the server is in memguard's tree and is killed by its exit sweep at the latest.
# Gotchas inherited from the previous project (same image): flashinfer cubin/version mismatch on sm_121 →
# FLASHINFER_DISABLE_VERSION_CHECK=1 and VLLM_USE_FLASHINFER_SAMPLER=0; python is python3.
set -euo pipefail
MODEL="$1"; NAME="$2"; PORT="${3:-8001}"; MAXLEN="${4:-32768}"; MEM="${5:-0.55}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${VLLM_LOG_DIR:-/workspace/logs}"
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
mkdir -p "$LOGS"
VLLM_PORT="$PORT" bash "$HERE/stop_vllm.sh" || { echo "VLLM_PREV_NOT_STOPPED $NAME"; exit 1; }
WANT="$NAME"; in_lora=0
for w in ${VLLM_EXTRA_ARGS:-}; do
  case "$w" in
    --lora-modules) in_lora=1 ;;
    --*) in_lora=0 ;;
    *=*) if [ "$in_lora" = 1 ]; then WANT="$WANT ${w%%=*}"; fi ;;
  esac
done
nohup vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$MEM" --dtype bfloat16 --max-num-seqs 64 ${VLLM_EXTRA_ARGS:-} \
  > "$LOGS/vllm-$NAME.log" 2>&1 9>&- &
PID=$!
echo "$PID" > "$LOGS/vllm-$NAME.pid"
echo "$PORT" > "$LOGS/vllm-$NAME.port"
ready() {
  curl -sf --noproxy '*' --max-time 10 "http://localhost:$PORT/v1/models" 2>/dev/null | python3 -c '
import json, sys
try:
    ids = {m["id"] for m in json.load(sys.stdin).get("data", [])}
except ValueError:
    sys.exit(1)
sys.exit(0 if set(sys.argv[1:]) <= ids else 1)' $WANT
}
for i in $(seq 1 180); do
  if ready; then echo "READY $NAME $PORT"; exit 0; fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "VLLM_DIED $NAME"; tail -40 "$LOGS/vllm-$NAME.log"
    VLLM_PORT="$PORT" bash "$HERE/stop_vllm.sh" || true   # reaps an orphaned EngineCore; a VLLM_STOP_FAILED must stay visible
    exit 1
  fi
  sleep 5
done
echo "VLLM_TIMEOUT $NAME (wanted: $WANT)"; tail -40 "$LOGS/vllm-$NAME.log"
VLLM_PORT="$PORT" bash "$HERE/stop_vllm.sh" || true
exit 1
