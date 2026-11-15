#!/bin/bash
# Local base path for Tiny Aya Global (verified mirror of the official weights) on the Spark: serve with vLLM, run the frozen seed grid,
# measure throughput. The trained models are evaluated through this same path. Ends LOCAL_TINYAYA_DONE.
# The vLLM server (0.45 of the shared box) is stopped on exit unless KEEP_VLLM=1 (an idle server once held ~59 GB for 7 h).
# ONE GPU job at a time: run it under memguard (python3 training/memguard.py --name local_tinyaya --budget-gb 70 -- bash
# experiments/replication/run_local_tinyaya.sh); it then proceeds as part of that guarded job (LOCAL_TINYAYA_UNDER_MEMGUARD)
# because the lock file names a memguard.py that is its own ancestor. Run bare, it holds memguard's lock
# (/workspace/.gpu_job.lock, flock(2) on fd 9) until exit. Any other holder: LOCAL_TINYAYA_REFUSE, exit 75; checked before the
# EXIT trap is set, so a refused run never stops the holder's server. The server runs with fd 9 closed (serve_vllm.sh): a bare
# KEEP_VLLM=1 server holds no lock and the next memguard sees it only via MemAvailable; under memguard it dies with the command.
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
  0) echo "run_local_tinyaya pid=$$ since=$(date '+%Y-%m-%d %H:%M:%S')" > "$GPU_LOCK" ;;   # by path: the flock stays on fd 9
  75) if under_memguard; then echo "LOCAL_TINYAYA_UNDER_MEMGUARD $(cat "$GPU_LOCK")"
      else echo "LOCAL_TINYAYA_REFUSE another GPU job holds $GPU_LOCK: $(cat "$GPU_LOCK" 2>/dev/null)"; exit 75; fi ;;
  *) exit 1 ;;
esac
trap '[ "${KEEP_VLLM:-0}" = 1 ] || bash experiments/infra/stop_vllm.sh' EXIT
MP=$(sed -n 2p /workspace/logs/mirror_paths.txt)
bash experiments/infra/serve_vllm.sh "$MP" tiny-aya-global-local 8001 16384 0.45 || { echo SERVE_FAILED; exit 1; }
python3 -m wtm.run --scenarios dataset/frozen_replication_v1/*.jsonl --models vllm:tiny-aya-global-local@http://localhost:8001/v1 \
  --levels L0 L0r L1 L2 L2s --prompts P0 P1 --mode decision --workers 32 \
  --out experiments/replication/results/grid_local_tinyaya.jsonl 2>&1 | grep -E "jobs|done \(|cost|Traceback|Error|RUN_SUMMARY" | tail -5
grep -E "Avg prompt throughput|Avg generation throughput" /workspace/logs/vllm-tiny-aya-global-local.log | tail -5
echo LOCAL_TINYAYA_DONE
