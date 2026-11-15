# chain2_lib.sh — helpers sourced by every training/chain2_steps/*.sh (see training/run_chain2.py for the step protocol:
# exit 0 done · 3 skipped (CHAIN2_SKIP <reason>) · 4 not yet (driver retries after CHAIN2_POLL_S) · other failed;
# a 'CHAIN2_ATTENTION <text>' line wakes the session).
set -uo pipefail
WS="${WS:-/workspace}"
cd "$WS" || exit 1
CDIR="${CHAIN2_DIR:-$WS/runs/chain2}"
mkdir -p "$CDIR/memguard"
BASE="${BASE_MODEL:-$(ls -d /root/.cache/huggingface/hub/models--1-800-LLMs--tiny-aya-global/snapshots/* 2>/dev/null | head -n 1)}"
M1="$WS/runs/m1_sft/final"; M1_MERGED="$WS/runs/m1_sft/merged"; M3="$WS/runs/m3_dpo/final"
APPROVALS="$WS/training/approvals"
EXCLUSION_LIST="$WS/experiments/dataset_audit/lead/results/exclusion_list.json"
GPU_CUTOFF_IST="${CHAIN2_GPU_CUTOFF_IST:-2026-11-16 07:30}"   # all GPU work must end by then (deadline Mon 17:29 IST)
export PURE_TRL_MODE=1

skip() { echo "CHAIN2_SKIP $*"; exit 3; }
attn() { echo "CHAIN2_ATTENTION $*"; }
ist_now() { TZ=IST-5:30 date '+%Y-%m-%d %H:%M IST'; }
mins_until_ist() {   # mins_until_ist "YYYY-MM-DD HH:MM" (IST) -> whole minutes from now (negative when past)
  python3 -c 'import calendar, sys, time
t = calendar.timegm(time.strptime(sys.argv[1], "%Y-%m-%d %H:%M")) - 5.5 * 3600
print(int((t - time.time()) // 60))' "$1"
}
# settle: wait (<= 10 min) until MemAvailable is stable (two reads 20 s apart within 1 GB). vLLM's start-up memory profiling
# asserts when memory is released while it measures (the GRPO v2 smoke at 10:15 IST on 2026-11-15 started while the killed
# eval's 13.6 GB was still being freed: "Initial free memory 90.71 GiB, current free memory 96.19 GiB").
settle() {
  local a b i
  for i in $(seq 1 30); do
    a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); sleep 20
    b=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    if [ $((a > b ? a - b : b - a)) -le 1 ]; then echo "memory settled: ${b} GB available ($(ist_now))"; return 0; fi
    echo "memory still moving (${a} -> ${b} GB available); waiting"
  done
  echo "memory did not settle within 10 min; continuing"
}
# pipeline_v2 <env assignments...>: settle, then one run_pipeline_v2.sh invocation (failures surface: SOFT_STAGES empty)
pipeline_v2() { settle; env SOFT_STAGES="" PREFLIGHT_WAIT_MIN=30 "$@" bash training/run_pipeline_v2.sh; }
# guarded <name> <budget_gb> <cap_gb|0> <max_min> <alloc conf|-> <online|offline> -- cmd ...
# = run_pipeline.sh's memguard wrapper: own budget, 16 GB floor, one GPU job (the lock), refusal (75) retried 36 x every 300 s.
guarded() {
  local name=$1 budget=$2 cap=$3 mins=$4 alloc=$5 hub=$6; shift 6
  [ "${1:-}" = -- ] && shift
  local rc tries=0 envs=()
  [ "$hub" = offline ] && envs+=(HF_HUB_OFFLINE=1)
  while :; do
    settle
    echo "guarded $name: budget ${budget}G cap ${cap}G wall ${mins} min alloc ${alloc} hub ${hub} ($(ist_now))"
    if [ "$alloc" = - ]; then
      env -u PYTORCH_CUDA_ALLOC_CONF "${envs[@]}" python3 training/memguard.py --name "chain2-$name" --budget-gb "$budget" \
          --gpu-cap-gb "$cap" --max-minutes "$mins" --floor-gb 16 --summary "$CDIR/memguard/$name.json" -- "$@"
    else
      env PYTORCH_CUDA_ALLOC_CONF="$alloc" "${envs[@]}" python3 training/memguard.py --name "chain2-$name" --budget-gb "$budget" \
          --gpu-cap-gb "$cap" --max-minutes "$mins" --floor-gb 16 --summary "$CDIR/memguard/$name.json" -- "$@"
    fi
    rc=$?
    if [ "$rc" -eq 75 ] && [ "$tries" -lt 36 ]; then
      tries=$((tries + 1)); echo "memguard refused (75): retry $tries/36 in 300 s"; sleep 300; continue
    fi
    if [ -f "$CDIR/memguard/$name.json" ] && ! python3 -c 'import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("tree_gone") is not False else 1)' "$CDIR/memguard/$name.json"; then
      echo "CHAIN2_ATTENTION $name: memguard reports processes left behind (tree_gone false)"
    fi
    return "$rc"
  done
}
