#!/bin/bash
# realism-pressure audit lens: reruns every script in dependency order (CPU only, numpy + pandas, no network).
# Inputs are read-only; outputs go to results/. The hard-set panel episodes are read from the session scratchpad (see common.py).
set -euo pipefail
cd "$(dirname "$0")"
for s in rates pressure m1_levels m1_vs_base hint_crossrun discrimination unscripted consistency dup_records dup_mentions \
         examples target_invariance summary write_findings; do
  echo "== $s"; python3 -B "$s.py" > "results/_log_$s.txt" 2>&1 || { echo "FAILED $s"; tail -5 "results/_log_$s.txt"; exit 1; }
done
echo ALL_DONE
