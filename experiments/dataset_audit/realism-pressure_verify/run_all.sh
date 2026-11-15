#!/usr/bin/env bash
# Verify pass for the realism-pressure lens. Read-only over the repo; writes only into this folder's results/.
# Order matters: v3 writes results/v3_m1_prompts.csv (used by v5, v9, v10); v1 writes results/v1_episode_flags.csv (used by v15);
# v6 writes results/v6_world_specs.csv (used by v10).
set -euo pipefail
cd "$(dirname "$0")"
for s in v1_unscripted v2_contrasts v3_m1 v4_evidence v5_dups v6_world v7_hint_crossrun v8_crossrun_specs v9_m1_vs_base \
         v10_extra v11_hardset_sensitivity v12_target_invariance v13_target_grounding v14_echo v15_rp2_sensitivity v16_misc v17_spec_dups; do
  echo "== $s"
  python3 -B "$s.py" > "results/_log_${s}.txt" 2>&1
done
