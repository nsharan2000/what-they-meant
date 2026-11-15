#!/bin/bash
# Claude arm of the replication grid — pure Claude Code agents (wtm/drivers/claude_pure.py). Runs on the Mac (local Claude login).
# Usage: bash experiments/replication/run_claude_grid.sh <stage> <out_dir>
#   stage 1 = Opus + Haiku, L0r & L2, P1 (clean realism test: same golden norms at both levels)
#   stage 2 = Opus + Haiku, L0r & L2, P0 (app prompt only)
#   stage 3 = Sonnet, L0r & L2, P0 + P1
#   stage 4 = Fable, L0r & L2, P1
#   stage 5 = Opus + Haiku, L0 & L1, P0 + P1 (full ladder / hint effect / decomposition)
# out_dir must be outside ~/Desktop (long local runs lose write access there — log 21:00); copy results into the repo afterwards.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAGE="$1"; OUT="$2"; mkdir -p "$OUT"
SC="dataset/frozen_replication_v1/bills.jsonl dataset/frozen_replication_v1/bizops.jsonl dataset/frozen_replication_v1/shopping.jsonl dataset/frozen_replication_v1/wallet.jsonl"
run() { python3 -m wtm.run --scenarios $SC --mode decision --workers "${W:-12}" "$@"; }
case "$STAGE" in
  1) run --models claude:opus claude:haiku --levels L0r L2 --prompts P1 --out "$OUT/grid_claude.jsonl" ;;
  2) run --models claude:opus claude:haiku --levels L0r L2 --prompts P0 --out "$OUT/grid_claude.jsonl" ;;
  3) run --models claude:sonnet --levels L0r L2 --prompts P0 P1 --out "$OUT/grid_claude.jsonl" ;;
  4) run --models claude:fable --levels L0r L2 --prompts P1 --out "$OUT/grid_claude.jsonl" ;;
  5) run --models claude:opus claude:haiku --levels L0 L1 --prompts P0 P1 --out "$OUT/grid_claude.jsonl" ;;
  *) echo "unknown stage"; exit 2 ;;
esac
echo "CLAUDE_STAGE_${STAGE}_DONE"
