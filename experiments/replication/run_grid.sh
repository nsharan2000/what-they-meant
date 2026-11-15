#!/bin/bash
# Replication grid (preregistered in README.md). Resumable: re-running skips finished keys.
# Usage: bash experiments/replication/run_grid.sh <stage> [scenario_glob]
#   stage = open-core | claude-core | open-golden | claude-golden | controls | fable
# Open models via the HF router (HF credits) run locally; local vLLM models run on the Spark (see run_spark_models.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="$1"; SCN="${2:-dataset/scenarios/*.jsonl}"
R=experiments/replication/results; mkdir -p "$R"
SC=$(ls $SCN | tr '\n' ' ')
OPEN_ALL="hf:CohereLabs/tiny-aya-global:cohere hf:CohereLabs/tiny-aya-fire:cohere hf:CohereLabs/tiny-aya-water:cohere \
hf:CohereLabs/tiny-aya-earth:cohere hf:CohereLabs/aya-expanse-32b:cohere hf:CohereLabs/c4ai-command-r7b-12-2024:cohere \
hf:Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra hf:deepseek-ai/DeepSeek-V3.2:deepinfra hf:meta-llama/Llama-4-Scout-17B-16E-Instruct:deepinfra"
OPEN_KEY="hf:CohereLabs/tiny-aya-global:cohere hf:CohereLabs/aya-expanse-32b:cohere hf:Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra hf:deepseek-ai/DeepSeek-V3.2:deepinfra"
run() { python3 -m wtm.run --scenarios $SC --mode decision --workers "${W:-16}" "$@"; }
case "$STAGE" in
  open-core)     run --models $OPEN_ALL --levels L0 L0r L1 L2 --prompts P0 --out $R/grid_open_core.jsonl ;;
  claude-core)   W=12 run --models claude:haiku claude:sonnet claude:opus --levels L0 L0r L1 L2 --prompts P0 --out $R/grid_claude_core.jsonl ;;
  fable)         W=10 run --models claude:fable --levels L0r L2 --prompts P0 --out $R/grid_claude_fable.jsonl ;;
  open-golden)   run --models $OPEN_ALL --levels L0r L2 --prompts P1 --out $R/grid_open_golden.jsonl ;;
  claude-golden) W=12 run --models claude:haiku claude:opus --levels L0r L2 --prompts P1 --out $R/grid_claude_golden.jsonl ;;
  controls)      run --models $OPEN_KEY --levels L0p L2s L3 --prompts P0 --out $R/grid_open_controls.jsonl
                 W=12 run --models claude:haiku claude:opus --levels L0p L2s L3 --prompts P0 --out $R/grid_claude_controls.jsonl ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
