#!/bin/bash
# Hard-set lane (user request 2026-11-13). Run AFTER the v1.2 specs are final. Every Claude step is resumable and wrapped in
# until_done.sh (waits out the 5-hour usage windows). Outputs in $H (scratchpad) and experiments/hardset/results/.
#  1. panel on the mined pool (seed + variants v1.2) at L2 under P2: Claude Opus/Sonnet/Fable/Haiku + Qwen3-235B + DeepSeek-V3.2
#  2. judge queue (non-clear-cut rows) → Opus judge v3 → select (mined hard set) → failure patterns
#  3. round-2 hard pairs (Claude Opus, patterns brief) → oracle replay judged for label validity
#  4. panel on ALL generated hard pairs at L2 and L0r under P2 (+ P0 for the panel's Claude models) → judge → final selection
# Usage: bash experiments/hardset/lane_hard.sh <scratchpad dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
H="$1/hard"; mkdir -p "$H" experiments/hardset/results
U=experiments/infra/until_done.sh
L="$H/lane_hard.log"
CLAUDE="claude:opus claude:sonnet claude:fable claude:haiku"
OPEN="hf:Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra hf:deepseek-ai/DeepSeek-V3.2:deepinfra"
POOL="dataset/scenarios/*.jsonl dataset/scenarios_synth/*.jsonl"
J="python3 -m wtm.judge --workers 8"

# 1. mined pool (open models in parallel with the Claude panel; they use the HF router, not the Claude window)
python3 -m wtm.run --scenarios $POOL --models $OPEN --levels L2 --prompts P2 --mode decision --workers 16 \
  --out "$H/panel_mined_open.jsonl" >> "$H/panel_open.log" 2>&1 &
OPEN_PID=$!
W=12 $U "$L" python3 -m wtm.run --scenarios $POOL --models $CLAUDE --levels L2 --prompts P2 --mode decision --workers 12 \
  --out "$H/panel_mined_claude.jsonl"
wait $OPEN_PID
# 2. judge + select (mined)
python3 experiments/hardset/select_hard.py --rows "$H/panel_mined_*.jsonl" --queue "$H/q_mined.jsonl" >> "$L" 2>&1
$U "$L" $J --results "$H/q_mined.jsonl" --scenarios $POOL --out "$H/judge_mined.jsonl"
python3 experiments/hardset/select_hard.py --rows "$H/panel_mined_*.jsonl" --judge "$H/judge_mined.jsonl" \
  --labels experiments/replication/results/judge_seedlabels_v3.jsonl experiments/replication/results/judge_variantlabels_v3.jsonl \
  --select experiments/hardset/results/hardset_mined.json >> "$L" 2>&1
python3 experiments/hardset/patterns.py experiments/hardset/results/hardset_mined.json experiments/hardset/results/patterns.json >> "$L" 2>&1
# 3. round-2 generation with the mined patterns
$U "$L" python3 dataset/hard_variants.py --k 3 --model opus --workers 6 --patterns-file experiments/hardset/results/patterns.json
python3 experiments/replication/judge_planted.py --scenarios "dataset/scenarios_hard/*.jsonl" --level L2 --prompt P2 --policies oracle \
  --out "$H/hard_oracle.jsonl" >> "$L" 2>&1
$U "$L" $J --results "$H/hard_oracle.jsonl" --scenarios dataset/scenarios_hard/*.jsonl --out "$H/judge_hard_labels.jsonl"
# 4. panel on generated hard pairs (L2 + L0r under P2; P0 for the gap without any hint)
python3 -m wtm.run --scenarios dataset/scenarios_hard/*.jsonl --models $OPEN --levels L2 L0r --prompts P2 --mode decision --workers 16 \
  --out "$H/panel_hard_open.jsonl" >> "$H/panel_open.log" 2>&1 &
OPEN_PID=$!
$U "$L" python3 -m wtm.run --scenarios dataset/scenarios_hard/*.jsonl --models $CLAUDE --levels L2 L0r --prompts P2 --mode decision \
  --workers 12 --out "$H/panel_hard_claude.jsonl"
$U "$L" python3 -m wtm.run --scenarios dataset/scenarios_hard/*.jsonl --models claude:opus claude:sonnet --levels L2 --prompts P0 \
  --mode decision --workers 12 --out "$H/panel_hard_claude.jsonl"
wait $OPEN_PID
python3 experiments/hardset/select_hard.py --rows "$H/panel_hard_*.jsonl" --queue "$H/q_hard.jsonl" >> "$L" 2>&1
$U "$L" $J --results "$H/q_hard.jsonl" --scenarios dataset/scenarios_hard/*.jsonl --out "$H/judge_hard.jsonl"
python3 experiments/hardset/select_hard.py --rows "$H/panel_hard_*.jsonl" --judge "$H/judge_hard.jsonl" \
  --labels "$H/judge_hard_labels.jsonl" --select experiments/hardset/results/hardset_generated.json >> "$L" 2>&1
echo LANE_HARD_DONE
