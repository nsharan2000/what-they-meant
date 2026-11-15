#!/bin/bash
# HF-router open-model grid, run on the Spark (CPU/network only; coexists with the GPU chain). Resumable. Ends with OPEN_GRID_DONE.
cd /workspace
OPEN="hf:CohereLabs/tiny-aya-global:cohere hf:CohereLabs/tiny-aya-fire:cohere hf:CohereLabs/tiny-aya-water:cohere hf:CohereLabs/tiny-aya-earth:cohere hf:CohereLabs/aya-expanse-32b:cohere hf:CohereLabs/c4ai-command-r7b-12-2024:cohere hf:Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra hf:deepseek-ai/DeepSeek-V3.2:deepinfra hf:meta-llama/Llama-4-Scout-17B-16E-Instruct:deepinfra"
python3 -m wtm.run --scenarios dataset/scenarios/*.jsonl --models $OPEN --levels L0 L0r L1 L2 --prompts P0 --mode decision --workers 20 \
  --out experiments/replication/results/grid_open_P0.jsonl 2>&1 | grep -E "jobs|done \(|cost|Traceback|Error" | tail -3
echo OPEN_P0_DONE
python3 -m wtm.run --scenarios dataset/scenarios/*.jsonl --models $OPEN --levels L0r L2 --prompts P1 --mode decision --workers 20 \
  --out experiments/replication/results/grid_open_P1.jsonl 2>&1 | grep -E "jobs|done \(|cost|Traceback|Error" | tail -3
echo OPEN_GRID_DONE
