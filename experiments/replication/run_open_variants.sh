#!/bin/bash
# Open-model grid on the synthetic variants (raises N for the realism study; base models are never trained on these). Ends OPEN_VARIANTS_DONE.
cd /workspace
KEY="hf:CohereLabs/tiny-aya-global:cohere hf:CohereLabs/aya-expanse-32b:cohere hf:Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra hf:deepseek-ai/DeepSeek-V3.2:deepinfra"
python3 -m wtm.run --scenarios dataset/scenarios_synth/*.jsonl --models $KEY --levels L0r L1 L2 L2s --prompts P1 --mode decision --workers 24 \
  --out experiments/replication/results/grid_open_variants.jsonl 2>&1 | grep -E "jobs|Traceback|Error" | tail -2
python3 -m wtm.run --scenarios dataset/scenarios_synth/*.jsonl --models $KEY --levels L0r L2 --prompts P0 --mode decision --workers 24 \
  --out experiments/replication/results/grid_open_variants.jsonl 2>&1 | grep -E "jobs|Traceback|Error" | tail -2
echo OPEN_VARIANTS_DONE
