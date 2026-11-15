#!/bin/bash
# Extra open-model conditions under fixed norms (P1): L1 (decomposition) + controls L0p / L2s / L3, key models. Ends OPEN_EXTRA_DONE.
cd /workspace
KEY="hf:CohereLabs/tiny-aya-global:cohere hf:CohereLabs/aya-expanse-32b:cohere hf:Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra hf:deepseek-ai/DeepSeek-V3.2:deepinfra"
python3 -m wtm.run --scenarios dataset/scenarios/*.jsonl --models $KEY --levels L1 --prompts P1 --mode decision --workers 20 \
  --out experiments/replication/results/grid_open_P1.jsonl 2>&1 | grep -E "jobs|Traceback|Error" | tail -2
python3 -m wtm.run --scenarios dataset/scenarios/*.jsonl --models $KEY --levels L0p L2s L3 --prompts P1 --mode decision --workers 20 \
  --out experiments/replication/results/grid_open_controls.jsonl 2>&1 | grep -E "jobs|Traceback|Error" | tail -2
echo OPEN_EXTRA_DONE
