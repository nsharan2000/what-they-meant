#!/bin/bash
# Hard-set lane v2 (2026-11-13 ~11:00). Cost-aware: Claude episodes at the realistic level cost ~$0.15 (Opus) because every scenario's
# long member history is a fresh prompt, and one 5-hour window was used up in about an hour. Panel for SELECTION = 5 big models
# (Claude Opus / Sonnet / Haiku + Qwen3-235B + DeepSeek-V3.2); Fable 5.2 (the most expensive) runs only on the final hard pairs.
# Every Claude step is resumable and wrapped in until_done.sh. Usage: bash experiments/hardset/lane_hard2.sh <scratchpad dir> [judge_pid]
set -uo pipefail
cd "$(dirname "$0")/../.."
H="$1/hard"; JPID="${2:-}"
U=experiments/infra/until_done.sh
L="$H/lane_hard2.log"
POOL="dataset/scenarios/*.jsonl dataset/scenarios_synth/*.jsonl"
HARD="dataset/scenarios_hard/*.jsonl"
J="python3 -m wtm.judge --workers 8"
SEL=experiments/hardset/select_hard.py
LABELS="experiments/replication/results/judge_seedlabels_v3.jsonl experiments/replication/results/judge_variantlabels_v3.jsonl"
# 1. complete the mined panel (Opus already done) + the few missing generated-hard rows
$U "$L" python3 -m wtm.run --scenarios $POOL --models claude:sonnet claude:haiku --levels L2 --prompts P2 --mode decision --workers 12 \
  --out "$H/panel_mined_claude.jsonl"
$U "$L" python3 -m wtm.run --scenarios $HARD --models claude:haiku --levels L2 L0r --prompts P2 --mode decision --workers 12 \
  --out "$H/panel_hard_claude.jsonl"
# 2. judge everything not clear-cut (wait for an already running judge on the same file first)
if [ -n "$JPID" ]; then while kill -0 "$JPID" 2>/dev/null; do sleep 30; done; fi
python3 $SEL --rows "$H/panel_mined_*.jsonl" --queue "$H/q_mined.jsonl" >> "$L" 2>&1
$U "$L" $J --results "$H/q_mined.jsonl" --scenarios $POOL --out "$H/judge_mined.jsonl"
python3 $SEL --rows "$H/panel_hard_*.jsonl" --queue "$H/q_hard.jsonl" >> "$L" 2>&1
$U "$L" $J --results "$H/q_hard.jsonl" --scenarios $HARD --out "$H/judge_hard.jsonl"
# 3. label validity of generated pairs: oracle replay at L2/P2 judged by Opus (must be judged correct AND label_valid)
python3 experiments/replication/judge_planted.py --scenarios "$HARD" --level L2 --prompt P2 --policies oracle --out "$H/hard_oracle.jsonl" >> "$L" 2>&1
$U "$L" $J --results "$H/hard_oracle.jsonl" --scenarios $HARD --out "$H/judge_hard_labels.jsonl"
# 4. selection: mined (5-model panel) and generated (5-model panel; Fable reported separately)
python3 $SEL --rows "$H/panel_mined_*.jsonl" --judge "$H/judge_mined.jsonl" --labels $LABELS --min-fail 3 \
  --select experiments/hardset/results/hardset_mined.json >> "$L" 2>&1
python3 - "$H" >> "$L" 2>&1 <<'PY'
import json, sys
H = sys.argv[1]
rows = [json.loads(l) for l in open(f"{H}/panel_hard_claude.jsonl")] + [json.loads(l) for l in open(f"{H}/panel_hard_open.jsonl")]
with open(f"{H}/panel_hard_5models.jsonl", "w") as fh:
    for r in rows:
        if r["model"] != "claude:fable":
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
PY
python3 $SEL --rows "$H/panel_hard_5models.jsonl" --judge "$H/judge_hard.jsonl" --labels "$H/judge_hard_labels.jsonl" --min-fail 3 \
  --select experiments/hardset/results/hardset_generated.json >> "$L" 2>&1
python3 $SEL --rows "$H/panel_hard_claude.jsonl" "$H/panel_hard_open.jsonl" --judge "$H/judge_hard.jsonl" --labels "$H/judge_hard_labels.jsonl" \
  --min-fail 4 --select experiments/hardset/results/hardset_generated_6models.json >> "$L" 2>&1
# 5. Fable on the selected mined hard pairs (L2/P2) for the final report
python3 - <<'PY' > "$H/mined_hard_ids.txt"
import json
r = json.load(open("experiments/hardset/results/hardset_mined.json"))
print(" ".join(f"{p}A {p}B" for p in r["hard_pairs"]))
PY
IDS=$(cat "$H/mined_hard_ids.txt")
if [ -n "$IDS" ]; then
  $U "$L" python3 -m wtm.run --scenarios $POOL --ids $IDS --models claude:fable --levels L2 --prompts P2 --mode decision --workers 8 \
    --out "$H/panel_mined_fable.jsonl"
fi
echo LANE_HARD2_DONE
