#!/bin/bash
# Plumbing check for the M1 → M2/M3 serving path (no training, no data): two RANDOM LoRA adapters (PEFT default init, B scaled
# down) → merge the first (merge_lora.py logit check) → second adapter on the merged model → stack_lora.py (logit check of
# base + stacked vs merged-first + second) → vLLM serves base + the stacked rank-32 adapter and answers one chat request.
# Every GPU/memory-heavy step runs under training/memguard.py (budget 25 GB for the PEFT steps, 60 GB for the vLLM step; one
# guarded job at a time) and the chain stops at the first failure ("SMOKE_STOP <step> rc=<rc>"), because the shared GB10 host
# hangs instead of raising an OOM. The vLLM server is stopped by its PID file (stop_vllm.sh), never by name.
# Weights under /workspace/runs/smoke_stack are deleted at the end and on any failure; logs stay in
# /workspace/runs/smoke_stack_logs, and every merge_info.json / stack_info.json (also from <out>.tmp, where merge_lora/stack_lora
# leave a failed check's numbers) is copied there first: merge_delta vs adapter_effect is what a false MERGE/STACK_FAIL is read from.
# Refuses (75) while one of OUR vLLM servers is up (a PID recorded by serve_vllm.sh whose command line is a vLLM server, or a vLLM
# process in our own cgroup); another project's server, visible if lx-exp shares the host PID namespace, does not block us.
# Run inside lx-exp: bash training/smoke_stack.sh
set -uo pipefail
cd /workspace
export HF_HUB_OFFLINE=1 PURE_TRL_MODE=1
BASE=$(ls -d /root/.cache/huggingface/hub/models--1-800-LLMs--tiny-aya-global/snapshots/* 2>/dev/null | head -n 1)
R=/workspace/runs/smoke_stack
L=/workspace/runs/smoke_stack_logs
LOGS=/workspace/logs
our_vllm() {   # PIDs of our live vLLM processes (ps -o args reads /proc/<pid>/cmdline)
  local f p
  for f in "$LOGS"/vllm-*.pid; do
    p=$(cat "$f" 2>/dev/null)
    [[ "$p" =~ ^[0-9]+$ ]] && ps -o args= -p "$p" 2>/dev/null | grep -qE 'vllm serve|VLLM::' && echo "$p"
  done
  [ -r /proc/self/cgroup ] || return 0
  for p in $(pgrep -f 'vllm serve|VLLM::EngineCore' 2>/dev/null); do
    [ "$(cat "/proc/$p/cgroup" 2>/dev/null)" = "$(cat /proc/self/cgroup)" ] && echo "$p"
  done
  return 0
}
PIDS=$(our_vllm | sort -u | tr '\n' ' ')
if [ -n "$PIDS" ]; then
  echo "SMOKE_REFUSE our vLLM server is running (pids ${PIDS% }); stop it first (bash experiments/infra/stop_vllm.sh)"; exit 75
fi
[ -n "$BASE" ] && [ -f "$BASE/config.json" ] || { echo "SMOKE_REFUSE Tiny Aya snapshot not found"; exit 1; }
rm -rf "$R" "$L"; mkdir -p "$R" "$L"

step() {   # step <name> <budget_gb> <cap_gb, 0 = none> <PYTORCH_CUDA_ALLOC_CONF or -> <cmd...>
  local name="$1" budget="$2" cap="$3" conf="$4"; shift 4
  local envs=(env -u PYTORCH_CUDA_ALLOC_CONF)
  [ "$conf" = - ] || envs=(env PYTORCH_CUDA_ALLOC_CONF="$conf")
  "${envs[@]}" python3 training/memguard.py --name "smoke-stack-$name" --budget-gb "$budget" --floor-gb 16 --gpu-cap-gb "$cap" \
      --max-minutes 30 --summary "$L/$name.memguard.json" -- "$@" > "$L/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc"
  grep -E "RAND_ADAPTER|MERGE|STACK|VLLM_|READY|GPU_CAP|MEMGUARD" "$L/$name.log" | cut -c1-300
  [ "$rc" -eq 0 ] || fail "$name" "$rc"
}
keep_info() {   # copy the merge/stack check numbers into the logs dir, then delete the weights
  local f
  for f in "$R"/*/merge_info.json "$R"/*/stack_info.json; do
    [ -f "$f" ] && cp "$f" "$L/$(basename "$(dirname "$f")").$(basename "$f")"
  done
  rm -rf "$R"
}
fail() {   # fail <step> <rc>
  echo "SMOKE_STOP $1 rc=$2"
  keep_info
  exit "$2"
}
merged_ok() {   # merged_ok <dir>: merge_lora.py's verdict, not just its rc
  python3 - "$1" <<'PY'
import json, os, sys
m = sys.argv[1]
p = os.path.join(m, "merge_info.json")
ok = os.path.exists(p) and json.load(open(p)).get("ok") is True and os.path.exists(os.path.join(m, "config.json"))
sys.exit(0 if ok else f"merge gate failed for {m}")
PY
}

cat > "$R/rand_adapter.py" <<'PY'
import sys, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
model, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)
m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
p = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=T, init_lora_weights=False, task_type="CAUSAL_LM"))
with torch.no_grad():
    for n, w in p.named_parameters():
        if "lora_B" in n:
            w.mul_(0.02)
p.save_pretrained(out)
print("RAND_ADAPTER", out)
PY
cat > "$R/vllm_step.sh" <<'SH'
set -uo pipefail
trap 'bash experiments/infra/stop_vllm.sh smoke-stack' EXIT
bash experiments/infra/serve_vllm.sh "$1" smoke-stack 8001 16384 0.30 || { echo "VLLM_LORA_SERVE_FAILED"; exit 1; }
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"stk","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); ok="choices" in d; print("VLLM_LORA_REPLY", repr(d["choices"][0]["message"]["content"][:120]) if ok else ("VLLM_LORA_BAD_REPLY", d)); sys.exit(0 if ok else 1)'
SH

EXP=expandable_segments:True
step a1 25 0 "$EXP" python3 "$R/rand_adapter.py" "$BASE" "$R/a1" 1
step merge 25 14 "$EXP" python3 training/merge_lora.py --base "$BASE" --adapter "$R/a1" --out "$R/a1_merged"
merged_ok "$R/a1_merged" || fail merge 1
step a2 25 0 "$EXP" python3 "$R/rand_adapter.py" "$R/a1_merged" "$R/a2" 2
step stack 25 20 "$EXP" python3 training/stack_lora.py --base "$BASE" --first "$R/a1" --first-merged "$R/a1_merged" \
    --second "$R/a2" --out "$R/stacked"
export VLLM_EXTRA_ARGS="--enable-lora --max-lora-rank 64 --max-loras 4 --lora-modules stk=$R/stacked"
step vllm 60 0 - bash "$R/vllm_step.sh" "$BASE"
keep_info
echo SMOKE_STACK_DONE
