#!/bin/bash
# Pre-approval PLUMBING check on the real core model (checklist 6.1: "allowed before approval, tiny subsets only"). NOT training:
# 1-3 optimizer steps per method on the longest rows, and every weight file is deleted at the end (logs + report kept).
# Stop-on-failure chain; EVERY stage runs under training/memguard.py (own process group, budget/floor kill, one GPU job at a time)
# because the 2026-11-14 smoke hung the shared GB10 host: the old chain ignored a failed stage and went on to the next, larger one.
#   stage0      mem_accounting_test.py: does MemAvailable see CUDA memory, does the torch cap raise OOM? (budget 15, cap 10)
#   sft         16 longest SFT rows, batch 1 x grad-accum 8 = 2 optimizer steps, 4 val rows               (budget 30, cap 26)
#   merge       merge_lora.py --min-effect 0 (smoke adapter); gate on rc 0 AND merge_info.json "ok": true (budget 25, cap 14)
#   grpo_ramp1  from the merged model: 1 prompt x 2 generations, 64-token completions, 1 step          (budget 65, cap 54)
#   grpo_ramp2  4 prompts x 4 generations, 3 steps (seconds/step for the real --max-steps)             (budget 65, cap 54)
#   dpo         8 longest pairs, batch 1 x grad-accum 4 = 2 steps (AlignTune backend)                   (budget 70, cap 60)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for stage0/sft/merge/dpo, NOT for GRPO (untested with colocated vLLM).
# No retries: a memguard kill (137) or refusal (75) or any other non-zero rc stops the chain ("SMOKE_STOP <stage> rc=<rc>").
# Report: /workspace/training/smoke_aya_report.json (per stage: rc, memguard summary, result.json without log_history, guards.json,
# merge_info / mem_accounting, vLLM memory lines for GRPO, the GRPO per-step metrics incl. train_grpo.py's wtm/* end-token
# fractions, and grpo_zero_gradient: true when every GRPO step had grad_norm 0 — the 2026-11-14 symptom; reported, not a stop).
# Do NOT wrap this script in memguard: every stage already runs under its own memguard. Run inside lx-exp: bash training/smoke_aya.sh
# Data: $SMOKE_DATA, default /workspace/training/data_v121_curated (the v1.2.1 views; the older data_v12_curated still holds the
# S14/S42 conflicts). The source path and the sha256 of its 4 files go into the report so the data version is on record.
# Refuses (75) while one of OUR vLLM servers is up: a PID recorded by serve_vllm.sh whose command line is a vLLM server, or a
# vLLM process in our own cgroup. Another project's server, visible if lx-exp shares the host PID namespace, does not block us.
set -uo pipefail
cd /workspace
export PURE_TRL_MODE=1 HF_HUB_OFFLINE=1
EXP=expandable_segments:True
REPORT=/workspace/training/smoke_aya_report.json
BASE=$(ls -d /root/.cache/huggingface/hub/models--1-800-LLMs--tiny-aya-global/snapshots/* 2>/dev/null | head -n 1)
SRC=${SMOKE_DATA:-/workspace/training/data_v121_curated}
D=/workspace/training/smoke_aya_data
R=/workspace/runs/smoke_aya
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
vllm_running() {
  local pids
  pids=$(our_vllm | sort -u | tr '\n' ' ')
  [ -n "$pids" ] || return 1
  echo "SMOKE_REFUSE our vLLM server is running (pids ${pids% }); stop it first (bash experiments/infra/stop_vllm.sh)"
}
vllm_running && exit 75
[ -n "$BASE" ] && [ -f "$BASE/config.json" ] || { echo "SMOKE_REFUSE Tiny Aya snapshot not found"; exit 1; }
for f in sft_train.jsonl sft_val.jsonl dpo_train.jsonl grpo_train.jsonl; do
  [ -s "$SRC/$f" ] || { echo "SMOKE_REFUSE $SRC/$f is missing or empty: build the v1.2.1 views there or set SMOKE_DATA"; exit 1; }
done

rm -rf "$D" "$R"; mkdir -p "$D" "$R"
python3 - "$SRC" "$D" > "$R/data_selection.json" <<'PY' || { echo "SMOKE_STOP data rc=1"; exit 1; }
import hashlib, json, sys
src, dst = sys.argv[1], sys.argv[2]
NAMES = (("sft_train.jsonl", 16), ("sft_val.jsonl", 4), ("dpo_train.jsonl", 8), ("grpo_train.jsonl", 8))
def longest(name, k):
    rows = [json.loads(l) for l in open(f"{src}/{name}") if l.strip()]
    rows.sort(key=lambda r: -len(json.dumps(r["prompt"], ensure_ascii=False)))
    with open(f"{dst}/{name}", "w") as fh:
        for r in rows[:k]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [(r.get("scenario_id"), r.get("level")) for r in rows[:k]]
print(json.dumps({"src": src, "src_sha256": {n: hashlib.sha256(open(f"{src}/{n}", "rb").read()).hexdigest() for n, _ in NAMES}}
                 | {n: longest(n, k) for n, k in NAMES}))
PY

STAGES=()
report() {   # report [stopped_at]
  python3 - "$R" "$REPORT" "${1:-}" "${STAGES[@]}" <<'PY'
import glob, json, os, re, sys
R, out, stopped, stages = sys.argv[1], sys.argv[2], sys.argv[3] or None, sys.argv[4:]
DIRS = {"stage0": "stage0", "sft": "sft", "merge": "sft_merged", "grpo_ramp1": "grpo_ramp1", "grpo_ramp2": "grpo_ramp2", "dpo": "dpo"}
VLLM = ("Model loading took", "Available KV cache memory", "KV cache size")
# Whether GRPO learned anything: the 2026-11-14 smoke had clipped_ratio 1, grad_norm 0 and loss 0 in every step.
LEARN = ("loss", "grad_norm", "learning_rate", "reward", "reward_std", "frac_reward_zero_std", "entropy",
         "completions/clipped_ratio", "completions/mean_length", "completions/mean_terminated_length")
# train_grpo.py turns TRL's truncation mask off (Tiny Aya also stops on <EOS_TOKEN> 3, which TRL counts as "clipped") and logs
# its own end-token fractions as wtm/*; sampling/* holds vLLM's importance-sampling ratios.
EXTRA_PREFIXES = ("wtm/", "sampling/")
def metric(txt, key):   # transformers 5 ProgressCallback prints floats as quoted '%.4g' strings, PrinterCallback as bare numbers
    return re.findall(rf"'{re.escape(key)}': '?([^',}}\s]+)", txt)[-3:]
def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None
rep = {"stopped_at": stopped, "data": load(os.path.join(R, "data_selection.json")), "stages": {}}
for st in stages:
    d = {"rc": int(open(os.path.join(R, f"{st}.rc")).read()) if os.path.exists(os.path.join(R, f"{st}.rc")) else None,
         "memguard": load(os.path.join(R, f"{st}.memguard.json"))}
    base = os.path.join(R, DIRS[st])
    res = load(os.path.join(base, "result.json"))
    if res is not None:
        d["result"] = {k: v for k, v in res.items() if k != "log_history"}
        d["result"]["log_history_tail"] = (res.get("log_history") or [])[-3:]
        if st.startswith("grpo"):
            steps = [{k: v for k, v in h.items() if k in LEARN or k.startswith(EXTRA_PREFIXES)}
                     for h in res.get("log_history") or [] if "completions/clipped_ratio" in h]
            d["grpo_steps"] = steps
            # the real "learned nothing" test: every step's gradient was zero (clipped_ratio 1 alone is expected now: see EXTRA_PREFIXES)
            d["grpo_zero_gradient"] = bool(steps) and all(not float(s.get("grad_norm") or 0) for s in steps)
            d["grpo_truncated_frac"] = [s.get("wtm/truncated_frac") for s in steps]
    d["guards"] = load(os.path.join(base, "guards.json"))
    if st == "stage0":
        d["mem_accounting"] = load(os.path.join(base, "mem_accounting.json"))
    if st == "merge":   # the merged dir only appears when the check passed; a failed check leaves its info elsewhere
        infos = sorted(glob.glob(os.path.join(R, "sft_merged*", "merge_info.json")))
        d["merge_info"] = {p: load(p) for p in infos}
    log = os.path.join(R, f"{st}.log")
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        lines = txt.splitlines()
        d["errors"] = [l[:300] for l in lines if re.search(r"Error|GUARD_FAIL|MERGE_FAIL|Traceback|MEMGUARD_(KILL|REFUSE|LEFTOVER)", l)][-6:]
        d["gpu_cap_line"] = next((l[:200] for l in lines if l.startswith("GPU_CAP")), None)
        d["warnings_mismatch"] = len(re.findall(r"Mismatch between tokenized prompt", txt))
        if st.startswith("grpo"):
            d["vllm_memory"] = [l[-300:] for l in lines if any(k in l for k in VLLM)][:12]
            for k, key in (("clipped_ratio", "completions/clipped_ratio"), ("reward_mean", "reward"), ("reward_std", "reward_std"),
                           ("grad_norm", "grad_norm"), ("mean_terminated_length", "completions/mean_terminated_length")):
                d[k] = metric(txt, key)
    rep["stages"][st] = d
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rep, open(out, "w"), indent=1, default=str)
def brief(d):
    mg, r = d.get("memguard") or {}, d.get("result") or {}
    b = {"rc": d["rc"], "host_peak_used_gb": mg.get("peak_used_gb"), "kill": mg.get("kill_reason"),
         "peak_mem": r.get("peak_mem", r.get("peak_mem_gb")), "minutes": mg.get("minutes"), "errors": (d.get("errors") or [])[:2]}
    if "grpo_zero_gradient" in d:
        b["grpo_zero_gradient"] = d["grpo_zero_gradient"]
        b["grpo_truncated_frac"] = d["grpo_truncated_frac"]
    return b
print(json.dumps({"stopped_at": stopped} | {k: brief(v) for k, v in rep["stages"].items()}, indent=1, default=str))
PY
}
cleanup() {   # plumbing check only: delete every weight file; keep logs, guards, results and the report
  find "$R" \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.pth" \) -delete
  rm -rf "$R"/*/checkpoint-* "$R"/sft_merged*
}
stop() {   # stop <stage> <rc>
  echo "SMOKE_STOP $1 rc=$2"
  report "$1"
  cleanup
  exit "$2"
}
run() {   # run <stage> <budget_gb> <cap_gb> <max_minutes> <PYTORCH_CUDA_ALLOC_CONF or -> <cmd...>
  local stage="$1" budget="$2" cap="$3" mins="$4" conf="$5"; shift 5
  STAGES+=("$stage")
  vllm_running && { echo 75 > "$R/$stage.rc"; stop "$stage" 75; }
  local envs=(env -u PYTORCH_CUDA_ALLOC_CONF)
  [ "$conf" = - ] || envs=(env PYTORCH_CUDA_ALLOC_CONF="$conf")
  "${envs[@]}" python3 training/memguard.py --name "smoke-$stage" --budget-gb "$budget" --floor-gb 16 --gpu-cap-gb "$cap" \
      --max-minutes "$mins" --summary "$R/$stage.memguard.json" -- "$@" > "$R/$stage.log" 2>&1
  local rc=$?
  echo "$rc" > "$R/$stage.rc"
  echo "$stage rc=$rc"
  grep -E "GPU_CAP|GUARD_OK|GUARD_FAIL|MERGE|_DONE|MEMGUARD|MEM_ACCOUNTING" "$R/$stage.log" | cut -c1-400
  [ "$rc" -eq 0 ] || stop "$stage" "$rc"
}

run stage0 15 10 10 "$EXP" python3 training/mem_accounting_test.py --out "$R/stage0/mem_accounting.json"
python3 - "$R/stage0/mem_accounting.json" <<'PY' || stop stage0 1
import json, sys
v = json.load(open(sys.argv[1]))["verdict"]
bad = [k for k, ok in v.items() if not ok]
print("stage0 verdict:", v)
sys.exit(f"stage0: {bad} false: memguard (MemAvailable) or the torch cap cannot be trusted on this host" if bad else 0)
PY

run sft 30 26 45 "$EXP" python3 training/train_sft.py --model "$BASE" --data "$D" --out "$R/sft" --smoke --batch 1 \
    --grad-accum 8 --fresh

run merge 25 14 20 "$EXP" python3 training/merge_lora.py --base "$BASE" --adapter "$R/sft/final" --out "$R/sft_merged" \
    --min-effect 0
M="$R/sft_merged"
python3 - "$M" <<'PY' || stop merge 1
import json, os, sys
m = sys.argv[1]
info = json.load(open(os.path.join(m, "merge_info.json"))) if os.path.exists(os.path.join(m, "merge_info.json")) else {}
if info.get("ok") is not True or not os.path.exists(os.path.join(m, "config.json")):
    sys.exit(f"merge gate: merge_info ok={info.get('ok')!r}, config.json present={os.path.exists(os.path.join(m, 'config.json'))}")
PY

run grpo_ramp1 65 54 40 - python3 training/train_grpo.py --model "$M" --data "$D" --out "$R/grpo_ramp1" --smoke --batch 1 \
    --num-generations 2 --max-completion 64 --max-steps 1 --vllm-mem 0.20 --fresh
run grpo_ramp2 65 54 60 - python3 training/train_grpo.py --model "$M" --data "$D" --out "$R/grpo_ramp2" --smoke --max-steps 3 \
    --vllm-mem 0.20 --fresh

run dpo 70 60 60 "$EXP" python3 training/train_dpo.py --model "$M" --data "$D/dpo_train.jsonl" --out "$R/dpo" --smoke --batch 1 \
    --grad-accum 4 --fresh

report
cleanup
echo SMOKE_AYA_DONE
