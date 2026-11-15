#!/bin/bash
# run_pipeline_v2.sh (2026-11-15) = run_pipeline.sh + env hooks for the remaining experiments (training/run_chain2.py); every
# hook's default reproduces run_pipeline.sh exactly, and the signatures of m1_sft, m1_merge, m3_dpo and audits_base are unchanged:
#   GRPO_LR (default 5e-6, passed as --lr) · GRPO_DATA_DIR (default $DATA: m2_grpo's --data) · GRPO_EXTRA (extra train_grpo.py
#   args, e.g. "--no-shuffle --reward-variant v2 --log-completions --save-steps 6") — the three enter sig_of m2_grpo only when
#   they differ from the defaults · M2_ADAPTER (default runs/m2_grpo/final: the GRPO adapter m2_stack stacks, e.g. a checkpoint
#   chosen on validation; part of sig_of m2_stack through $M2) · EXTRA_ARMS ("name=path ..." appended to avail_arms, so eval,
#   audits_tuned and summary include them; a dir with stack_info.json must say ok and verified; a missing dir is skipped) ·
#   EVAL_BUDGET (default 75: the eval stage's memguard budget in GB; 2026-11-15 the capability step — lm-eval log-likelihood
#   tasks make vLLM compute prompt log-probs over the 262K vocabulary, ~17 GB transient — was killed at 75.8 > 75 GB).
#
# run_pipeline.sh — the ONE self-driving script that runs every remaining GPU stage of "What They Meant" on the DGX Spark, in order,
# with no scheduled wake-up of the Claude conversation (user, 2026-11-14: nothing may auto-start the conversation; the box drives
# itself and the result is read later with one `cat`).
#
# WHY each stage runs under training/memguard.py with its own budget: GB10 has 121 GB of UNIFIED memory, CUDA allocations come out
# of host RAM and are invisible to RSS and the container cgroup (experiments/infra/results/stage0_mem_accounting.json), so an
# overrun does not raise a CUDA OOM — it starves the whole shared box. On 2026-11-14 an unguarded DPO stage of ours did exactly
# that for 2 h and the kernel OOM killer hit other tenants. memguard kills our job first (budget / floor), holds the one-GPU-job
# lock (/workspace/.gpu_job.lock) and exports WTM_GPU_CAP_GB (guards.apply_mem_cap → torch allocator cap). This script is NEVER
# wrapped in memguard itself (each stage is; a wrapped pipeline would hold the lock and every stage would refuse).
#
# Stages (fixed names, in order)          memguard budget / cap GB   alloc conf   hub      wall limit (min, env)
#   audits_base   safety_eval.py + alignment_audit.py, arms: base     60 / 50 each   expandable   online   AUDIT_BASE_MAX_MIN 240
#   m1_sft        train_sft.py --model BASE --epochs $SFT_EPOCHS       30 / 26        expandable   offline  300 x epochs + 60
#   m1_merge      merge_lora.py → runs/m1_sft/merged (gate: ok true)   25 / 14        expandable   offline  45
#   m2_grpo       train_grpo.py --model merged --max-steps $GRPO_STEPS 65 / 54        UNSET        offline  20 x steps + 60
#   m2_stack      stack_lora.py M1 + M2 → runs/m2_grpo/stacked         25 / 14        expandable   offline  45
#                 (if it fails ONLY because the GRPO adapter's effect is ≤ --min-effect: one retry with --min-effect 0, recorded
#                 in runs/pipeline/m2_stack_min_effect_retry.json; the tolerance check still applies)
#   m3_dpo        train_dpo.py --model BASE --epochs $DPO_EPOCHS       70 / 60        expandable   offline  360 x epochs + 120
#   eval          ONE memguard command: KEEP_VLLM=1 run_eval.sh (test split) → EVAL_SCENARIOS=hard_test run_eval.sh →
#                 capability_eval.py base m1 m2 m3 → stop_vllm.sh      75 / none      UNSET        online   EVAL_MAX_MIN 720
#                 then auditkit_eval.py on both result files (CPU only, precomputed scoring: outside memguard, failure recorded
#                 as AUDITKIT_FAIL but not a stop — it can be rerun offline from the JSONL at any time)
#   audits_tuned  safety_eval.py + alignment_audit.py, arms m1 m2 m3   60 / 50 each   expandable   online   AUDIT_TUNED_MAX_MIN 600
#                 safety: ONE file results/safety.json (base from audits_base + m1-m3; safety_eval.py skips arms already there
#                 with the same config_hash and compares against the base arm in the file); alignment: results/alignment_base.json,
#                 then results/alignment_tuned.json with --baseline-arm results/alignment_base.arms/base.json (its drift baseline)
#   summary       always runs (also on a stop): runs/pipeline/summary.json
# Arms: m1 = runs/m1_sft/final · m2 = runs/m2_grpo/stacked (M1+M2 as ONE LoRA on the base) · m3 = runs/m3_dpo/final (DPO from the base).
# Training stages set HF_HUB_OFFLINE=1 (the base is in the HF cache); audits and eval do not (benchmark data / judges / lm-eval
# datasets are downloaded). PURE_TRL_MODE=1 everywhere. The audits use plain HF generate (no vLLM), with expandable_segments as
# both scripts recommend (override: AUDIT_ALLOC_CONF=-  → unset); the eval stage serves vLLM, so it is unset there.
#
# Resume semantics: a finished stage leaves runs/pipeline/markers/<stage>.done (JSON with its config signature); a rerun skips it
# (STAGE_SKIP). A stage whose signature changed (e.g. GRPO_STEPS) runs again (STAGE_CONFIG_CHANGED), and a stage that runs
# invalidates the markers of the stages that consume its output (m1_sft → merge, grpo, stack, eval, audits_tuned; m2_grpo → stack,
# eval, audits_tuned; m3_dpo → eval, audits_tuned; …). A killed/failed stage has no marker: rerunning the script resumes it — the
# trainers resume from their own checkpoints (and print *_ALREADY_DONE without touching result.json when already finished),
# run_eval.sh skips finished episodes. Sub-steps of the audit stages (safety, alignment) have their own markers.
# Fresh vs resume (decided by the pipeline, logged): the trainers refuse to resume when their arguments changed (train_sft.py:
#   147-160, train_dpo.py:155-164: GUARD_FAIL). So m1_sft / m3_dpo pass --fresh by themselves (STAGE_AUTO_FRESH) when
#   runs/<stage>/args.json holds another model, data or epochs — i.e. after SFT_EPOCHS / DPO_EPOCHS changed, also when the
#   run with the new value was killed or refused before it started; the trainers move the old outputs to stale_*.
#   m2_grpo is tied to the CONTENT of the M1 it trains on: train_grpo.py checks only the --model path (RESUME_KEYS, :66-68), so
#   every fresh GRPO start records merge_info.json's adapter_sha256 (merge_lora.py:161) in runs/m2_grpo/pipeline_upstream.json,
#   and GRPO gets --fresh (GRPO_UPSTREAM_CHANGED; old run → runs/m2_grpo/stale-<run_id>) when that sha differs from the current
#   merged M1, when checkpoints exist without a record or predate it, or when a checkpoint is past GRPO_STEPS. Hence a retrained
#   M1 (FRESH=m1_sft, SFT_EPOCHS changed) ALWAYS means a fresh GRPO, never a resume of the old one. The m1_merge gate also
#   checks merge_info adapter_sha256 == sha256(runs/m1_sft/final/adapter_model.safetensors).
# Stop at the first failure (PIPELINE_STOP <stage> rc=<rc>): no retries, except (a) a memguard REFUSAL (exit 75 before the job
# started: lock busy or MemAvailable < budget + floor) is retried REFUSE_RETRIES times every REFUSE_WAIT_S s, and (b) the m2_stack
# min-effect retry above. A memguard kill (137) or processes left behind after a stage (tree_gone false) always stop the chain.
#
# Env (defaults): SFT_EPOCHS=2 GRPO_STEPS=40 DPO_EPOCHS=1 · START_AT=audits_base STOP_AFTER=summary (run a subset) ·
#   FORCE="<stages>|all" (ignore their markers) · FRESH="<m1_sft m2_grpo m3_dpo>" (pass --fresh: old checkpoints are renamed
#   stale_* by the trainer; set it only for the run that should restart, never for a resume; not needed after an epochs change
#   or an M1 retrain — see "Fresh vs resume") · SOFT_STAGES="<stages>" (a plain
#   failure there is recorded as STAGE_FAIL_SOFT and the chain continues; never for kills/leftovers/refusals; default none) ·
#   CAP_LIMIT_SCALE=1.0 (capability_eval.py --limit-scale) · REFUSE_RETRIES=6 REFUSE_WAIT_S=300 · FLOOR_GB=16 ·
#   PREFLIGHT_WAIT_MIN=0 (>0: wait up to that long for a running guarded job / our vLLM server instead of refusing; poll
#   PREFLIGHT_POLL_S=60) ·
#   *_MAX_MIN overrides (SFT_MAX_MIN, MERGE_MAX_MIN, GRPO_MAX_MIN, STACK_MAX_MIN, DPO_MAX_MIN, EVAL_MAX_MIN, AUDIT_BASE_MAX_MIN,
#   AUDIT_TUNED_MAX_MIN) · AUDIT_BUDGET=60 AUDIT_CAP=50 · DATA=/workspace/training/data_v121_curated · BASE_MODEL (default: the
#   Tiny Aya Global snapshot in the HF cache) · MG_EXTRA (extra memguard args, e.g. "--psi-full 40"; the offline tests pass
#   "--meminfo <file>") · WTM_WS=/workspace WTM_LOGS=$WTM_WS/logs (tests only).
# HF auth for the audits/eval downloads: HF_TOKEN in the container env or the token file in the container's HF cache (hf auth
# login inside lx-exp). The pipeline logs only WHICH source exists, never the token. allenai/wildguard is gated and not accepted:
# the audit scripts are expected to handle its 403 themselves.
#
# Refuses (exit 75) when another run_pipeline.sh runs, another guarded GPU job holds /workspace/.gpu_job.lock, or one of OUR vLLM
# servers runs (PID files of serve_vllm.sh or a vLLM process in our own cgroup; same scoped check as training/smoke_aya.sh —
# another project's server does not block us). Bad parameters / missing inputs: exit 2.
#
# Outputs: /workspace/logs/pipeline.log (one line per stage start / end / rc + key lines), /workspace/logs/pipeline.status (JSON:
# stage, state waiting|running|done|skipped|failed|soft_failed|refused|aborted|finished, ts, ts_ist, rc, pipeline_pid — if state is running
# but the pid is gone, the pipeline died hard), runs/pipeline/summary.json (every result.json without its log_history, memguard
# summaries, merge/stack checks, GRPO per-step learning signal, eval completeness lines, capability headlines, audit outputs),
# runs/pipeline/logs/<stage>[.<sub>].log (full output, one '##### … ATTEMPT' header per attempt), runs/pipeline/config.json.
# Markers printed: STAGE_START / STAGE_DONE / STAGE_SKIP / STAGE_FAIL / STAGE_FAIL_SOFT <stage>, PIPELINE_STOP, PIPELINE_DONE.
#
# How to run (from the laptop):   source connect.sh
#   dbg wtm_pipeline "cd /workspace && bash training/run_pipeline.sh"              # nohup'd inside lx-exp; stdout → logs/wtm_pipeline.log
#   dexec "cat /workspace/logs/pipeline.status; tail -n 30 /workspace/logs/pipeline.log"
#   dexec "cat /workspace/runs/pipeline/summary.json"
#   e.g. only the evaluation part:  dbg wtm_pipeline "cd /workspace && START_AT=eval bash training/run_pipeline.sh"
# (Do not name the dbg job "pipeline": dbg would truncate /workspace/logs/pipeline.log, which this script appends to.)
# Stop it: kill -TERM <pipeline_pid from pipeline.status> — the running stage's memguard kills its job tree, status → aborted;
# a rerun resumes. Do not kill memguard's child directly.
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
WS="${WTM_WS:-/workspace}"
LOGS="${WTM_LOGS:-$WS/logs}"
RUNS="$WS/runs"
PDIR="$RUNS/pipeline"
PLOG="$LOGS/pipeline.log"
PSTATUS="$LOGS/pipeline.status"
GPU_LOCK="$WS/.gpu_job.lock"
cd "$WS" || { echo "PIPELINE_REFUSE cannot cd to $WS"; exit 2; }
export PURE_TRL_MODE=1

# Training first (the critical path), audits last: nothing wakes the Claude session mid-run (user, 2026-11-14), so an audit
# failure must never stand in front of training. GRPO/stack/audits are soft: their failure skips only what depends on them.
ALL_STAGES="m1_sft m1_merge m2_grpo m2_stack m3_dpo eval audits_base audits_tuned summary"
SFT_EPOCHS="${SFT_EPOCHS:-2}"; GRPO_STEPS="${GRPO_STEPS:-40}"; DPO_EPOCHS="${DPO_EPOCHS:-1}"
START_AT="${START_AT:-m1_sft}"; STOP_AFTER="${STOP_AFTER:-summary}"
FORCE="${FORCE:-}"; FRESH="${FRESH:-}"; SOFT_STAGES="${SOFT_STAGES-m2_grpo m2_stack audits_base audits_tuned}"
CAP_LIMIT_SCALE="${CAP_LIMIT_SCALE:-1.0}"
REFUSE_RETRIES="${REFUSE_RETRIES:-36}"; REFUSE_WAIT_S="${REFUSE_WAIT_S:-300}"; FLOOR_GB="${FLOOR_GB:-16}"
AUDIT_BUDGET="${AUDIT_BUDGET:-60}"; AUDIT_CAP="${AUDIT_CAP:-50}";   # kept for config.json; per-sub values below
SAFETY_BUDGET="${SAFETY_BUDGET:-44}"; SAFETY_CAP="${SAFETY_CAP:-32}"; ALIGN_BUDGET="${ALIGN_BUDGET:-24}"; ALIGN_CAP="${ALIGN_CAP:-16}"; AUDIT_ALLOC_CONF="${AUDIT_ALLOC_CONF:-expandable_segments:True}"
MG_EXTRA="${MG_EXTRA:-}"
DATA="${DATA:-$WS/training/data_v121_curated}"
GRPO_LR="${GRPO_LR:-5e-6}"; GRPO_DATA_DIR="${GRPO_DATA_DIR:-$DATA}"; GRPO_EXTRA="${GRPO_EXTRA:-}"   # v2 hooks (header)
EXTRA_ARMS="${EXTRA_ARMS:-}"; export EXTRA_ARMS   # also read by the __eval_inner re-invocation of this script
EVAL_BUDGET="${EVAL_BUDGET:-75}"   # v2 hook: the eval stage's memguard budget (capability's prompt log-probs need ~17 GB more)
BASE="${BASE_MODEL:-$(ls -d /root/.cache/huggingface/hub/models--1-800-LLMs--tiny-aya-global/snapshots/* 2>/dev/null | head -n 1)}"
MP=$(sed -n 2p "$LOGS/mirror_paths.txt" 2>/dev/null)   # the verified mirror path run_eval.sh serves (its tokenizer for lm-eval)
EXP=expandable_segments:True

M1="$RUNS/m1_sft/final"; M1_MERGED="$RUNS/m1_sft/merged"
M2="${M2_ADAPTER:-$RUNS/m2_grpo/final}"; M2S="$RUNS/m2_grpo/stacked"
M3="$RUNS/m3_dpo/final"
RES=experiments/training_eval/results
EVAL_TEST="$RES/eval_test.jsonl"; EVAL_HARD="$RES/eval_hard_test.jsonl"
TEST_GLOB='dataset/eval/test_*.jsonl'; HARD_GLOB='dataset/eval/hard_test_*.jsonl'
CAP_OUT="$RES/capability.json"
AK_TEST="$RES/auditkit_test.json"; AK_HARD="$RES/auditkit_hard_test.json"
SAFETY_OUT="$RES/safety.json"   # base (audits_base) + m1-m3 (audits_tuned) in one file: safety_eval.py compares against its base arm
ALIGN_BASE_OUT="$RES/alignment_base.json"; ALIGN_TUNED_OUT="$RES/alignment_tuned.json"
BASE_SERVED=tiny-aya-global-local; BASE_SPEC="vllm:$BASE_SERVED@http://localhost:8001/v1"   # = run_eval.sh BASE_NAME / URL

mins_of() { awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN { printf "%d", a * b + c }'; }
SFT_MAX_MIN="${SFT_MAX_MIN:-$(mins_of 300 "$SFT_EPOCHS" 60)}"
MERGE_MAX_MIN="${MERGE_MAX_MIN:-45}"
GRPO_MAX_MIN="${GRPO_MAX_MIN:-$(mins_of 20 "$GRPO_STEPS" 60)}"
STACK_MAX_MIN="${STACK_MAX_MIN:-45}"
DPO_MAX_MIN="${DPO_MAX_MIN:-$(mins_of 360 "$DPO_EPOCHS" 120)}"
EVAL_MAX_MIN="${EVAL_MAX_MIN:-720}"
AUDIT_BASE_MAX_MIN="${AUDIT_BASE_MAX_MIN:-240}"; AUDIT_TUNED_MAX_MIN="${AUDIT_TUNED_MAX_MIN:-600}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
ts_ist() { TZ=IST-5:30 date '+%H:%M IST'; }   # POSIX TZ string: works without tzdata (the container clock is UTC)

avail_arms() {   # the tuned arms that exist (m2 only when stack_lora verified it): a soft GRPO/stack failure leaves m1 + m3
  local a=()
  [ -f "$M1/adapter_config.json" ] && a+=("m1=$M1")
  if [ -f "$M2S/stack_info.json" ] && python3 -c 'import json, sys
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("ok") is True and d.get("verified") is True else 1)' "$M2S/stack_info.json" 2>/dev/null; then
    a+=("m2=$M2S")
  fi
  [ -f "$M3/adapter_config.json" ] && a+=("m3=$M3")
  local x xn xp
  for x in $EXTRA_ARMS; do   # v2: extra arms (name=path); stacked ones must be verified like m2
    xn="${x%%=*}"; xp="${x#*=}"
    [ -f "$xp/adapter_config.json" ] || continue
    if [ -f "$xp/stack_info.json" ] && ! python3 -c 'import json, sys
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("ok") is True and d.get("verified") is True else 1)' "$xp/stack_info.json" 2>/dev/null; then
      continue
    fi
    a+=("$xn=$xp")
  done
  echo "${a[*]}"
}
arm_labels() { local x; for x in $1; do printf '%s ' "${x%%=*}"; done; }

# ================================== eval stage body (runs INSIDE the eval stage's memguard) ==================================
# One guarded command because run_eval.sh's KEEP_VLLM=1 server lives only as long as memguard's command (memguard's exit sweep
# kills its whole tree); capability_eval.py reuses that server. run_eval.sh accepts memguard's lock because the memguard.py named
# in the lock file is its ancestor (EVAL_UNDER_MEMGUARD).
if [ "${1:-}" = __eval_inner ]; then
  rc=0
  ARMS=$(avail_arms); LABELS=$(arm_labels "$ARMS")
  [ -n "$ARMS" ] || { echo "EVAL_NO_ARMS no trained adapter exists"; exit 66; }
  echo "EVAL_ARMS $ARMS"
  echo "EVAL_STEP test $(ts)"
  # shellcheck disable=SC2086  # ARMS is a list of name=path words (paths have no spaces)
  EVAL_SCENARIOS="$TEST_GLOB" KEEP_VLLM=1 bash experiments/training_eval/run_eval.sh "$EVAL_TEST" $ARMS || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "EVAL_STEP hard_test $(ts)"
    # shellcheck disable=SC2086
    EVAL_SCENARIOS="$HARD_GLOB" KEEP_VLLM=1 bash experiments/training_eval/run_eval.sh "$EVAL_HARD" $ARMS || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    echo "EVAL_STEP capability $(ts)"
    # the LAST block of the adapters.txt of the server that is up now names what it serves (content-hashed names)
    # shellcheck disable=SC2086
    python3 experiments/training_eval/capability_eval.py --arms "base=$BASE_SERVED" $LABELS \
        --adapters-txt "${EVAL_HARD%.jsonl}.adapters.txt" --tokenizer "$MP" --out "$CAP_OUT" \
        --limit-scale "$CAP_LIMIT_SCALE" || rc=$?
  fi
  echo "EVAL_STEP stop_vllm $(ts)"
  bash experiments/infra/stop_vllm.sh || { [ "$rc" -ne 0 ] || rc=1; }
  echo "EVAL_INNER_RC $rc"
  exit "$rc"
fi

# ================================================= main: helpers =================================================
mkdir -p "$LOGS" "$PDIR/markers/stale" "$PDIR/logs" || { echo "PIPELINE_REFUSE cannot create $LOGS / $PDIR"; exit 2; }
STDOUT_IS_PLOG=0   # started as `> /workspace/logs/pipeline.log` (e.g. dbg pipeline …): write each line once
[ -e /proc/$$/fd/1 ] && [ "$(readlink -f /proc/$$/fd/1 2>/dev/null)" = "$(readlink -f "$PLOG" 2>/dev/null)" ] && STDOUT_IS_PLOG=1
plog() {
  local line; line="$(ts) $*"
  echo "$line"
  [ "$STDOUT_IS_PLOG" = 1 ] || echo "$line" >> "$PLOG" 2>/dev/null || true
}
plog_file() { local l; [ -f "$1" ] || return 0; while IFS= read -r l; do plog "  $l"; done < "$1"; }
status() {   # status <stage> <state> [rc] [detail]
  python3 - "$PSTATUS" "$1" "$2" "${3:-}" "${4:-}" "$$" "$PDIR/summary.json" "$PLOG" <<'PY' >/dev/null 2>&1 || true
import json, os, sys, time
p, stage, state, rc, detail, pid, summ, plog = sys.argv[1:9]
d = {"stage": stage, "state": state, "ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
     "ts_ist": time.strftime("%Y-%m-%d %H:%M IST", time.gmtime(time.time() + 5.5 * 3600)),
     "rc": int(rc) if rc.lstrip("-").isdigit() else None, "detail": detail or None, "pipeline_pid": int(pid),
     "log": plog, "summary": summ}
with open(p + ".tmp", "w") as fh:
    json.dump(d, fh)
os.replace(p + ".tmp", p)
PY
}
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }
stage_idx() { local i=0 s; for s in $ALL_STAGES; do [ "$s" = "$1" ] && { echo "$i"; return 0; }; i=$((i + 1)); done; return 1; }
deps_of() {   # stages that consume this stage's output: their markers are invalidated when it runs again
  case "$1" in
    m1_sft) echo "m1_merge m2_grpo m2_stack eval audits_tuned" ;;
    m1_merge) echo "m2_grpo m2_stack eval audits_tuned" ;;
    m2_grpo) echo "m2_stack eval audits_tuned" ;;
    m2_stack|m3_dpo) echo "eval audits_tuned" ;;
    audits_base) echo "audits_tuned" ;;   # its base arms are audits_tuned's safety comparison and alignment drift baseline
    *) echo "" ;;
  esac
}
sig_of() {   # config signature stored in the marker; a change re-runs the stage
  case "$1" in
    audits_base) echo "safety_eval+alignment_audit base=$BASE arms=base safety=$SAFETY_OUT align=$ALIGN_BASE_OUT" ;;
    m1_sft) echo "train_sft model=$BASE data=$DATA epochs=$SFT_EPOCHS" ;;
    m1_merge) echo "merge_lora base=$BASE adapter=$M1 out=$M1_MERGED" ;;
    m2_grpo) local v2=""   # v2 hooks enter the signature only when they differ from the v1 defaults
      [ "$GRPO_LR" != 5e-6 ] && v2+=" lr=$GRPO_LR"
      [ "$GRPO_DATA_DIR" != "$DATA" ] && v2+=" grpo_data=$GRPO_DATA_DIR"
      [ -n "$GRPO_EXTRA" ] && v2+=" extra=$GRPO_EXTRA"
      echo "train_grpo model=$M1_MERGED m1_adapter_sha256=$(m1_merged_sha) data=$DATA max_steps=$GRPO_STEPS$v2" ;;
    m2_stack) echo "stack_lora base=$BASE first=$M1 first_merged=$M1_MERGED second=$M2 out=$M2S" ;;
    m3_dpo) echo "train_dpo model=$BASE data=$DATA/dpo_train.jsonl epochs=$DPO_EPOCHS" ;;
    eval) echo "run_eval test+hard_test arms=$(avail_arms); capability limit_scale=$CAP_LIMIT_SCALE; auditkit" ;;
    audits_tuned) echo "safety_eval+alignment_audit base=$BASE arms=$(avail_arms) safety=$SAFETY_OUT align=$ALIGN_TUNED_OUT" ;;
    *) echo "$1" ;;
  esac
}
marker_sig() {   # marker_sig <marker file> → its stored signature (empty if unreadable)
  python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1])).get("sig", ""))
except Exception:
    print("")' "$1" 2>/dev/null
}
write_marker() {   # write_marker <file> <stage> <sig> <seconds> [extra json]
  python3 - "$@" <<'PY'
import json, os, sys, time
p, stage, sig, secs = sys.argv[1:5]
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}
d = {"stage": stage, "sig": sig, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "minutes": round(int(secs) / 60, 1)} | extra
with open(p + ".tmp", "w") as fh:
    json.dump(d, fh, indent=1)
os.replace(p + ".tmp", p)
PY
}
stale_marker() { [ -f "$1" ] && mv -f "$1" "$PDIR/markers/stale/$(basename "$1").$(date +%Y%m%d-%H%M%S)"; return 0; }
json_get() {   # json_get <file> <key> → that top-level value (empty if the file or key is missing/unreadable)
  python3 -c 'import json, sys
try:
    v = json.load(open(sys.argv[1])).get(sys.argv[2])
    print("" if v is None else v)
except Exception:
    print("")' "$1" "$2" 2>/dev/null
}
# CONTENT identity of the M1 that the merged dir was built from (merge_lora.py:161 writes adapter_sha256 of the adapter it
# merged). GRPO trains on the merged dir, but train_grpo.py's resume check compares the --model PATH (RESUME_KEYS,
# train_grpo.py:66-68), not its content — so the pipeline itself must tie M2 to the M1 it was trained on.
m1_merged_sha() { json_get "$M1_MERGED/merge_info.json" adapter_sha256; }
trainer_args_changed() {   # trainer_args_changed <out> <model> <data> <epochs>: prints why <out>/args.json differs, rc 0 if so
  # train_sft.py:147-160 / train_dpo.py:155-164 refuse (GUARD_FAIL) to resume when any compared argument differs from
  # <out>/args.json; the pipeline only varies model, data and epochs, so a difference there means a resume is impossible and the
  # stage must start with --fresh (the trainers set the old outputs aside as stale_*: train_sft.py:170-174, train_dpo.py:174-177).
  python3 - "$@" <<'PY'
import json, os, sys
out, model, data, epochs = sys.argv[1:5]
try:
    a = json.load(open(os.path.join(out, "args.json")))
except Exception:
    sys.exit(1)   # no previous run (or no args.json: the trainer itself explains what to do)
rp = lambda p: os.path.realpath(p) if isinstance(p, str) and os.path.exists(p) else p
why = []
try:
    if float(a.get("epochs")) != float(epochs):
        why.append(f"epochs {a.get('epochs')} -> {epochs}")
except (TypeError, ValueError):
    why.append(f"epochs {a.get('epochs')!r} -> {epochs}")
for k, v in (("model", model), ("data", data)):
    if rp(a.get(k)) != rp(v):
        why.append(f"{k} {a.get(k)} -> {v}")
if not why:
    sys.exit(1)
print("; ".join(why))
PY
}
grpo_fresh_reason() {   # grpo_fresh_reason <current M1 sha>: prints why the GRPO run dir must NOT be resumed (empty: resume is safe)
  python3 - "$RUNS/m2_grpo" "$1" "$GRPO_STEPS" <<'PY'
import json, os, re, sys
out, cur, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not os.path.isdir(out):
    sys.exit(0)
ck = [int(m.group(1)) for d in os.listdir(out) if (m := re.match(r"^checkpoint-(\d+)$", d))]
if not ck and not os.path.exists(os.path.join(out, "final")):
    sys.exit(0)   # nothing train_grpo.py could resume from
try:
    up = json.load(open(os.path.join(out, "pipeline_upstream.json")))
    rec, at = up.get("m1_adapter_sha256"), float(up.get("recorded_epoch") or 0)
except Exception:
    rec, at = None, 0.0
# The record is written right before every fresh start, so every checkpoint of that run is younger than it. An older one
# survived a fresh start that never ran (memguard refused, or a kill before train_grpo set the old run aside).
older = sorted(d for d in os.listdir(out) if (re.match(r"^checkpoint-\d+$", d) or d == "final")
               and os.path.getmtime(os.path.join(out, d)) < at)
if not rec:
    print("GRPO checkpoints exist but pipeline_upstream.json does not record which M1 they were trained on")
elif older:
    print(f"{', '.join(older)} predate the last fresh start recorded in pipeline_upstream.json (that start never ran)")
elif rec != cur:
    print(f"M1 changed: the GRPO checkpoints were trained on merged M1 adapter sha256 {rec[:12]}, the merged dir now holds {cur[:12]}")
elif ck and max(ck) > steps:
    # train_grpo.py:150-155 would print GRPO_ALREADY_DONE and keep a final adapter trained for more steps than asked
    print(f"checkpoint-{max(ck)} is past GRPO_STEPS={steps}")
PY
}

our_vllm() {   # PIDs of our live vLLM processes (same scoped check as training/smoke_aya.sh)
  local f p
  for f in "$LOGS"/vllm-*.pid; do
    [ -f "$f" ] || continue
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
  plog "PIPELINE_REFUSE our vLLM server is running (pids ${pids% }); stop it first (bash experiments/infra/stop_vllm.sh)"
}
gpu_lock_holder() {   # prints the holder and returns 1 when another guarded job holds the GPU lock
  python3 - "$GPU_LOCK" <<'PY'
import fcntl, sys
fh = open(sys.argv[1], "a+")
try:
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    fh.seek(0)
    print(fh.read().strip() or "unknown holder")
    sys.exit(1)
fcntl.flock(fh, fcntl.LOCK_UN)
PY
}

# ---- summary (also written on a stop / abort) ----
STAGE_STATES=""
write_summary() {   # write_summary <stopped_at or ""> <rc>
  STAGE_STATES="$STAGE_STATES" PDIR="$PDIR" WS="$WS" RES="$RES" RUNS="$RUNS" EVAL_TEST="$EVAL_TEST" EVAL_HARD="$EVAL_HARD" \
  CAP_OUT="$CAP_OUT" AK_TEST="$AK_TEST" AK_HARD="$AK_HARD" ALL_STAGES="$ALL_STAGES" SAFETY_OUT="$SAFETY_OUT" \
  ALIGN_BASE_OUT="$ALIGN_BASE_OUT" ALIGN_TUNED_OUT="$ALIGN_TUNED_OUT" \
  python3 - "${1:-}" "${2:-0}" <<'PY'
import glob, json, os, re, sys, time
E = os.environ
P, WS, RUNS = E["PDIR"], E["WS"], E["RUNS"]
stopped, rc = sys.argv[1] or None, int(sys.argv[2])


def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def ws(p):
    return p if os.path.isabs(p) else os.path.join(WS, p)


def last_attempt(path):
    """Text of the last '##### … ATTEMPT' block of a stage log (every attempt is appended to the same file)."""
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return ""
    i = txt.rfind("\n##### ")
    return txt[i + 1:] if i >= 0 else txt


def strip_history(res):
    if not isinstance(res, dict):
        return res
    out = {k: v for k, v in res.items() if k != "log_history"}
    out["log_history_tail"] = (res.get("log_history") or [])[-3:]
    return out


def small_json(p, limit=200_000):
    """The whole JSON when small, else its top-level keys (the file path is always given)."""
    d = {"path": p, "exists": os.path.exists(ws(p))}
    if d["exists"]:
        d["bytes"] = os.path.getsize(ws(p))
        j = load(ws(p))
        if j is None:
            d["error"] = "unreadable JSON"
        elif d["bytes"] <= limit:
            d["content"] = j
        else:
            d["top_level_keys"] = sorted(j) if isinstance(j, dict) else type(j).__name__
            for k in ("summary", "headline", "verdict", "drift", "compare"):
                if isinstance(j, dict) and k in j:
                    d[k] = j[k]
    return d


def errors(txt):
    return [l[:300] for l in txt.splitlines()
            if re.search(r"Traceback|Error|_FAIL|GUARD_FAIL|MEMGUARD_(KILL|REFUSE|LEFTOVER|TERM)|ARM_|REFUSE", l)][-8:]


states = dict(s.split("=", 1) for s in E.get("STAGE_STATES", "").split() if "=" in s)
rep = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "stopped_at": stopped, "rc": rc,
       "finished": stopped is None and rc == 0, "soft_failed": sorted(k for k, v in states.items() if v == "soft_failed"), "config": load(os.path.join(P, "config.json")), "stages": {}}
for st in E["ALL_STAGES"].split():
    d = {"this_run": states.get(st, "not_run"), "marker": load(os.path.join(P, "markers", f"{st}.done"))}
    mg = {}
    for p in sorted(glob.glob(os.path.join(P, f"{st}.memguard.json")) + glob.glob(os.path.join(P, f"{st}.*.memguard.json"))):
        mg[os.path.basename(p)[:-len(".memguard.json")]] = load(p)
    if mg:
        d["memguard"] = mg
    subs = sorted(glob.glob(os.path.join(P, "markers", f"{st}.*.done")))
    if subs:
        d["sub_markers"] = {os.path.basename(p)[:-5]: load(p) for p in subs}
    logs = sorted(p for p in glob.glob(os.path.join(P, "logs", f"{st}.log")) + glob.glob(os.path.join(P, "logs", f"{st}.*.log"))
                  if not p.endswith(".last.log"))
    if logs:
        d["logs"] = logs
        errs = {os.path.basename(l): errors(last_attempt(l)) for l in logs}
        if any(errs.values()):
            d["errors_last_attempt"] = {k: v for k, v in errs.items() if v}
    rep["stages"][st] = d

for st, sub in (("m1_sft", "m1_sft"), ("m2_grpo", "m2_grpo"), ("m3_dpo", "m3_dpo")):
    d = rep["stages"][st]
    res = load(os.path.join(RUNS, sub, "result.json"))
    d["result"] = strip_history(res)
    d["guards"] = load(os.path.join(RUNS, sub, "guards.json"))
    if st == "m2_grpo" and isinstance(res, dict):
        # did GRPO learn anything? (the 2026-11-14 smoke: grad_norm 0 in every step). Reported, not a stop.
        keys = ("loss", "grad_norm", "reward", "reward_std", "frac_reward_zero_std", "entropy", "completions/mean_length")
        steps = [{k: v for k, v in h.items() if k in keys or k.startswith("wtm/")}
                 for h in res.get("log_history") or [] if "reward" in h or "completions/clipped_ratio" in h]
        d["grpo_steps"] = steps
        d["grpo_zero_gradient"] = bool(steps) and all(not float(s.get("grad_norm") or 0) for s in steps)
        d["grpo_truncated_frac"] = [s.get("wtm/truncated_frac") for s in steps]
rep["stages"]["m1_merge"]["merge_info"] = {p: load(p) for p in
                                           glob.glob(os.path.join(RUNS, "m1_sft", "merged*", "merge_info.json"))}
rep["stages"]["m2_stack"]["stack_info"] = {p: load(p) for p in
                                           glob.glob(os.path.join(RUNS, "m2_grpo", "stacked*", "stack_info.json"))}
rep["stages"]["m2_stack"]["min_effect_retry"] = load(os.path.join(P, "m2_stack_min_effect_retry.json"))
# which merged M1 the GRPO run was trained on (written at every fresh GRPO start) vs the merged M1 now: they must match
up = load(os.path.join(RUNS, "m2_grpo", "pipeline_upstream.json"))
mi = load(os.path.join(RUNS, "m1_sft", "merged", "merge_info.json"))
rep["stages"]["m2_grpo"]["upstream"] = {"record": up, "merged_m1_adapter_sha256_now": (mi or {}).get("adapter_sha256"),
                                        "matches": isinstance(up, dict) and isinstance(mi, dict)
                                        and up.get("m1_adapter_sha256") == mi.get("adapter_sha256")}

ev = rep["stages"]["eval"]
txt = last_attempt(os.path.join(P, "logs", "eval.log"))
section, ev["completeness"], ev["eval_done"] = None, {}, {}
for l in txt.splitlines():   # the eval body prints 'EVAL_STEP <test|hard_test|capability|stop_vllm> <ts>' before each step
    if l.startswith("EVAL_STEP "):
        section = l.split()[1]
    elif re.match(r"\s*(EVAL_COMPLETE|EVAL_INCOMPLETE|EVAL_RUN_FAILED)", l):
        ev["completeness"].setdefault(section, []).append(l.strip())
    elif l.strip() == "EVAL_DONE":
        ev["eval_done"][section] = True
ev["adapters_served"] = [l.strip() for l in txt.splitlines() if l.startswith("ADAPTER ")]
ev["capability_done"] = "CAPABILITY_DONE" in txt
ev["results"] = {}
for p in (E["EVAL_TEST"], E["EVAL_HARD"]):
    q = ws(p)
    ev["results"][p] = {"exists": os.path.exists(q), "rows": sum(1 for l in open(q) if l.strip()) if os.path.exists(q) else 0,
                        "adapters_txt": p[:-len(".jsonl")] + ".adapters.txt"}
cap = load(ws(E["CAP_OUT"]))
if isinstance(cap, dict):
    arms = cap.get("arms") or {}
    ev["capability"] = {"path": E["CAP_OUT"], "arms": arms, "compare": cap.get("compare"),
                        "task_errors": sum(1 for a in arms.values() for t in a.values() if isinstance(t, dict) and "error" in t)}
ev["auditkit"] = {}
for name, p in (("test", E["AK_TEST"]), ("hard_test", E["AK_HARD"])):
    j = load(ws(p))
    rcf = os.path.join(P, f"eval.auditkit_{name}.rc")
    ev["auditkit"][name] = {"path": p, "exists": j is not None,
                            "rc": int(open(rcf).read().strip()) if os.path.exists(rcf) else None,
                            "cells": len(j.get("cells", [])) if isinstance(j, dict) else None,
                            "comparisons": len(j.get("base_vs_tuned", [])) if isinstance(j, dict) else None}
rep["stages"]["audits_base"]["outputs"] = {"alignment": small_json(E["ALIGN_BASE_OUT"])}
rep["stages"]["audits_tuned"]["outputs"] = {"alignment": small_json(E["ALIGN_TUNED_OUT"])}
rep["safety"] = small_json(E["SAFETY_OUT"])   # one file for both audit stages (arms base, m1, m2, m3 + comparisons vs base)

os.makedirs(P, exist_ok=True)
out = os.path.join(P, "summary.json")
with open(out + ".tmp", "w") as fh:
    json.dump(rep, fh, indent=1, default=str)
os.replace(out + ".tmp", out)
brief = {st: (d["this_run"], (d.get("marker") or {}).get("minutes")) for st, d in rep["stages"].items()}
print("SUMMARY " + json.dumps({"stopped_at": stopped, "rc": rc, "stages": brief,
                               "grpo_zero_gradient": rep["stages"]["m2_grpo"].get("grpo_zero_gradient"),
                               "eval_completeness": ev["completeness"]}, default=str))
PY
}

# ---- stopping ----
CUR_STAGE=""; MGPID=""
finish() {   # finish <stage or ""> <rc> <state>: summary + status, then exit
  local stage="$1" rc="$2" state="$3"
  write_summary "$stage" "$rc" 2>&1 | tail -n 1 | while read -r l; do plog "$l"; done
  status "${stage:-summary}" "$state" "$rc" "$PDIR/summary.json"
  exit "$rc"
}
on_abort() {
  plog "PIPELINE_ABORT signal $1 during ${CUR_STAGE:-no stage}"
  if [ -n "$MGPID" ] && kill -0 "$MGPID" 2>/dev/null; then
    kill -TERM "$MGPID" 2>/dev/null   # memguard SIGKILLs its job tree, waits until it is gone, then exits 137
    wait "$MGPID" 2>/dev/null
  fi
  STAGE_STATES="$STAGE_STATES ${CUR_STAGE:-none}=aborted"
  finish "${CUR_STAGE:-}" 143 aborted
}

# ---- one guarded GPU command ----
HARD_FAIL=0; LAST_SEG=""
guarded() {   # guarded <tag> <budget_gb> <cap_gb|0> <max_min> <alloc conf|-> <online|offline> -- cmd ...
  local tag="$1" budget="$2" cap="$3" mins="$4" conf="$5" net="$6"; shift 7
  local log="$PDIR/logs/$tag.log" sum="$PDIR/$tag.memguard.json" attempt=0 rc off
  local envs=(env -u PYTORCH_CUDA_ALLOC_CONF -u WTM_GPU_CAP_GB)
  [ "$net" = online ] && envs+=(-u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE)
  [ "$conf" = - ] || envs+=(PYTORCH_CUDA_ALLOC_CONF="$conf")
  [ "$net" = offline ] && envs+=(HF_HUB_OFFLINE=1)
  LAST_SEG="$PDIR/logs/$tag.last.log"
  while :; do
    attempt=$((attempt + 1))
    if vllm_running; then HARD_FAIL=1; return 75; fi
    if ! python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
      plog "  $tag: NO_CUDA torch.cuda.is_available() is False (lx-exp lost the GPU? docker restart lx-exp)"; HARD_FAIL=1; return 69
    fi
    rm -f "$sum"
    off=0; [ -f "$log" ] && off=$(wc -c < "$log" | tr -d ' ')
    echo "##### $(ts) ATTEMPT $tag n=$attempt budget=${budget}G cap=${cap}G max_min=$mins alloc=$conf hub=$net cmd: $*" >> "$log"
    plog "  $tag: memguard budget ${budget}G cap ${cap}G wall ${mins} min (attempt $attempt) → $log"
    # shellcheck disable=SC2086  # MG_EXTRA is a word list on purpose
    "${envs[@]}" python3 training/memguard.py --name "pipe-$tag" --budget-gb "$budget" --floor-gb "$FLOOR_GB" \
        --gpu-cap-gb "$cap" --max-minutes "$mins" --lock "$GPU_LOCK" --log "$LOGS/memguard.log" --status-dir "$LOGS" \
        --summary "$sum" $MG_EXTRA -- "$@" >> "$log" 2>&1 8>&- &
    MGPID=$!
    wait "$MGPID"; rc=$?
    MGPID=""
    tail -c +$((off + 1)) "$log" > "$LAST_SEG"
    if [ "$rc" -eq 75 ] && [ ! -f "$sum" ] && grep -q "MEMGUARD_REFUSE" "$LAST_SEG"; then   # refused before the job started
      plog "  $tag: $(grep -m1 -o 'MEMGUARD_REFUSE.*' "$LAST_SEG" | cut -c1-250)"
      if [ "$attempt" -le "$REFUSE_RETRIES" ]; then
        plog "  $tag: refused, retry $attempt/$REFUSE_RETRIES in ${REFUSE_WAIT_S}s"
        sleep "$REFUSE_WAIT_S" & wait $!
        continue
      fi
      HARD_FAIL=1
      return 75
    fi
    break
  done
  grep -E "GPU_CAP|GUARD_OK|GUARD_FAIL|MERGED|MERGE_FAIL|STACKED|STACK_FAIL|_DONE|_ALREADY_DONE|RESUMING|MEMGUARD_|EVAL_|ADAPTER |READY|VLLM_|SERVE_FAILED|ARM_|SAFETY_(FAIL|REFUSE|WARN|ARM_DONE)|AUDIT_(FAIL|REFUSE|ARM_FAIL|ARM_CACHED)|NO_DRIFT|NO_CUDA|BASELINE|DRIFT |Traceback" \
      "$LAST_SEG" | grep -v '^#####' | tail -n 25 | cut -c1-300 | while IFS= read -r l; do plog "  | $l"; done
  [ "$rc" -eq 137 ] && HARD_FAIL=1
  if [ -f "$sum" ]; then
    local gone
    gone=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("tree_gone"))' "$sum" 2>/dev/null)
    if [ "$gone" != True ]; then
      plog "  $tag: MEMGUARD_LEFTOVER (tree_gone=$gone): processes may still hold memory; stopping"
      HARD_FAIL=1
      [ "$rc" -eq 0 ] && rc=70
    fi
  elif [ "$rc" -eq 0 ]; then
    plog "  $tag: WARNING no memguard summary written ($sum)"
  fi
  plog "  $tag rc=$rc"
  return "$rc"
}
has() { grep -qE "$1" "$LAST_SEG"; }
need() {   # need <path>...: inputs a stage consumes (missing → exit 66 for this stage)
  local p
  for p in "$@"; do [ -e "$p" ] || { plog "  missing input $p"; return 66; }; done
}
fresh_flag() { in_list "$1" "$FRESH" && echo --fresh; return 0; }
auto_fresh() {   # auto_fresh <stage> <out> <model> <data> <epochs>: sets FRESH_ARG=--fresh when FRESH names the stage or
  # <out>/args.json was written with other model/data/epochs (STAGE_CONFIG_CHANGED for SFT_EPOCHS/DPO_EPOCHS, also after a run
  # with the new value was killed or refused before it set the old outputs aside); otherwise FRESH_ARG is empty (resume).
  local why
  FRESH_ARG=""
  if in_list "$1" "$FRESH"; then FRESH_ARG=--fresh; return 0; fi
  if why=$(trainer_args_changed "$2" "$3" "$4" "$5"); then
    plog "  STAGE_AUTO_FRESH $1: $2/args.json differs ($why), so a resume is impossible; passing --fresh (old outputs → stale_*)"
    FRESH_ARG=--fresh
  fi
  return 0
}

# ================================================= stages =================================================
audit_pair() {   # audit_pair <stage> <tag base|tuned> <max_min> <arms...>
  # safety_eval.py merges arms into ONE --out file (an arm whose config_hash — incl. the adapter's content sha — is already there
  # is skipped) and computes comparisons vs the 'base' arm found in that file: audits_base writes base into SAFETY_OUT,
  # audits_tuned adds m1-m3 to the same file. alignment_audit.py's documented two-stage form: base → ALIGN_BASE_OUT, tuned →
  # ALIGN_TUNED_OUT with --baseline-arm <ALIGN_BASE_OUT minus .json>.arms/base.json (its drift baseline; without base or a
  # baseline it refuses non-base arms: NO_DRIFT, exit 1). Both cache per arm, so a rerun after a kill redoes only missing arms.
  local stage="$1" tag="$2" mins="$3"; shift 3
  local sub script done_line out mk rc extra arms b c
  for sub in safety alignment; do
    extra=(); arms=("$@")
    if [ "$sub" = safety ]; then
      script=experiments/training_eval/safety_eval.py; done_line=SAFETY_DONE; out="$SAFETY_OUT"; b=$SAFETY_BUDGET; c=$SAFETY_CAP
      extra=(--gen-dir "$RES/cache/safety.gen")   # raw HarmBench replies: under */cache/ (git-ignored), never published
      [ "$tag" = tuned ] && arms=(base "$@")      # same-run base arm: comparisons need an identical measure_hash
    else
      script=experiments/training_eval/alignment_audit.py; done_line=AUDIT_DONE; b=$ALIGN_BUDGET; c=$ALIGN_CAP
      if [ "$tag" = base ]; then out="$ALIGN_BASE_OUT"
      else
        out="$ALIGN_TUNED_OUT"; extra=(--baseline-arm "${ALIGN_BASE_OUT%.json}.arms/base.json")
      fi
    fi
    mk="$PDIR/markers/$stage.$sub.done"
    if [ -f "$mk" ] && [ "$(marker_sig "$mk")" = "$(sig_of "$stage") $sub" ] && ! in_list "$stage" "$FORCE" && [ "$FORCE" != all ]; then
      plog "  $stage.$sub: SUB_SKIP (marker $mk)"; continue
    fi
    need "$script" || return $?
    if [ "$sub" = alignment ] && [ "${#extra[@]}" -gt 0 ] && ! need "${extra[1]}"; then
      plog "  $stage.alignment: the drift baseline comes from audits_base (FORCE=audits_base to redo it)"; return 66
    fi
    local t0=$SECONDS
    guarded "$stage.$sub" "$b" "$c" "$mins" "$AUDIT_ALLOC_CONF" online -- \
        python3 "$script" --base "$BASE" --arms "${arms[@]}" --out "$out" ${extra[@]+"${extra[@]}"}
    rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    has "$done_line" || { plog "  $stage.$sub: rc 0 but no $done_line line"; return 1; }
    [ -s "$out" ] || { plog "  $stage.$sub: $out missing"; return 1; }
    write_marker "$mk" "$stage.$sub" "$(sig_of "$stage") $sub" $((SECONDS - t0)) "{\"out\": \"$out\"}"
  done
}

stage_audits_base() { audit_pair audits_base base "$AUDIT_BASE_MAX_MIN" base; }
stage_audits_tuned() {
  local arms; arms=$(avail_arms)
  [ -n "$arms" ] || { plog "  audits_tuned: no trained adapter exists"; return 66; }
  # shellcheck disable=SC2086
  audit_pair audits_tuned tuned "$AUDIT_TUNED_MAX_MIN" $arms
}

stage_m1_sft() {
  local rc
  auto_fresh m1_sft "$RUNS/m1_sft" "$BASE" "$DATA" "$SFT_EPOCHS"
  guarded m1_sft 30 26 "$SFT_MAX_MIN" "$EXP" offline -- python3 training/train_sft.py --model "$BASE" --data "$DATA" \
      --out "$RUNS/m1_sft" --epochs "$SFT_EPOCHS" $FRESH_ARG
  rc=$?; [ "$rc" -eq 0 ] || return "$rc"
  has "SFT_DONE|SFT_ALREADY_DONE" || { plog "  m1_sft: no SFT_DONE line"; return 1; }
  need "$M1/adapter_model.safetensors" "$M1/adapter_config.json" "$RUNS/m1_sft/result.json"
}

stage_m1_merge() {
  local rc
  need "$M1/adapter_model.safetensors" || return $?
  guarded m1_merge 25 14 "$MERGE_MAX_MIN" "$EXP" offline -- python3 training/merge_lora.py --base "$BASE" --adapter "$M1" \
      --out "$M1_MERGED"
  rc=$?; [ "$rc" -eq 0 ] || return "$rc"
  python3 - "$M1_MERGED" "$M1/adapter_model.safetensors" > "$PDIR/logs/m1_merge.gate.txt" 2>&1 <<'PY'
import hashlib, json, os, sys
m, adapter = sys.argv[1:3]
info = json.load(open(os.path.join(m, "merge_info.json"))) if os.path.exists(os.path.join(m, "merge_info.json")) else {}
h = hashlib.sha256()
with open(adapter, "rb") as fh:
    for c in iter(lambda: fh.read(1 << 20), b""):
        h.update(c)
same = info.get("adapter_sha256") == h.hexdigest()   # merge_lora.py:161: the merged dir is built from THIS M1
ok = info.get("ok") is True and os.path.exists(os.path.join(m, "config.json")) and same
print(f"merge gate: ok={info.get('ok')!r} config.json={os.path.exists(os.path.join(m, 'config.json'))} "
      f"adapter_sha256_matches_m1={same} merge_delta={info.get('merge_delta')} adapter_effect={info.get('adapter_effect')}")
sys.exit(0 if ok else 1)
PY
  rc=$?
  plog_file "$PDIR/logs/m1_merge.gate.txt"
  [ "$rc" -eq 0 ] || { plog "  m1_merge: gate failed (merge_info ok must be true, config.json present, adapter_sha256 = M1's)"; return 1; }
}

stage_m2_grpo() {
  local rc cur why flag
  need "$M1_MERGED/config.json" "$M1_MERGED/merge_info.json" "$GRPO_DATA_DIR/grpo_train.jsonl" || return $?
  # M2 must be trained on THIS M1. train_grpo.py's resume check sees only the --model path (RESUME_KEYS, train_grpo.py:66-68),
  # so after an M1 retrain it would resume from — or report GRPO_ALREADY_DONE for (:150-155) — checkpoints trained on the OLD
  # merged M1, and stack_lora would happily stack the new M1 with that stale delta. The pipeline records the merged M1's
  # adapter content sha (merge_lora.py:161) next to the GRPO run whenever GRPO starts fresh, and forces --fresh when it differs.
  cur=$(m1_merged_sha)
  [ -n "$cur" ] || { plog "  m2_grpo: $M1_MERGED/merge_info.json has no adapter_sha256 (rerun m1_merge: FORCE=m1_merge)"; return 66; }
  flag=$(fresh_flag m2_grpo)
  why=$(grpo_fresh_reason "$cur") || { plog "  m2_grpo: could not inspect $RUNS/m2_grpo"; return 1; }
  if [ -z "$flag" ] && [ -n "$why" ]; then
    plog "  GRPO_UPSTREAM_CHANGED: $why; passing --fresh (train_grpo.py moves the old run to $RUNS/m2_grpo/stale-<run_id>)"
    flag=--fresh
  fi
  if [ -n "$flag" ] || [ -z "$(ls -d "$RUNS"/m2_grpo/checkpoint-* "$RUNS/m2_grpo/final" 2>/dev/null)" ]; then
    # this attempt starts GRPO fresh: record which M1 it trains on (kept by train_grpo's set-aside, which moves only
    # checkpoints/final/progress/result/guards)
    mkdir -p "$RUNS/m2_grpo"
    python3 - "$RUNS/m2_grpo/pipeline_upstream.json" "$cur" "$M1_MERGED" "$GRPO_STEPS" "${why:-${flag:+FRESH=m2_grpo}}" <<'PY' || return 1
import json, os, sys, time
p, sha, merged, steps, why = sys.argv[1:6]
d = {"m1_adapter_sha256": sha, "m1_merged": merged, "grpo_steps": int(steps), "fresh_reason": why or "no earlier GRPO run",
     "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "recorded_epoch": time.time()}
with open(p + ".tmp", "w") as fh:
    json.dump(d, fh, indent=1)
os.replace(p + ".tmp", p)
PY
    plog "  m2_grpo: GRPO starts fresh on merged M1 adapter sha256 ${cur:0:12} (recorded in $RUNS/m2_grpo/pipeline_upstream.json)"
  else
    plog "  m2_grpo: resuming GRPO trained on the same merged M1 (adapter sha256 ${cur:0:12})"
  fi
  # shellcheck disable=SC2086  # GRPO_EXTRA is a list of train_grpo.py flags (v2 hook)
  guarded m2_grpo 65 54 "$GRPO_MAX_MIN" - offline -- python3 training/train_grpo.py --model "$M1_MERGED" --data "$GRPO_DATA_DIR" \
      --out "$RUNS/m2_grpo" --max-steps "$GRPO_STEPS" --lr "$GRPO_LR" $GRPO_EXTRA $flag
  rc=$?; [ "$rc" -eq 0 ] || return "$rc"
  has "GRPO_DONE|GRPO_ALREADY_DONE" || { plog "  m2_grpo: no GRPO_DONE line"; return 1; }
  need "$M2/adapter_model.safetensors" "$RUNS/m2_grpo/result.json" || return $?
  python3 - "$RUNS/m2_grpo/result.json" <<'PY' | while IFS= read -r l; do plog "  $l"; done
import json, sys
h = [x for x in json.load(open(sys.argv[1])).get("log_history") or [] if "grad_norm" in x and ("reward" in x or "completions/clipped_ratio" in x)]
if h and all(not float(x.get("grad_norm") or 0) for x in h):
    print(f"WARNING GRPO_ZERO_GRADIENT: grad_norm 0 in all {len(h)} logged steps (GRPO learned nothing; see summary.json)")
elif h:
    print(f"grpo learning signal: {sum(1 for x in h if float(x.get('grad_norm') or 0))}/{len(h)} steps with grad_norm > 0")
PY
  return 0
}

stage_m2_stack() {
  local rc tmpinfo="$M2S.tmp/stack_info.json"
  rm -f "$tmpinfo"   # a leftover from an earlier run must not trigger the --min-effect 0 retry
  need "$M1/adapter_config.json" "$M1_MERGED/config.json" "$M2/adapter_config.json" || return $?
  local f   # a previous run's retry record must not describe this run
  for f in "$PDIR/m2_stack_min_effect_retry.json" "$PDIR/m2_stack.attempt1.stack_info.json" "$PDIR/m2_stack.retry.memguard.json"; do
    stale_marker "$f"
  done
  local cmd=(python3 training/stack_lora.py --base "$BASE" --first "$M1" --first-merged "$M1_MERGED" --second "$M2" --out "$M2S")
  guarded m2_stack 25 14 "$STACK_MAX_MIN" "$EXP" offline -- "${cmd[@]}"
  rc=$?
  if [ "$rc" -eq 1 ] && [ "$HARD_FAIL" = 0 ] && [ -f "$tmpinfo" ]; then
    # Retry ONLY when every reason is "adapter effect … ≤ --min-effect …" (merge_lora.verdict): the stack itself was within --tol
    # but GRPO moved the probe too little to verify two-sidedly. Recorded, never silent.
    python3 - "$tmpinfo" "$PDIR/m2_stack_min_effect_retry.json" > "$PDIR/logs/m2_stack.retry_check.txt" 2>&1 <<'PY'
import json, os, sys, time
info = json.load(open(sys.argv[1]))
reasons = info.get("reasons") or []
only = bool(reasons) and all("--min-effect" in r for r in reasons)
rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "first_attempt": {k: info.get(k) for k in (
       "reasons", "second_effect", "stack_delta", "first_merge_residual", "served_second_effect", "min_effect", "tol")},
       "retried_with_min_effect_0": only}
if only:
    with open(sys.argv[2], "w") as fh:
        json.dump(rec, fh, indent=1)
print("stack first attempt: " + json.dumps(rec)[:400])
sys.exit(0 if only else 1)
PY
    local only=$?
    plog_file "$PDIR/logs/m2_stack.retry_check.txt"
    if [ "$only" -eq 0 ]; then
      cp -f "$tmpinfo" "$PDIR/m2_stack.attempt1.stack_info.json"
      plog "  m2_stack: STACK_MIN_EFFECT_RETRY — failed only on --min-effect; retrying once with --min-effect 0 (recorded)"
      guarded m2_stack.retry 25 14 "$STACK_MAX_MIN" "$EXP" offline -- "${cmd[@]}" --min-effect 0
      rc=$?
    fi
  fi
  [ "$rc" -eq 0 ] || return "$rc"
  python3 - "$M2S" > "$PDIR/logs/m2_stack.gate.txt" 2>&1 <<'PY'
import json, os, sys
d = sys.argv[1]
info = json.load(open(os.path.join(d, "stack_info.json"))) if os.path.exists(os.path.join(d, "stack_info.json")) else {}
print(f"stack gate: ok={info.get('ok')!r} verified={info.get('verified')!r} min_effect={info.get('min_effect')} "
      f"stack_delta={info.get('stack_delta')} second_effect={info.get('second_effect')}")
sys.exit(0 if info.get("ok") is True and info.get("verified") is True else 1)
PY
  rc=$?
  plog_file "$PDIR/logs/m2_stack.gate.txt"
  [ "$rc" -eq 0 ] || { plog "  m2_stack: gate failed (stack_info ok and verified must be true)"; return 1; }
}

stage_m3_dpo() {
  local rc
  need "$DATA/dpo_train.jsonl" || return $?
  auto_fresh m3_dpo "$RUNS/m3_dpo" "$BASE" "$DATA/dpo_train.jsonl" "$DPO_EPOCHS"
  guarded m3_dpo 70 60 "$DPO_MAX_MIN" "$EXP" offline -- python3 training/train_dpo.py --model "$BASE" \
      --data "$DATA/dpo_train.jsonl" --out "$RUNS/m3_dpo" --epochs "$DPO_EPOCHS" $FRESH_ARG
  rc=$?; [ "$rc" -eq 0 ] || return "$rc"
  has "DPO_DONE|DPO_ALREADY_DONE" || { plog "  m3_dpo: no DPO_DONE line"; return 1; }
  need "$M3/adapter_model.safetensors" "$RUNS/m3_dpo/result.json"
}

stage_eval() {
  local rc name results scen out
  local arms; arms=$(avail_arms)
  [ -n "$arms" ] || { plog "  eval: no trained adapter exists (m1/m2/m3)"; return 66; }
  plog "  eval: arms $arms"
  need "$LOGS/mirror_paths.txt" || return $?
  [ -n "$MP" ] || { plog "  eval: line 2 of $LOGS/mirror_paths.txt is empty"; return 66; }
  guarded eval "$EVAL_BUDGET" 0 "$EVAL_MAX_MIN" - online -- bash "$SELF" __eval_inner
  rc=$?; [ "$rc" -eq 0 ] || return "$rc"
  [ "$(grep -c '^EVAL_DONE' "$LAST_SEG")" -ge 2 ] || { plog "  eval: expected 2 EVAL_DONE lines"; return 1; }
  has "^CAPABILITY_DONE" || { plog "  eval: no CAPABILITY_DONE"; return 1; }
  local nerr
  nerr=$(python3 -c 'import json, sys
a = json.load(open(sys.argv[1])).get("arms", {})
print(sum(1 for x in a.values() for t in x.values() if isinstance(t, dict) and "error" in t))' "$CAP_OUT" 2>/dev/null)
  [ "${nerr:-0}" = 0 ] || plog "  eval: WARNING CAPABILITY_ERRORS $nerr task(s) recorded an error in $CAP_OUT"
  # AuditKIT re-scoring of the recorded episodes: CPU only (model="precomputed"), so no memguard; never a stop.
  for name in test hard_test; do
    if [ "$name" = test ]; then results="$EVAL_TEST"; scen="$TEST_GLOB"; out="$AK_TEST"
    else results="$EVAL_HARD"; scen="$HARD_GLOB"; out="$AK_HARD"; fi
    python3 experiments/training_eval/auditkit_eval.py --results "$results" --scenarios "$scen" --base "$BASE_SPEC" \
        --adapters-txt "${results%.jsonl}.adapters.txt" --arms $(arm_labels "$arms") --out "$out" \
        >> "$PDIR/logs/eval.auditkit_$name.log" 2>&1
    rc=$?
    echo "$rc" > "$PDIR/eval.auditkit_$name.rc"
    if [ "$rc" -eq 0 ]; then plog "  eval: auditkit_$name ok → $out"
    else plog "  eval: AUDITKIT_FAIL $name rc=$rc (see $PDIR/logs/eval.auditkit_$name.log; rerun offline from the JSONL)"; fi
  done
  return 0
}

# ================================================= main =================================================
trap 'on_abort TERM' TERM
trap 'on_abort INT' INT

# single instance: fd 8 holds runs/pipeline/.pipeline.lock for the whole run (closed for every memguard child)
exec 8>>"$PDIR/.pipeline.lock" || { echo "PIPELINE_REFUSE cannot open $PDIR/.pipeline.lock"; exit 2; }
python3 -c 'import errno, fcntl, sys
try:
    fcntl.flock(8, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as e:
    sys.exit(75 if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES) else 2)'
case $? in
  0) echo "run_pipeline pid=$$ since=$(ts)" > "$PDIR/.pipeline.lock" ;;   # by path: the flock stays on fd 8
  75) echo "PIPELINE_REFUSE another run_pipeline.sh holds $PDIR/.pipeline.lock: $(cat "$PDIR/.pipeline.lock" 2>/dev/null)"
      exit 75 ;;
  *) echo "PIPELINE_REFUSE cannot lock $PDIR/.pipeline.lock"; exit 2 ;;
esac

bad() { plog "PIPELINE_REFUSE $*"; status preflight refused 2 "$*"; exit 2; }
si=$(stage_idx "$START_AT") || bad "START_AT=$START_AT is not a stage ($ALL_STAGES)"
ei=$(stage_idx "$STOP_AFTER") || bad "STOP_AFTER=$STOP_AFTER is not a stage ($ALL_STAGES)"
[ "$si" -le "$ei" ] || bad "START_AT=$START_AT comes after STOP_AFTER=$STOP_AFTER"
[[ "$SFT_EPOCHS" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v e="$SFT_EPOCHS" 'BEGIN { exit !(e > 0) }' || bad "SFT_EPOCHS=$SFT_EPOCHS"
[[ "$GRPO_STEPS" =~ ^[1-9][0-9]*$ ]] || bad "GRPO_STEPS=$GRPO_STEPS (positive integer)"
[[ "$DPO_EPOCHS" =~ ^[1-9][0-9]*$ ]] || bad "DPO_EPOCHS=$DPO_EPOCHS (train_dpo.py --epochs is an integer)"
[[ "${PREFLIGHT_WAIT_MIN:-0}" =~ ^[0-9]+$ ]] || bad "PREFLIGHT_WAIT_MIN=$PREFLIGHT_WAIT_MIN (minutes, integer)"
for s in $FORCE $FRESH $SOFT_STAGES; do
  [ "$s" = all ] || stage_idx "$s" >/dev/null || bad "unknown stage '$s' in FORCE/FRESH/SOFT_STAGES"
done
[ -n "$BASE" ] && [ -f "$BASE/config.json" ] || bad "Tiny Aya snapshot not found (BASE_MODEL='$BASE')"
if [ "$si" -le "$(stage_idx m2_grpo)" ] && [ "$ei" -ge "$(stage_idx m2_grpo)" ]; then
  [ -s "$GRPO_DATA_DIR/grpo_train.jsonl" ] || bad "GRPO_DATA_DIR=$GRPO_DATA_DIR has no grpo_train.jsonl"
  [[ "$GRPO_LR" =~ ^[0-9.]+(e-?[0-9]+)?$ ]] || bad "GRPO_LR=$GRPO_LR"
fi
for f in sft_train.jsonl sft_val.jsonl dpo_train.jsonl grpo_train.jsonl; do
  [ -s "$DATA/$f" ] || bad "$DATA/$f is missing or empty"
done
if [ "$si" -le "$(stage_idx eval)" ] && [ "$ei" -ge "$(stage_idx eval)" ]; then
  [ -n "$MP" ] || bad "eval needs the mirror path on line 2 of $LOGS/mirror_paths.txt"
  ls $TEST_GLOB >/dev/null 2>&1 && ls $HARD_GLOB >/dev/null 2>&1 || bad "eval scenario files $TEST_GLOB / $HARD_GLOB missing"
fi
# PREFLIGHT_WAIT_MIN > 0: wait (polling every PREFLIGHT_POLL_S) for a running guarded job / our vLLM to finish instead of refusing
wait_deadline=$(( $(date +%s) + ${PREFLIGHT_WAIT_MIN:-0} * 60 ))
while :; do
  if ! holder=$(gpu_lock_holder); then
    why="another guarded GPU job holds $GPU_LOCK: $holder (never wrap run_pipeline.sh in memguard)"
  elif our=$(our_vllm | sort -u | tr '\n' ' ') && [ -n "$our" ]; then
    why="our vLLM server is running (pids ${our% }); stop it first (bash experiments/infra/stop_vllm.sh)"
  else
    break
  fi
  if [ "$(date +%s)" -ge "$wait_deadline" ]; then
    plog "PIPELINE_REFUSE $why"; status preflight refused 75 "$why"; exit 75
  fi
  plog "PIPELINE_WAIT $why; waiting (PREFLIGHT_WAIT_MIN=${PREFLIGHT_WAIT_MIN})"
  status preflight waiting "" "$why"
  sleep "${PREFLIGHT_POLL_S:-60}" & wait $!
done

if [ -n "${HF_TOKEN:-}" ]; then HF_AUTH=env
elif [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then HF_AUTH=token_file
else HF_AUTH=none; fi
python3 - "$PDIR/config.json" <<PY
import hashlib, json, os, time
data = "$DATA"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
cfg = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "pid": $$, "stages": "$ALL_STAGES".split(),
       "start_at": "$START_AT", "stop_after": "$STOP_AFTER", "force": "$FORCE", "fresh": "$FRESH", "soft_stages": "$SOFT_STAGES",
       "sft_epochs": "$SFT_EPOCHS", "grpo_steps": int("$GRPO_STEPS"), "dpo_epochs": int("$DPO_EPOCHS"),
       "cap_limit_scale": "$CAP_LIMIT_SCALE", "base": "$BASE", "mirror_path": "$MP", "data": data,
       "data_sha256": {f: sha(os.path.join(data, f)) for f in ("sft_train.jsonl", "sft_val.jsonl", "dpo_train.jsonl", "grpo_train.jsonl")},
       "max_minutes": {"sft": $SFT_MAX_MIN, "merge": $MERGE_MAX_MIN, "grpo": $GRPO_MAX_MIN, "stack": $STACK_MAX_MIN,
                       "dpo": $DPO_MAX_MIN, "eval": $EVAL_MAX_MIN, "audit_base": $AUDIT_BASE_MAX_MIN, "audit_tuned": $AUDIT_TUNED_MAX_MIN},
       "floor_gb": $FLOOR_GB, "audit_budget_gb": $AUDIT_BUDGET, "audit_cap_gb": $AUDIT_CAP, "audit_alloc_conf": "$AUDIT_ALLOC_CONF",
       "refuse_retries": $REFUSE_RETRIES, "refuse_wait_s": $REFUSE_WAIT_S, "hf_auth": "$HF_AUTH", "mg_extra": "$MG_EXTRA",
       "v2": {"grpo_lr": "$GRPO_LR", "grpo_data_dir": "$GRPO_DATA_DIR", "grpo_extra": "$GRPO_EXTRA", "m2_adapter": "$M2",
              "extra_arms": "$EXTRA_ARMS", "eval_budget_gb": $EVAL_BUDGET}}
with open("$PDIR/config.json.tmp", "w") as fh:
    json.dump(cfg, fh, indent=1)
os.replace("$PDIR/config.json.tmp", "$PDIR/config.json")
PY
[ $? -eq 0 ] || bad "could not write $PDIR/config.json (data files unreadable?)"
[ -n "$MP" ] && [ "$(readlink -f "$MP")" != "$(readlink -f "$BASE")" ] && \
  bad "BASE ($BASE) and the served mirror path ($MP, mirror_paths.txt line 2) differ: run_eval.sh would refuse every adapter"
python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null || \
  bad "NO_CUDA: torch.cuda.is_available() is False in lx-exp (GPU access lost? docker restart lx-exp)"
plog "PIPELINE_START pid=$$ ($(ts_ist)) stages $START_AT..$STOP_AFTER SFT_EPOCHS=$SFT_EPOCHS GRPO_STEPS=$GRPO_STEPS DPO_EPOCHS=$DPO_EPOCHS hf_auth=$HF_AUTH base=$BASE"
plog "  v2 hooks: GRPO_LR=$GRPO_LR GRPO_DATA_DIR=$GRPO_DATA_DIR GRPO_EXTRA='$GRPO_EXTRA' M2=$M2 EXTRA_ARMS='$EXTRA_ARMS'"
[ "$HF_AUTH" = none ] && plog "  WARNING no HF token (env or token file): the audits/eval download only public data"

idx=-1
for st in $ALL_STAGES; do
  idx=$((idx + 1))
  [ "$idx" -ge "$si" ] && [ "$idx" -le "$ei" ] || continue
  CUR_STAGE="$st"
  if [ "$st" = summary ]; then
    status summary running
    STAGE_STATES="$STAGE_STATES summary=done"
    break
  fi
  mk="$PDIR/markers/$st.done"; sig=$(sig_of "$st")
  forced=0; { [ "$FORCE" = all ] || in_list "$st" "$FORCE"; } && forced=1
  if [ -f "$mk" ] && [ "$forced" = 0 ]; then
    if [ "$(marker_sig "$mk")" = "$sig" ]; then
      plog "STAGE_SKIP $st (finished earlier: $mk)"
      STAGE_STATES="$STAGE_STATES $st=skipped"
      status "$st" skipped 0
      continue
    fi
    plog "STAGE_CONFIG_CHANGED $st: marker signature differs from '$sig'; running it again"
  fi
  stale_marker "$mk"
  for d in $(deps_of "$st"); do   # whatever consumed this stage's old output must be redone
    for f in "$PDIR/markers/$d.done" "$PDIR"/markers/"$d".*.done; do
      [ -f "$f" ] && { plog "  invalidating $(basename "$f") (consumes $st)"; stale_marker "$f"; }
    done
  done
  if [ "$forced" = 1 ]; then for f in "$PDIR"/markers/"$st".*.done; do stale_marker "$f"; done; fi
  plog "STAGE_START $st ($(ts_ist))"
  status "$st" running "" "memguard status: $LOGS/memguard-pipe-$st*.status"
  HARD_FAIL=0; t0=$SECONDS
  "stage_$st"; rc=$?
  if [ "$rc" -eq 0 ]; then
    write_marker "$mk" "$st" "$sig" $((SECONDS - t0))
    plog "STAGE_DONE $st in $(( (SECONDS - t0) / 60 )) min"
    STAGE_STATES="$STAGE_STATES $st=done"
    status "$st" done 0
  elif [ "$HARD_FAIL" = 0 ] && in_list "$st" "$SOFT_STAGES"; then
    plog "STAGE_FAIL_SOFT $st rc=$rc (in SOFT_STAGES: no marker, the chain continues; rerun to retry it)"
    STAGE_STATES="$STAGE_STATES $st=soft_failed"
    status "$st" soft_failed "$rc"
  else
    plog "STAGE_FAIL $st rc=$rc"
    plog "PIPELINE_STOP $st rc=$rc (fix, then rerun: finished stages are skipped, trainers resume from checkpoints)"
    STAGE_STATES="$STAGE_STATES $st=failed"
    status "$st" failed "$rc"
    finish "$st" "$rc" failed
  fi
done
CUR_STAGE=""
write_summary "" 0 2>&1 | tail -n 1 | while IFS= read -r l; do plog "$l"; done
soft_failed_list() { local x; for x in $STAGE_STATES; do [ "${x#*=}" = soft_failed ] && printf '%s ' "${x%=*}"; done; return 0; }
SOFT_FAILED=$(soft_failed_list)
status summary finished 0 "$PDIR/summary.json${SOFT_FAILED:+ (soft failures: ${SOFT_FAILED% })}"
plog "PIPELINE_DONE ($(ts_ist))${SOFT_FAILED:+ WITH SOFT FAILURES: ${SOFT_FAILED% } (rerun to retry them)} summary: $PDIR/summary.json"
