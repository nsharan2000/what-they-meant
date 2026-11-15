#!/bin/bash
# Before/after evaluation on the HELD-OUT test split (never trained on) — ONE serving path for every arm:
# vLLM serves the base Tiny Aya Global (verified mirror of CohereLabs/tiny-aya-global) and each trained LoRA adapter side by side.
# Usage (inside lx-exp): bash experiments/training_eval/run_eval.sh <out.jsonl> <adapter_name=path> [<adapter_name=path> ...]
#   e.g. bash experiments/training_eval/run_eval.sh experiments/training_eval/results/eval_test.jsonl \
#          m1=/workspace/runs/m1_sft/final m2=/workspace/runs/m2_grpo/stacked m3=/workspace/runs/m3_dpo/final
#   Every adapter must sit on the ORIGINAL base: M2 (trained on merged M1) is served as the stacked adapter, never m2_grpo/final
#   (that would serve W0 + dGRPO, a model that was never trained); M3 is m3_dpo/final when trained from base, m3_dpo/stacked
#   when trained from merged M1. The preflight refuses anything else (ADAPTER_BASE_MISMATCH, exit 1).
# Each adapter is served and queried as <name>-<10-hex content hash of adapter_config.json + adapter_model.safetensors> (same as
# `cat both | sha256sum | cut -c1-10`): the response cache is keyed on the served name, so a retrained adapter can never be
# answered from an older adapter's cached outputs. Once the server is READY, one block is appended to
# <out without .jsonl>.adapters.txt: a '=== <date> run_eval READY out=<out>' line, then '<date> <name> -> <served> = <path>' per
# adapter. Analysis takes each label's served name from the LAST block (capability_eval.py / auditkit_eval.py --adapters-txt):
# a retrained adapter adds rows under a new hash to the same <out>, so older blocks name older versions.
# An adapter dir with a stack_info.json (training/stack_lora.py) must say ok true AND verified true.
# Scenarios: $EVAL_SCENARIOS (default dataset/eval/test_*.jsonl; hard_test_*.jsonl = held-out hard pairs), written by
# experiments/training_eval/make_eval_set.py from splits.json.
# Levels L0r (toy) / L1 / L2 (realistic) / L2s (salience control) × prompts P0 (no instruction) / P2 (1-2 line hint); decision mode.
# Full run log: <out>.log. Resumable (wtm.run skips finished keys). Ends EVAL_DONE only when every arm has every
# scenario × level × prompt row; otherwise EVAL_INCOMPLETE <model> have/want and exit 1.
# The vLLM server (0.45 of the shared box) is stopped on exit unless KEEP_VLLM=1 (set it only when capability_eval.py /
# auditkit_eval.py reuse the server next; the last of them must run experiments/infra/stop_vllm.sh).
# ONE GPU job at a time: run it under memguard (the RULE for every GPU job: budget/floor kill, oom_score_adj 1000 inherited
# by the server), e.g.
#   python3 training/memguard.py --name eval --budget-gb 70 -- bash experiments/training_eval/run_eval.sh <out> m1=... m2=...
# (budget: the 0.45 pool is ~54 GB of 121; an idle 0.45 server was seen holding ~59 GB). memguard holds /workspace/.gpu_job.lock
# on its own file description, so this script accepts it only when the lock file names a memguard.py that is its own ancestor
# (EVAL_UNDER_MEMGUARD). Run bare, it takes the same lock itself (flock(2) on fd 9, memguard's call) from start to exit and
# writes 'run_eval pid=... since=...' into it. Any other holder: EVAL_REFUSE, exit 75. The lock is checked before the EXIT
# trap is set, so a refused run never stops the holder's server. The vLLM server runs with fd 9 closed (serve_vllm.sh).
# KEEP_VLLM=1 under memguard: the server lives only until memguard's command ends (its exit sweep kills the whole tree), so the
# clients that reuse it go inside the same command:
#   memguard.py ... -- bash -c 'KEEP_VLLM=1 bash experiments/training_eval/run_eval.sh ... && python3 \
#     experiments/training_eval/capability_eval.py ...; bash experiments/infra/stop_vllm.sh'
# Run bare with KEEP_VLLM=1, the kept server holds no lock and the next memguard preflight sees it only through MemAvailable.
set -uo pipefail
cd /workspace
[ $# -ge 1 ] || { echo "usage: run_eval.sh <out.jsonl> [name=adapter_dir ...]"; exit 1; }
OUT="$1"; shift
GPU_LOCK=/workspace/.gpu_job.lock
under_memguard() {   # the lock's holder is the memguard.py named in the lock file AND an ancestor of this script
  local holder p=$PPID
  holder=$(sed -n 's/^.* pid=\([0-9][0-9]*\) since=.*$/\1/p' "$GPU_LOCK" | head -1)
  while [ -n "$holder" ] && [ "${p:-1}" -gt 1 ]; do
    [ "$p" = "$holder" ] && { ps -o args= -p "$p" | grep -q 'memguard\.py'; return; }
    p=$(ps -o ppid= -p "$p" | tr -d ' ')
  done
  return 1
}
exec 9>>"$GPU_LOCK" || exit 1
python3 -c 'import errno, fcntl, sys
try:
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as e:
    sys.exit(75 if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES) else f"GPU lock: {e}")'
case $? in
  0) echo "run_eval pid=$$ since=$(date '+%Y-%m-%d %H:%M:%S')" > "$GPU_LOCK" ;;   # by path: the flock stays on fd 9
  75) if under_memguard; then echo "EVAL_UNDER_MEMGUARD $(cat "$GPU_LOCK")"
      else echo "EVAL_REFUSE another GPU job holds $GPU_LOCK: $(cat "$GPU_LOCK" 2>/dev/null)"; exit 75; fi ;;
  *) exit 1 ;;
esac
BASE_NAME=tiny-aya-global-local; URL=http://localhost:8001/v1
LEVELS="L0r L1 L2 L2s"; PROMPTS="P0 P2"
# EVAL_SCENARIOS picks the scenario files (default: the held-out test split). The held-out hard pairs are a second run:
#   EVAL_SCENARIOS='dataset/eval/hard_test_*.jsonl' bash experiments/training_eval/run_eval.sh <out_hard.jsonl> m1=... m2=...
SCEN="${EVAL_SCENARIOS:-dataset/eval/test_*.jsonl}"
trap '[ "${KEEP_VLLM:-0}" = 1 ] || bash experiments/infra/stop_vllm.sh' EXIT
MP=$(sed -n 2p /workspace/logs/mirror_paths.txt)
[ -n "$MP" ] || { echo "BASE_NOT_FOUND '$MP' (line 2 of /workspace/logs/mirror_paths.txt)"; exit 1; }
mkdir -p "$(dirname "$OUT")"

ADAPTERS=$(python3 - "$MP" "$@" <<'PY'
"""Preflight + content hash for each name=path adapter → stdout lines '<name> <served name> <path>'."""
import hashlib, json, os, re, sys

HUB_IDS = {"CohereLabs/tiny-aya-global", "1-800-LLMs/tiny-aya-global"}   # the mirror is verified identical to the official weights
MAX_R = 64                                                                  # = --max-lora-rank below
base, bad, seen = os.path.realpath(sys.argv[1]), [], set()


def load(p):
    """Parsed JSON, or the reason it could not be read (a str) — a broken file must be refused, not crash the preflight."""
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"


for kv in sys.argv[2:]:
    name, _, path = kv.partition("=")
    cfg_p, w_p = os.path.join(path, "adapter_config.json"), os.path.join(path, "adapter_model.safetensors")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in seen or not path or re.search(r"\s", path):
        bad.append(f"{kv}: need a unique name [A-Za-z0-9._-]+ and a path without spaces"); continue
    seen.add(name)
    if not (os.path.isfile(cfg_p) and os.path.isfile(w_p)):
        bad.append(f"{name}: {path} has no adapter_config.json + adapter_model.safetensors"); continue
    cfg = load(cfg_p)
    if not isinstance(cfg, dict):
        bad.append(f"{name}: unreadable adapter_config.json ({cfg})"); continue
    si_p = os.path.join(path, "stack_info.json")
    if os.path.exists(si_p):
        si = load(si_p)
        if not isinstance(si, dict):
            bad.append(f"{name}: unreadable stack_info.json ({si})"); continue
        if si.get("ok") is not True or si.get("verified") is not True:
            bad.append(f"{name}: stack_info.json ok={si.get('ok')!r} verified={si.get('verified')!r} "
                       "(restack with --first-merged until the check passes)"); continue
    rec = cfg.get("base_model_name_or_path") or ""
    if rec not in HUB_IDS and os.path.realpath(rec) != base:
        bad.append(f"{name}: trained on {rec!r}, served base is {base}"); continue
    if not isinstance(cfg.get("r"), int) or cfg["r"] > MAX_R:
        bad.append(f"{name}: r={cfg.get('r')!r} (max {MAX_R})"); continue
    h = hashlib.sha256()
    for f in (cfg_p, w_p):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    print(name, f"{name}-{h.hexdigest()[:10]}", path)
for b in bad:
    print("ADAPTER_BASE_MISMATCH", b, file=sys.stderr)
sys.exit(1 if bad else 0)
PY
) || exit 1

LORA=""; MODELS="vllm:$BASE_NAME@$URL"
while read -r name served path; do
  [ -n "$name" ] || continue
  LORA="$LORA $served=$path"; MODELS="$MODELS vllm:$served@$URL"
  echo "ADAPTER $name -> $served = $path"
done <<< "$ADAPTERS"
export VLLM_EXTRA_ARGS=""
[ -n "$LORA" ] && export VLLM_EXTRA_ARGS="--enable-lora --max-lora-rank 64 --max-loras 4 --lora-modules${LORA}"
bash experiments/infra/serve_vllm.sh "$MP" "$BASE_NAME" 8001 16384 0.45 || { echo SERVE_FAILED; exit 1; }
TS=$(date '+%Y-%m-%d %H:%M:%S')
{ echo "=== $TS run_eval READY out=$OUT"
  while read -r name served path; do [ -z "$name" ] || echo "$TS $name -> $served = $path"; done <<< "$ADAPTERS"
} >> "${OUT%.jsonl}.adapters.txt"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') run_eval $MODELS" >> "$OUT.log"
python3 -m wtm.run --scenarios $SCEN --models $MODELS \
  --levels $LEVELS --prompts $PROMPTS --mode decision --temperature 0 --workers 32 --strict --out "$OUT" 2>&1 \
  | tee -a "$OUT.log" | grep -E "jobs|done \(|Traceback|Error|RUN_SUMMARY" | tail -6
RC=${PIPESTATUS[0]}
[ "$RC" = 0 ] || echo "EVAL_RUN_FAILED rc=$RC (episode errors are in $OUT.log)"

python3 - "$OUT" "$LEVELS" "$PROMPTS" "$SCEN" $MODELS <<'PY'
"""Count the rows each arm has in <out> against every expected key (n_test_scenarios × levels × prompts)."""
import glob, json, os, sys
from wtm.run import job_key

out, levels, prompts, scen, models = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4], sys.argv[5:]
sids = [json.loads(l)["id"] for pat in scen.split() for f in sorted(glob.glob(pat)) for l in open(f) if l.strip()]
present = {json.loads(l)["key"] for l in open(out) if l.strip()} if os.path.exists(out) else set()
short = 0
for m in models:
    want = {job_key(m, s, lv, p, "decision", 0) for s in sids for lv in levels for p in prompts}
    have = len(want & present)
    print(f"{'EVAL_COMPLETE' if have == len(want) else 'EVAL_INCOMPLETE'} {m} {have}/{len(want)}")
    short += have < len(want)
sys.exit(1 if short or not sids else 0)
PY
COUNT_RC=$?
[ "$RC" = 0 ] && [ "$COUNT_RC" = 0 ] || exit 1
echo EVAL_DONE
