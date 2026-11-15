# Wait until the v1 pipeline (START_AT=eval run of 2026-11-15 09:54) and every guarded GPU job have exited.
source "$WS/training/chain2_lib.sh"
python3 - <<'PY' || { echo "waiting: a pipeline or guarded GPU job still holds its lock ($(ist_now))"; exit 4; }
import fcntl, sys
for p in ("/workspace/runs/pipeline/.pipeline.lock", "/workspace/.gpu_job.lock"):
    try:
        f = open(p, "a+")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        print("busy:", p)
        sys.exit(1)
PY
if pgrep -f "vllm.entrypoints|vllm serve|EngineCore" >/dev/null 2>&1; then echo "waiting: a vLLM process is still up"; exit 4; fi
python3 - <<'PY'
import json
d = json.load(open("/workspace/logs/pipeline.status"))
print("v1 pipeline ended:", json.dumps(d))
if d.get("state") != "finished":
    print(f"CHAIN2_ATTENTION v1 pipeline ended in state {d.get('state')} at stage {d.get('stage')} rc={d.get('rc')} "
          "(the eval step after GRPO v2 re-runs eval for every arm; cached episodes are kept)")
PY
exit 0
