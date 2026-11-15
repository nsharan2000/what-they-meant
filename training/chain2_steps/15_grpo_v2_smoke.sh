# 2-step GRPO v2 smoke on the real merged M1 with every new flag; checks that the scored prompts follow the file order
# (shuffle off) and that completions are logged. Weights deleted afterwards.
source "$WS/training/chain2_lib.sh"
D=$(cat "$CDIR/grpo_v2_data_dir.txt" 2>/dev/null) || skip "no GRPO v2 data (step 10)"
SM="$WS/runs/chain2_smoke/grpo"; rm -rf "$SM"
guarded grpo_smoke 65 54 45 - offline -- python3 training/train_grpo.py --smoke --model "$M1_MERGED" --data "$D" --out "$SM" \
    --no-shuffle --reward-variant v2 --log-completions --lr 5e-5 --fresh
rc=$?
if [ "$rc" -ne 0 ]; then attn "GRPO v2 smoke failed rc=$rc (logs/chain2/15_grpo_v2_smoke.log); GRPO v2 will be skipped"; exit 1; fi
python3 - "$SM/scores.jsonl" "$D/grpo_train.jsonl" <<'PY' || { attn "GRPO v2 smoke: order/logging check failed"; exit 1; }
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()][:8]
def meta(r):
    return r.get("meta") or {}
steps = sorted({meta(r).get("step") for r in recs})
print("score records:", len(recs), "steps:", steps, "first record keys:", sorted(recs[0]) if recs else None)
assert len(steps) == 2, steps
for k, s in enumerate(steps):
    got = sorted({(meta(r)["scenario_id"], meta(r)["level"]) for r in recs if meta(r).get("step") == s})
    want = sorted({(r["scenario_id"], r["level"]) for r in rows[4 * k:4 * k + 4]})
    print(f"step {s}: scored {got}\n        file   {want}")
    assert got == want, "prompt order differs from the file (shuffle not off?)"
assert all(isinstance(meta(r).get("completion"), str) for r in recs), "completion text missing from scores.jsonl"
print("GRPO_V2_SMOKE_CHECK_OK")
PY
rm -rf "$SM"
