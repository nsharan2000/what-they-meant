# J4 smoke: 16 rows (13 decision rows), 1 epoch, --decision-weight 10 --keep-epoch-ends on the real base; checks the weighted
# path ran (weighted rows seen), the loss is finite, epoch-1 was saved. Weights deleted afterwards.
source "$WS/training/chain2_lib.sh"
SM="$WS/runs/chain2_smoke/sft_dw"; rm -rf "$SM"
guarded sft_dw_smoke 30 26 30 expandable_segments:True offline -- python3 training/train_sft.py --smoke --model "$BASE" \
    --data "$WS/training/data_v121_curated" --out "$SM" --decision-weight 10 --keep-epoch-ends --fresh
rc=$?
[ "$rc" -eq 0 ] || { attn "J4 smoke failed rc=$rc (logs/chain2/45_j4_smoke.log); J4 will be skipped"; exit 1; }
python3 - "$SM" <<'PY' || { attn "J4 smoke: result checks failed (logs/chain2/45_j4_smoke.log)"; exit 1; }
import json, math, os, sys
d = sys.argv[1]
r = json.load(open(os.path.join(d, "result.json")))
dw = r.get("decision_weight") or {}
print("decision_weight:", json.dumps(dw))
assert dw.get("weight") == 10.0 and dw["table"]["decision_rows"] > 0, dw
assert dw["seen_this_segment"].get("weighted_rows", 0) > 0, dw
assert os.path.exists(os.path.join(d, "epoch-1", "adapter_model.safetensors")), os.listdir(d)
losses = [h["loss"] for h in r["log_history"] if "loss" in h]
print("losses:", losses)
assert losses and all(math.isfinite(x) and x < 50 for x in losses), losses
print("J4_SMOKE_CHECK_OK")
PY
rm -rf "$SM"
