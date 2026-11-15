# GRPO v2 prompt order (training/build_grpo_v2.py): 36 steps x [A1,B1,A2,B2]. The 44 flagged rows are dropped only if the user
# approved it (questions-for-user Q12 -> training/approvals/exclude44.approved), checked when this step runs.
source "$WS/training/chain2_lib.sh"
OUT="$WS/training/data_v121_grpo_v2"; EXTRA=()
if [ -f "$APPROVALS/exclude44.approved" ]; then OUT="${OUT}_excl"; EXTRA=(--exclude "$EXCLUSION_LIST"); echo "exclusion approved"; fi
python3 training/build_grpo_v2.py --data "$WS/training/data_v121_curated/grpo_train.jsonl" --out "$OUT" --steps 36 \
    ${EXTRA[@]+"${EXTRA[@]}"} || exit 1
echo "$OUT" > "$CDIR/grpo_v2_data_dir.txt"
