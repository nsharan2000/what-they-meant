# E0 (audit): weight-space size of M3 (DPO from base, lr 5e-6) vs M1 and the lr-5e-6 GRPO adapter. CPU only, ~1 min.
source "$WS/training/chain2_lib.sh"
G1="$WS/runs/m2_grpo/final"; [ -d "$WS/runs/m2_grpo_v1_lr5e-6/final" ] && G1="$WS/runs/m2_grpo_v1_lr5e-6/final"
CUDA_VISIBLE_DEVICES= python3 experiments/training_eval/adapter_effect.py --adapter "m1=$M1" --adapter "m3=$M3" \
    --adapter "m2_grpo_v1=$G1" --out experiments/training_eval/results/train_logs/adapter_effect_m3.json
