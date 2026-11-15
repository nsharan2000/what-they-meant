# Pricing inputs for the ROI comparison (fetched 2026-11-12 ~21:55 IST)

## Self-hosting (Hugging Face Inference Endpoints, AWS) — https://huggingface.co/pricing (VERIFIED, page fetched)
| GPU | memory | $/hour |
|---|---|---|
| NVIDIA L4 ×1 | 24 GB | 0.80 |
| NVIDIA L40S ×1 | 48 GB | 1.80 |
| NVIDIA A100 ×1 | 80 GB | 2.50 |
| NVIDIA H100 ×1 | 80 GB | 4.50 |
Tiny Aya Global (3.35B, bf16 6.7 GB; LoRA merged) fits one L4. Throughput is measured on our DGX Spark (GB10) with vLLM 0.29 and reported
with a sensitivity band (×0.5 for an L4) rather than assumed.

## Closed-source per-decision cost — measured, not assumed
Claude Code reports `total_cost_usd` per call on a list-price basis (`costBasis: "list"`); every Claude-arm episode row stores `cost_usd`,
so the per-decision cost of Haiku 5.5 / Sonnet 5.5 / Opus 5.5 / Fable 5.2 comes straight from experiments/replication/results.
