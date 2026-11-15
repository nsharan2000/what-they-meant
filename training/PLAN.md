# Training plan — APPROVED by the user 2026-11-14 04:06 IST ("Please proceed you have my approval"; dataset v1.2.1)

Core model: CohereLabs/tiny-aya-global (3.35B, Cohere2). Weights: the ungated mirror 1-800-LLMs/tiny-aya-global, VERIFIED tensor-for-tensor
identical to Cohere's official ungated bf16 GGUF (log 2026-11-12 21:54); cite base_model CohereLabs/tiny-aya-global. Official gate (Q1) pending.

## Data (v1.2.1, approved)
Group split by source post (dataset/splits.json): train 76 pairs (27 seed + 49 variants) · val 17 (5 + 12) · test 41 (13 + 28, never
trained on). Views (training/build_datasets.py; deterministic): every look-up step + the decision turn; renderings L0r / L1 / L2 (+ L3
when a noisy request exists); prompt P0 (no instruction) ~70 % / P2 (one-or-two-line intent hint) ~30 % — never the long golden block
(user, 2026-11-13). v1.2.1 consistency filters (build_report.json): one normalised prompt → one target (S42 look-up rows dropped; a
conflicting decision row fails the build), pushback rows out of DPO/GRPO, targets may not cite identifiers the prompt never shows, DPO
rejected = an observed base failure at the same level scoring ≤ 0 by wtm.reward (else synthetic). Spec fixes: S14B checks the charge
status first; V344A handle made consistent (dataset/fix_v121.py). Counts: SFT 1,000 / val 224 / DPO 508 / GRPO 510 → CuratorKIT
(exact dedup + SecretsGate, twin-safe) → training/from_curated.py (the ORIGINAL rows CuratorKIT kept; its exports flatten the prompt
and collapse whitespace, so they decide selection only) → /workspace/training/data_v121_curated: SFT 904 · val 224 · DPO 496 · GRPO 496.
Sequence limit 12,288 tokens everywhere (longest row 7,581 tokens; vLLM serves at 16,384).

## Safety rules for every GPU job (incident 2026-11-14: an unguarded DPO stage exhausted the shared host)
- Every stage runs under training/memguard.py (own process group, oom_score_adj 1000, MemAvailable floor 16 GB, per-stage budget,
  exit 137 on a kill, 75 on refusal, one guarded job at a time via /workspace/.gpu_job.lock) + guards.apply_mem_cap() (torch cap via
  WTM_GPU_CAP_GB). Stage 0 verified on the GB10: CUDA memory lowers MemAvailable but is invisible to RSS and the container cgroup.
- training/guards.py checks each trainer's own prepared dataset before step 1 (no dropped/truncated rows; trained span = target +
  <|END_RESPONSE|><|END_OF_TURN_TOKEN|>, token ids; DPO prompt ends with the generation prompt). Trainers resume from checkpoints.

## Methods
- M1 SFT (AgentTune `create_distill_trainer` + completion-only factory, TRL SFTTrainer): LoRA r16 on the 7 projections, lr 1e-4,
  batch 1 × grad-accum 16, 2 epochs (time from the guarded smoke; 2026-11-14 rate ≈ 2.8 ms per trained token → ~6.5–8 h for 2 epochs).
  Targets: seed ideal moves (user-authored) and open-teacher variant targets (Q5: never Claude-written).
- M2 GRPO (AgentTune `create_agentic_trainer("grpo", backend="trl")`, vLLM colocate 0.20, enforce_eager): warm start from merged M1,
  reward wtm/reward.py (+ −1 for real truncation), β = 0, loss dapo, 4 prompts × 8 generations per optimizer step (on-policy),
  ~40 steps (from the smoke's s/step). TRL's truncation mask is OFF: Tiny Aya also stops on <EOS_TOKEN> 3, which TRL 1.13 counts as
  "clipped" — the 2026-11-14 smoke masked every completion and learned nothing.
- M3 DPO (AlignTune `create_rl_trainer(algorithm="dpo", backend="trl")`, DataManager bypassed — it flattens chat rows into plain text):
  from the BASE (method comparison), batch 1 × grad-accum 16, precompute_ref_log_probs, β 0.1, 1 epoch.
- Serving for evaluation: every arm is base + ONE LoRA: m1 = m1_sft/final, m2 = m2_grpo/stacked (training/stack_lora.py: M1+M2 as one
  rank-32 LoRA), m3 = m3_dpo/final. Adapters are served under content-hashed names (response cache can never mix versions).

## Evaluation (experiments/training_eval/)
run_eval.sh (one vLLM server, base + adapters): held-out test split (41 pairs) and, as a second run, the held-out hard pairs
(dataset/eval/hard_test_*: 13 generated pairs from test clusters; the 27 train-cluster hard pairs are reported as in-distribution only)
× L0r/L1/L2/L2s × P0 (primary) / P2, decision mode, temperature 0. Metrics: paired pass (strict, judged), informed-ask, harm, needless
ask, protocol validity; Opus judge on ambiguous decisions (blind to arm); AuditKIT re-scoring + ak.compare (auditkit_eval.py);
capability retention via ak.run_lmeval (IFEval, GSM8K, Global-MMLU, MGSM; capability_eval.py); SafeTune drift (safety_eval.py);
AlignTune AlignmentAuditor drift (alignment_audit.py); ROI (experiments/roi/roi.py).

## Running it
training/smoke_aya.sh = plumbing check on the longest rows (stage0 → sft → merge → grpo ramp → dpo; weights deleted).
training/run_pipeline.sh = the self-driving chain on the Spark (audits_base → m1_sft → m1_merge → m2_grpo → m2_stack → m3_dpo → eval →
audits_tuned → summary), each stage under memguard, stop at the first failure, resumable; no scheduled wake-ups of the Claude session.
