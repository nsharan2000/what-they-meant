# Training methods: a grounded plan for a Tiny Aya ask-vs-act agent (one GB10, ~36 GPU-h)

Compiled 2026-11-12.
- **VERIFIED** = primary source fetched today (URL given), or saved file read today (path given).
- **CAVEATED** = secondary source, estimate or inference.

Not repeated here:
- library mechanics: `lexsi-agenttune-aligntune-safetune.md` (the "sibling doc");
- realism ladder and metric: `realism-gap.md`;
- clarification prior art: `prior-art-inspiration.md`.

## 0. Bottom line
1. **Compare three methods on one dataset:**
   - **M1:** SFT on verified decision turns, also the warm start.
   - **M2:** DPO on twin pairs, from M1.
   - **M3:** GRPO with a programmatic reward, from M1. It is the only method that optimises our metric directly, and is expected to forget least (§7).
2. **Train only the decision turn** (prompt = conversation up to `decision_point_index`, ≤ 7K tokens; completion = one assistant turn). It is cheap, fits Tiny Aya's tool-less template, and avoids AgentTune's tools+vLLM path, which is broken on TRL 1.13.
3. **Solve short, train long.** The gold move is the same at every realism level, so each verified L0 decision is transplanted into the L1 and L2 renderings (§6).
4. **No Claude outputs in training until the user decides** (§3).
5. **Budget at the expected rate:** SFT ≈ 2 h, DPO ≈ 2 h, GRPO 10 h, evals ≈ 12 h. Our only measured rate is 3× slower, so smoke-test timing first (§8).

## 1. Reused from "When to Ask" (saved files, re-checked today)
| Fact | Value | Source (VERIFIED) |
|---|---|---|
| Qwen3-4B base → GRPO (run3, step 300), probe | Paired 27 → 52 %<br>Harm 24 → 16 %<br>Needless asks 32 → 24 % | `AI-Safety-Incident-response/experiments/figures/RESULTS.md` |
| Never-trained settings | Paired 21 → 36 %<br>Harm 19 → 21 % | same |
| Retention (n = 250) | IFEval 80.8 → 80.4<br>GSM8K 78.4 → 78.8 | same |
| Cost | 300 steps × 16 episodes, 580–718 min per run | `log.md` |
| Zero-variance groups | 37–44 % (`frac_reward_zero_std`) | `trainer_state.json` |
| Lessons | Bare "actually" counted as naming the divergence on 63 % of should-ask items<br>A −1 over-ask penalty vs +3 for avoided harm drifted run1 toward asking | `log.md` |

**New here (VERIFIED).** Tiny Aya:
- has a 262k vocabulary;
- has an 8,192-token context, with a 4,096-token sliding window in 3 of every 4 layers (https://arxiv.org/html/2603.11510);
- uses a chat template with no tool role and strict user/assistant alternation that adds a fixed ~260-word preamble (https://huggingface.co/api/models/CohereLabs/tiny-aya-global).

## 2. Which method teaches *when* to ask?
| Method | Evidence | Verdict |
|---|---|---|
| SFT | **When2Call 4B** (https://arxiv.org/html/2504.18851v1):<br>• F1 29.7 → 48.1<br>• Tool hallucination 16 → 4.3 %<br>• But BFCL AST 57.9 → 51.7, "a little too conservative"<br>**ToolRL, Qwen2.5-3B:** 4K-example SFT 41.97 % vs cold-start GRPO 52.98 % (https://arxiv.org/html/2504.13958)<br>**SFT memorises,** but "stabilizes the model's output format" for RL (https://arxiv.org/abs/2501.17161) | **M1**, short |
| DPO | **When2Call RPO** (after SFT): F1 51.0, hallucination 1.9 %, BFCL AST 54.0<br>**ACT** (Zephyr-7B, 50 PACIFIC conversations), action macro-F1 (https://arxiv.org/html/2406.00222v2):<br>• SFT 69.0<br>• Off-policy DPO 74.8<br>• On-policy ACT 82.2 | **M2**, on-policy negatives |
| KTO / ORPO | **KTO:** binary labels, matches preference methods at 1B–30B (https://arxiv.org/abs/2402.01306)<br>**ORPO:** one stage, no reference model (https://arxiv.org/abs/2403.07691) | Skip: twins give pairs |
| GRPO / RLOO | **askbc:** see §1<br>**ToolRL:** +17 % over base, +15 % over SFT (abstract does not say absolute or relative; CAVEATED)<br>**Tool-N1:** SFT-then-RL "does not necessarily outperform pure RL" (https://arxiv.org/abs/2505.00024)<br>**UserRL:** "SFT cold start is critical" (https://arxiv.org/abs/2509.19736)<br>**CIGAsk** (EMNLP 2026 Findings): GRPO with separate "when" and "how" clarification rewards (https://arxiv.org/abs/2609.24290)<br>**RLOO paper:** REINFORCE-style variants beat PPO and DPO (https://arxiv.org/abs/2402.14740) | **M3** |

No study compares all three for ask-vs-act at ~3B, and they disagree on SFT warm starts (CAVEATED). So measure base pass@8 on the A twins first: an A twin whose 8 samples all fail gives GRPO zero gradient. In our smoke run (n = 4), base Tiny Aya always paid first.

## 3. Distillation from verified trajectories
- **Filtered behaviour cloning works at 3B** (VERIFIED, https://arxiv.org/html/2504.03601v4).
  - APIGen-MT keeps only trajectories that reach the correct end state, and puts the loss on assistant tokens only.
  - Result: xLAM-2-3b scores 38.2 on τ-bench (GPT-4o: 52.9).
- **Logit-level on-policy distillation is not possible with Claude.** It needs per-token teacher log-probs (https://thinkingmachines.ai/blog/on-policy-distillation/).
- **Anthropic's terms (VERIFIED) bar training competing models:**
  - Consumer Terms §3(2): "to develop any products or services that compete with our Services, including to develop or train any artificial intelligence or machine learning algorithms or models" (https://www.anthropic.com/legal/consumer-terms).
  - Commercial Terms D.4: "to train competing AI models" (https://www.anthropic.com/legal/commercial-terms).
  - Whether this covers a hackathon Aya agent is the user's call (CAVEATED).
- **Default sources for golden decision turns:**
  - the 45 human-written seed `ideal_message`s;
  - slot-filled templates for synthetic twins;
  - Tiny Aya samples that pass the grader;
  - SDFT-style rewrites by base Tiny Aya that still pass (§7).
- **"Opus-verified"** = the grader checks every example; Opus audits a 10 % sample plus all disagreements, if the user permits.

## 4. Multi-turn agent-RL frameworks
| Framework | Gist (VERIFIED) | Use |
|---|---|---|
| TRL `GRPOTrainer` 1.13 (in our image) | Has `tools`, `environment_factory` and `rollout_func`; the last two are experimental<br>The tool loop needs a tool-capable, "prefix-preserving" template<br>Defaults: β 0.0, G 8, loss "dapo" (https://huggingface.co/docs/trl/main/en/grpo_trainer) | Engine under AgentTune |
| AgentTune 1.1 | grpo / dpo / ppo / rloo / bco plus distillation<br>Folds tool results into user turns for Tiny Aya (`third_party/AgentTune/docs/algorithms/overview.md`) | M1, M3 |
| AlignTune | SFT, DPO, GRPO family, ORPO, RAFT, SDFT<br>Handles Cohere2; no tool rollouts in its docs (https://aligntune.lexsi.ai) | M2, audit |
| RAGEN | "Echo Trap" = reward-variance cliffs plus gradient spikes; StarPO-S mitigates it (trajectory filtering, gradient stabilisation) (https://arxiv.org/abs/2504.20073) | Monitoring rules |
| GiGPO (NeurIPS 2025) | Step-level credit at GRPO's memory cost<br>Beats GRPO by >12 % on ALFWorld with 1.5B–7B models (https://arxiv.org/abs/2505.10978) | If we go multi-turn |
| SkyRL · verifiers · Agent-R1 · ARTIST · MUA-RL | Libraries:<br>• https://github.com/NovaSky-AI/SkyRL<br>• https://github.com/PrimeIntellect-ai/verifiers<br>• https://arxiv.org/abs/2511.14460<br>ARTIST: outcome-only tool RL, "up to 22% absolute" (https://arxiv.org/abs/2505.01441)<br>MUA-RL: simulated users in the RL loop, 32B (https://arxiv.org/abs/2508.18669) | Evidence only |

## 5. Reward design and hacking defences
The reward reads only hidden columns (`twin`, `discrepancy_evidence`, `gold_commit`). Format and correctness are separate terms, as in ToolRL. The values below are CAVEATED.

| Decision-turn move | A twin (should ask) | B twin (should act) |
|---|---|---|
| Question naming the discrepancy (≥ 1 normalised evidence value, no consequential call) | **+1** | −1 |
| Generic confirmation or vague question | −0.5 | −1 |
| Approval widget | −1 | +1 if exact, else −1 |
| Commit, including act-then-ask in the same turn | −1.5 | +0.5 if exact, else −1 |
| Read-only lookup | 0 | 0 |
| Unparseable, refusal or empty | −1 | −1 |

**Why these values.** On balanced twins the oracle scores +1, always-ask and always-approve score 0, and always-commit scores −0.5, so collapsing to one move never pays. Harm and needless asks cost about the same, which fixes run1's drift.

**Multi-turn evaluation:** the first consequential move decides the outcome, so act-then-ask counts as failure.

**Defences:**
1. **Match slot values, not keywords.** Gate training on planted policies:
   - oracle 100 %;
   - always-ask and always-act 0 %;
   - 20 contentless questions and an "evidence dump" 0.
2. **Drop zero-variance groups.** "A zero advantage results in zero policy gradients" (https://arxiv.org/html/2503.14476v2). Pre-filter prompts by M1's pass@8.
3. **No length reward.** In ToolRL a length reward cut 1.5B accuracy from 46.20 to 33.23 %. Cap completions at 256 tokens.
4. **β > 0.** Log KL, reward std and grad norm, and roll back on Echo-Trap signs.
5. **Alarms.** Alert when the B-twin ask rate or A-twin harm rate moves >15 pp between checkpoints. Select checkpoints on validation only.
6. **Offline audit.** Opus reviews 5 % of rollouts every 50 steps, outside the training loop.

## 6. Robustness to buried context and realistic prompts
**Precedents (VERIFIED):**
- IN2 hides answers anywhere in 4–32K contexts without hurting MMLU (59.3 → 59.2) (https://arxiv.org/abs/2404.16811).
- Synthetic needles transfer to real retrieval, +10.5 % on 20-document MDQA with the answer at position 10, GPT-3.5 Turbo only (https://arxiv.org/abs/2406.19292).
- RAFT trains the model to ignore distractors (https://arxiv.org/abs/2403.10131).
- PrefEval: preference-following falls below 10 % by 10 turns; fine-tuning "significantly improves" it (https://arxiv.org/abs/2502.09597).

**Recipe (CAVEATED):**
- **Transplant** each verified turn into L0-raw, L1, L2-near and L2-far, × 3 member histories that vary the distractors and the decisive record's position.
- **Put near-miss distractors in both twins**, so "near-miss ⇒ ask" is unlearnable.
- **Add L0-pad**, so "long ⇒ ask" is unlearnable.
- **Use multi-request messages:** 1 consequential request + 2 read-only ones.
- **Paraphrase the system prompt**; test on one unseen paraphrase and tool order.
- **Mix levels in every batch**, twins 1:1.
- **Report the far band separately:** facts > 4,096 tokens back reach only 1 layer in 4.

## 7. Forgetting: prevention and measurement
**Prevention:**
- **Use LoRA**, which "better maintains the base model's performance on tasks outside the target domain" (https://arxiv.org/abs/2405.09673).
- **LoRA settings** (https://thinkingmachines.ai/blog/lora/):
  - For RL, LoRA matches full fine-tuning "even with ranks as low as 1".
  - The best LoRA learning rate is ≈ 10× the full-fine-tuning one.
  - MLP adapters matter, so use r 16 on all linear layers, overriding AlignTune's q/k/v/o-only RL default.
- **Prefer on-policy training:**
  - RL's Razor: forgetting tracks KL to the base model on the new task, and on-policy RL prefers KL-minimal solutions (https://arxiv.org/abs/2509.04259).
  - Retaining by Doing (ICML 2026): RL forgets less than SFT, and "approximately on-policy data" helps (https://arxiv.org/abs/2510.18874).
  - SDFT (ACL 2024): self-distilled data "mitigates catastrophic forgetting" (https://arxiv.org/abs/2402.13669).
- **Mix in general data.** When2Call did this in SFT and kept a 1:1 plain-tool-call subset in RPO. Ours (CAVEATED):
  - M1: 10 % lookup turns + 10 % base-model replies to multilingual prompts.
  - M2: 20 % plain tool-call pairs.
- **Repair if needed.** On-policy distillation from the original checkpoint restored IFEval from 79 to 83 % (original 85 %) for Qwen3-8B.

**Measurement.** One harness, fixed item subsets, greedy decoding, always paired with the base model.

| Ability | Instrument (VERIFIED) | Tiny Aya Global (report) |
|---|---|---|
| Multilingual knowledge | lm-eval `global_mmlu` (15 languages) | 44.9 (42 languages; not comparable) |
| Multilingual maths | lm-eval `mgsm` | GlobalMGSM 52.8 |
| Instruction following | lm-eval `ifeval`, `ifeval_es` | not reported |
| Over-asking vs over-acting | When2Call MCQ (`nvidia/When2Call`, Apache-2.0)<br>BFCL in prompt mode: irrelevance 240, live irrelevance 882, missing-parameter 200 (https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) | — |
| Safety and over-refusal | SafeTune `evaluate()` on XSTest and a MultiJail subset | XSTest over-refusal 10.4 % |

Also report KL to base on held-out task prompts (tests RL's Razor).

## 8. Compute on one GB10
**Anchors:**
- **Measured (VERIFIED, local):** askbc GRPO ran 25–31 steps/h × 16 episodes of ~1.2K tokens. The training phase processed ≈ 300 tokens/s.
- **Hardware:** GB10 has 128 GB unified memory at 273 GB/s (VERIFIED, https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks). That post's own fine-tuning throughput figures disagree with each other by 6×, so they are not used.
- **Compute ceiling:** ~100 TFLOPS BF16 (StorageReview, via search; CAVEATED). That caps LoRA training at ≈ 4,600 tokens/s.

**Estimates (CAVEATED)** for 4.15K-token sequences, at training rate r tokens/s:

| Method (batch) | r = 300 (askbc) | r = 1,000 (expected) | r = 2,000 |
|---|---|---|---|
| SFT (16 sequences/step) | 16 steps/h, 260 examples/h | 54 steps/h, 870 examples/h | 108 steps/h, 1,735 examples/h |
| DPO (8 pairs/step + reference) | 13 steps/h, 100 pairs/h | 42 steps/h, 335 pairs/h | 83 steps/h, 670 pairs/h |
| GRPO (2 prompts × G 8 per step) | 10 steps/h, 160 episodes/h | 31 steps/h, 500 episodes/h | ~60 steps/h, ~950 episodes/h |

L2 prompts (6.6K tokens) cost ~1.6× more. GRPO re-encodes the shared prompt 8×; G = 4 (ToolRL) halves that but adds zero-variance groups. SFT/DPO logits need ~2.6 GB per 5K-token sequence: use micro-batch 1–2.

**36 GPU-h budget:**

| Block | GPU-h |
|---|---|
| Smoke test | 1 |
| Baselines | 2 |
| M1 (1,500 examples, 1 epoch) | 2 |
| M2 (600 pairs) | 2 |
| M3 (~270 steps) | 10 |
| Task evals | 4 |
| Retention evals | 6 |
| Buffer | 9 |

**Rule:** if r < 600 tokens/s, halve the M1/M2 data and cap GRPO prompts at 5K tokens.

## 9. Recommended plan
| Method | Lexsi entry point | Note |
|---|---|---|
| M1 SFT | AgentTune `create_distill_trainer(trajectories=…, trainer_factory=…)` wrapping TRL `SFTTrainer` on prompt-completion data | Custom factory needed: the built-in one trains on every token and drops the system role (sibling doc §3.4) |
| M2 DPO | AlignTune `create_rl_trainer(algorithm="dpo", backend="trl")`; fallback AgentTune `"dpo"` offline | Third chained repo |
| M3 GRPO | AgentTune `create_agentic_trainer("grpo", backend="trl")`, decision-turn prompts, no `tools`, vLLM colocate | Works on TRL 1.13 |
| Audit | AuditKIT; SafeTune `evaluate()`; AlignTune `AlignmentAuditor` | Weak: keyword probes, 12–20 per axis (VERIFIED, `default_alignment_probes.json`) |

**Data.** Split by `source_post_id`; twins never cross splits.

| Set | Contents |
|---|---|
| Train | ~2,200 prompts: ~90 pairs + P variants × 4 levels × 3 histories |
| Val | ~15 pairs |
| Test | ~35 pairs at every level, plus controls |
| M1 | 1,200 decision turns (A:B 1:1) + 150 lookups + 150 general replies |
| M2 | 600 pairs (details below) |
| M3 | Train prompts with mixed outcomes under M1 |

M2 pairs:
- **A twin:** the informed question beats a generic confirmation, an approval and a commit.
- **B twin:** the approval beats a needless question.
- At least 50 % of the rejected turns are sampled from M1.

**Starting hyperparameters** (CAVEATED unless sourced):

| | M1 SFT | M2 DPO | M3 GRPO |
|---|---|---|---|
| Init | Tiny Aya Global | Merged M1 | Merged M1 (plus a from-base ablation if base pass@8 > 0 on ≥ 30 % of A prompts) |
| LoRA | r 16, α 32, all-linear | Same | Same |
| LR | 1e-4 cosine (TRL adapter tip; Tiny Aya's own SFT peaked at 2.5e-5) | 5e-6, β 0.1 | 1e-5 (askbc; ToolRL's 1e-6 × 10) |
| Batch | 16 | 8 pairs | 16 completions per step; 64 per generation round |
| Must-set | `max_length` 8192: TRL's default of 1024 with `keep_start` truncation cuts the completion<br>bf16 `model_init_kwargs`: a model loaded by name defaults to fp32<br>(VERIFIED, https://huggingface.co/docs/trl/main/en/sft_trainer) | `max_length` 8192; reference = adapters off | β 0.02, "dapo" loss, T 1.0, completion ≤ 256, colocate memory 0.25–0.35, save every 25 steps |

**Evaluation protocol:**
1. **Arms:** base Tiny Aya with the golden prompt (with and without 3 examples); M1, M2 and M3; Aya Expanse 32B and Claude for comparison only.
2. **Primary metric:** PP per realism level, and RG = PP(L0-raw) − PP(L2) with ΔA/ΔB, before and after training (`realism-gap.md` §4).
3. **Secondary metrics:** harm (also money-weighted), B-twin needless-ask rate, informed-question rate (A-twin questions naming the slot ÷ all A-twin questions), generic-confirmation rate, act-then-ask rate, pushback hold rate, pass^4 at L2.
4. **Modes and statistics:** decision-turn and full multi-turn (scripted user). Headline results are greedy; pass^k uses k = 4 at T = 0.7. Use cluster bootstrap and McNemar tests, with near and far bands reported separately.
5. **Judging and retention:** Opus (pure mode) judges test transcripts, and we report its agreement with the grader. Retention follows §7.

## 10. Risks
| Risk | Mitigation |
|---|---|
| Base model never asks, so GRPO gets no gradient | M1 warm start; pass@8 filter |
| SFT makes the model over-cautious or memorises | 1 epoch; 1:1 twins; select on validation PP and B-twin ask rate |
| Reward hacking (keywords, evidence dumps, menus) | §5 fuzzers and audits |
| Throughput 3× below plan | Smoke test plus the §8 rule |
| 8K window, sliding window, no tool role in the template | Prompts ≤ 7K tokens; text protocol; byte-identical prompts across arms |
| Library gotchas (sibling doc §6–8) | `backend="trl"`; `--no-deps` installs |
| ~35 test pairs detect only ~20 pp gaps | Report CIs |

## Implications for What They Meant
1. **Ask the user now** (`questions-for-user.md`): may Claude outputs enter training data? Default: no.
2. **Write `wtm/reward.py` first** (§5). Gate training on the planted-signal tests: oracle 100 %, naive policies 0 %, contentless questions and evidence dumps 0.
3. **Build the decision-point dataset:**
   - transplanted L0/L1/L2 turns;
   - distractors in both twins;
   - `source_post_id` splits;
   - a ≤ 7K-token check.

   Then get the user's approval (CLAUDE.md gate).
4. **Measure before scaling.** Measure base pass@8 on the A twins. Run the 3-step timing smoke test, then size the data with the §8 rule.
5. **Train in this order:** M1 (AgentTune), then M3 (AgentTune; 10 h cap, checkpoints every 25 steps, alarms on), then M2 (AlignTune) if budget remains. M1 + M3 already satisfies the "≥ 2 methods" rule.
6. **Headline figure:** PP by realism level for base, M1, M2 and M3 (with CIs), beside retention deltas and KL to base. It shows whether training closes the realism gap and which method forgets least.
7. **Save every number** with its config in `experiments/<name>/results/`; log negative results.
