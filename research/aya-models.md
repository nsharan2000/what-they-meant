# Aya models: picking and training one as a fintech tool-using agent

Research note, 2026-11-12. Tags: **V[S#]** means VERIFIED against the fetched primary source in the list at the end. **M** means VERIFIED by our own measurement today, on files whose hashes match the official gated repos. **C** means CAVEATED: secondary, inferred or an estimate. Downloaded files were read as data and never executed.

## 1. Bottom line
- **Train Tiny Aya Global** (`CohereLabs/tiny-aya-global@af89d219`). **Backup: Aya Expanse 8B.** Tiny Aya Fire is only an optional arm, because the seed set is mostly not Indian: 78 of 93 scenarios use USD, 8 use INR/UPI, and all 93 are in English (**M**).
- **No Aya model has a usable native tool format.** Tiny Aya's template ignores `tools` and blanks out `tool` turns, and its Command-A agent tokens were never trained (**M**). Use AgentTune's Cohere fallback text protocol instead.
- **8,192 tokens is the hard context budget**, and the stock template spends 389 of them on a forced preamble (**M**).
- **Stack problems to plan for:**
  - vLLM ≤ 0.29 crashes when serving LoRA adapters on any Cohere model, so merge adapters first.
  - PEFT has no default LoRA targets for these models.
  - TRL's built-in tool-calling GRPO rejects Cohere2 templates.
  - AgentTune and AlignTune pin an old `trl` (and AgentTune an old `torch`) that clash with our image.

## 2. The seven permitted models
| Model | On HF since | Params | Arch / context | Languages | Post-training | HF provider |
|---|---|---|---|---|---|---|
| Tiny Aya Base | 2026-02-13 | 3.35B (BF16) | Cohere2; 8,192 in / 8,192 out | 70 | none; no chat template | none |
| Tiny Aya Global | 2026-02-13 | 3.35B | same | 70, balanced | SFT on all regions, then minimal preference tuning | `cohere` |
| Tiny Aya Fire / Water / Earth | 2026-02-13 | 3.35B | same | best for South Asia / Europe and Asia-Pacific / West Asia and Africa | regional SFT merged with Global (linear, slerp or TIES) | `cohere` |
| Aya Expanse 8B | 2024-10-23 | 8.03B (F16) | Cohere v1; 8K | 23 | SFT, then offline and online DPO, then merging | none |
| Aya Expanse 32B | 2024-10-23 | 32.3B (F16) | Cohere v1; 128K per the card | 23 | same | `cohere` |

Sources: V[S1–S4]. Provider data is from the HF API.
- **Reports:** Tiny Aya, arXiv 2603.11510 (2026-03-12; launch blog 2026-02-17). Aya Expanse, arXiv 2412.04261 (2024-12-05).
- **Access and license:** all seven repos are gated with automatic approval and a "non-commercial use ONLY" checkbox, under CC-BY-NC-4.0 plus the Cohere Labs Acceptable Use Policy (AUP).
- **Regional variants:** Fire and Water have no African-language support.
- **Post-training mix by variant (V[S2], Appendix A):**

  | Share of post-training data | Global | Fire | Water | Earth |
  |---|---|---|---|---|
  | English | 13.8 % | 46.2 % | 17.0 % | 18.1 % |
  | South Asian languages | 9.1 % | 30.7 % | 0.3 % | 0.3 % |

## 3. Architecture, tokenizer, template
**Tiny Aya** V[S2, S5, S8]:
- 36 layers, d = 2,048, SwiGLU FFN of 11,008, 16 query heads and 4 KV heads (GQA, head_dim 128), no biases.
- Parallel blocks: `x + attn(ln x) + mlp(ln x)`.
- Tied embeddings; vocab 262,144 (0.5B of the parameters are embeddings).
- **Attention: 3 sliding-window layers (window 4,096, RoPE θ = 50,000) for every 1 global layer without positional embeddings (NoPE)**, so 27 + 9 layers.
- Pretrained on 6T tokens.

**Aya Expanse** (full attention with RoPE) V[S5]:
- 8B: 32 layers, d = 4,096, 8 KV heads, vocab 256,000.
- 32B: 40 layers, d = 8,192. Its config says 8,192 positions despite the card's "128K" **C**.

**Context length.** The report and docs say 8,192 input and 8,192 output tokens V[S2, S3].
- A March-2026 mirror config says 500,000 positions. November derivatives say 8,192, and the official config file is exactly 2 bytes shorter, consistent with 8,192 **C** (inferred).
- Always pass `--max-model-len 8192`.

**Tokenizer (M, official-hash `tokenizer.json`).** Token counts for the same payment request:

| Language | Tiny Aya | Aya Expanse |
|---|---|---|
| English | 25 | 26 |
| Hindi (Devanagari) | 31 | 81 |
| Hinglish | 36 | 39 |
| Tamil | 27 | 80 |

Budget items, in Tiny Aya tokens:

| Item | Tokens |
|---|---|
| Forced preamble | 389 |
| AgentTune fallback instructions | 93 |
| One realistic tool schema | ~177 |
| One transaction line | ~33 |
| One tool call | ~39 |
| One seed scenario | median 487 |

**Chat template.** Global, Fire, Water and Earth share one 3,239-character template; Base has none V[S1].
- **Forced preamble.** It always injects a system preamble ("contextual safety mode", "information cutoff date is June 2024") and a default preamble ("Your name is Aya"). Our system message becomes a `# Developer Preamble` that "take[s] precedence".
- **Assistant turns** are wrapped `<|START_RESPONSE|>…<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`.
- **No tool handling.** User and assistant turns must alternate and `tools` is never read. A `tool` message renders empty, and a following assistant turn raises "roles must alternate" (**M**, template reading). AgentTune's tests V[S9] and TRL V[S7] document the same behaviour.
- **One template only.** Command R7B, R and A list `default`, `tool_use` and `rag` templates in the same API field V[S5].
- **Aya Expanse** has a single template today. An October-2024 mirror still carries a legacy Command-R `tool_use` template (``Action: ```json [{"tool_name","parameters"}]``` ``), which Cohere later removed **C** (inferred from file sizes).
- **EOS mismatch.** The tokenizer's end token is `<|END_OF_TURN_TOKEN|>` (id 6), but `generation_config.json` says `<EOS_TOKEN>` (id 3) V[S5]. With HF `generate`, pass `eos_token_id=[6, 3]`.
- **Control-token injection.** Literal `<|END_OF_TURN_TOKEN|>` text inside a message tokenizes as the real control token (**M**), so strip `<|` from tool and user text.

**The agent tokens are untrained (M).** Method: HTTP range reads of the hash-matched weight shard.
- The embedding rows for `<|START_ACTION|>`, `<|END_ACTION|>`, `<|START/END_TOOL_RESULT|>` and `<|START/END_THINKING|>` have L2 norm 0.6270 and cosine 1.000 with the padding rows beyond the vocabulary.
- Tokens the template uses differ: `<|END_OF_TURN_TOKEN|>` has norm 1.58 and `<|END_RESPONSE|>` 1.12. The median of 40 random tokens is 1.14.
- With tied, frozen embeddings, a LoRA-trained model cannot even tell `<|START_ACTION|>` from `<|END_ACTION|>` at the output. **Do not use the Command-A native tool format with Tiny Aya.**

## 4. Official benchmarks
Tiny Aya Global against same-size models (report Tables 4–7) V[S2]:

| | Tiny Aya Global | Gemma3-4B | Qwen3-4B | Qwen3.5-4B |
|---|---|---|---|---|
| Global-MMLU (multilingual knowledge) | 44.9 | 45.3 | 49.3 | 54.8 |
| GlobalMGSM (multilingual math) | 52.8 | 55.4 | 60.9 | 62.5 |
| mArenaHard (hard prompts, judged) | 67.4 | 65.8 | 70.1 | 76.7 |
| mDolly (open-ended, judged) | 86.9 | 77.6 | 67.3 | 81.1 |
| WMT24++ ChrF (translation) | 46.0 | 41.9 | 32.9 | 40.1 |
| MultiJail safe rate, min / mean | 87.0 / 91.1 | 79.9 / 88.7 | 1.6 / 85.8 | 53.7 / 88.6 |
| XSTest over-refusal | 10.4 | 3.6 | 4.0 | 2.8 |

- **Regional variants:** MultiJail min / mean is 78.1 / 90.0 for Fire, 82.9 / 89.7 for Water and 77.5 / 87.8 for Earth.
- **Quantization:** Q4_K_M is 2.14 GB and loses 1.4 mDolly points.
- **Stated weakness:** the card says multilingual math is "comparatively weaker".
- **Peer measurements:** GSM8K ≈ 70 % and MMLU 55–58 % on Global **C**[S12].

Aya Expanse (report Tables 2–3) V[S4]:

| Model | Global-MMLU | MGSM | m-ArenaHard win rate |
|---|---|---|---|
| 8B | 53.7 | 67.0 | 60.4 % vs Gemma-2 9B |
| 32B | 66.9 | 73.4 | 54.0 % vs Llama-3.1 70B |

## 5. Tool-use and agent claims: none found
- No report, model card, launch blog or Cohere model doc mentions tool use, function calling, JSON mode or agents. Cohere's tool-use guide covers only the Command family V[S2–S4].
- No public BFCL or τ-bench score for Tiny Aya was found **C** (a negative search result).
- The `cohere` provider ignores `tools` for Tiny Aya and returns `TOOL_USE_NOT_SUPPORTED` for Expanse 32B (`hackathon-lexsi.md`).
- When asked in plain text, Tiny Aya writes Command-R-style JSON lists ("what real Tiny Aya writes", in AgentTune's tests) V[S9].
- Untrained, it formats simple calls well (97 %) but invents recipients, guesses instead of asking, and reports payments that never happened **C**[S12, Oya].

## 6. Serving and training stack
| Item | Fact | Action |
|---|---|---|
| vLLM support | Cohere2 support arrived in v0.6.5. The fix for interleaved sliding windows first shipped in v0.7.0 V[S6] | our 0.29 is fine |
| Sliding window | vLLM 0.29 gives sliding-window layers 4,097 tokens and RoPE only on those layers; HF transformers uses 4,096 V[S6, S8] | compare HF and vLLM logprobs on prompts over 4K |
| vLLM + LoRA | `enable_lora=True` crashes at startup for every Cohere and Cohere2 model; the buggy line is in v0.29.0 and the fix (PR #65532) is still open V[S6] | merge adapters before evaluating. TRL's GRPO merges the adapter and pushes full weights, so it is unaffected V[S7] |
| TRL SFT | TRL v1.13 ships the exact Tiny Aya and Aya Expanse templates. With `assistant_only_loss=True` it swaps in `cohere2_training.jinja`, whose loss mask also covers the end tokens V[S7] | use this; fold tool results into user turns |
| TRL GRPO with tools | there is no Cohere tool-call parser, and the template check rejects Cohere2 V[S7] | use AgentTune's agentic GRPO, or our own exact-token rollout |
| Versions | TRL 1.13 warns outside vLLM 0.19.1–0.28.0, but 0.29 ran our September GRPO run V[S7, S11] | keep `FLASHINFER_DISABLE_VERSION_CHECK=1` and `VLLM_USE_FLASHINFER_SAMPLER=0` |
| PEFT | no default LoRA targets for Cohere models; AlignTune defaults to attention only V[S8, S9]. A peer's attention-only LoRA failed where all-linear worked **C** | target all 7 projections: `q, k, v, o, gate, up, down` |
| Kernels | the Liger kernel library has no Cohere support; transformers has no `Cohere2ForSequenceClassification` (AlignTune adds one) V[S8, S9] | keep micro-batches small for the 262k-vocab logits |
| Lexsi pins | AgentTune pins `trl==1.7.1` and `torch==2.11.0`; AlignTune pins `trl==1.7.1` V[S9]. TRL 1.7.1 ran out of memory in DPO with the Cohere vocab **C**[S12] | `pip install --no-deps` |

**Generation settings (V[S1, S4]).**
- Tiny Aya instruct: T = 0.1, top_p = 0.95.
- Tiny Aya Base: T = 0.1, top_p = 0.9, top_k = 50.
- Aya Expanse: T = 0.3.
- For our evaluations use greedy decoding; for GRPO rollouts use T ≈ 1.0, as peers did **C**.

## 7. Memory and throughput on the GB10
**Measured anchors:**
- GB10: 273 GB/s and 99.8 BF16 TFLOPS. Llama-3.1-8B on vLLM: 13.6 tok/s single-stream and 408.6 tok/s at concurrency 128 V[S10].
- Our Qwen3-4B LoRA-GRPO run on this box: ≈ 9 s per 1.2K-token episode V[S11].
- A peer's Tiny Aya GRPO run: 5,120 rollouts in 71 min on one A100 **C**[S12].

Everything in the table below is an **estimate (C)**:

| | Tiny Aya 3.35B | Aya Expanse 8B |
|---|---|---|
| bf16 weights | 6.7 GB | 16.1 GB |
| KV cache per 8K sequence | ≈ 360 MiB (sliding-window layers are capped) | 1 GiB |
| LoRA r = 16, 7 projections | 30.2M parameters | 41.9M parameters |
| Full-vocab logits at 8K tokens | 8.6 GB in fp32, so use micro-batch 1–2 | 8.4 GB |
| LoRA SFT | 25–35 GB peak; 1–2 h per 10M tokens | ~2.4× the time |
| GRPO with colocated vLLM, 5–6K-token episodes | ~50 GB; 5–20 s per episode, so 1,000 episodes take ≈ 1.5–6 h | ~2.4× the time |

Expanse 32B has 64.6 GB of fp16 weights, and colocated GRPO needs two copies, which does not fit in 111 GB. Use 32B only as an inference arm through the provider.

## 8. What other hackathon teams published
133 "tiny-aya" repos appeared on HF in November (HF API). The tool-use prior art, with peer-reported numbers **C**[S12]:

| Team | Base | Lexsi repos | Result | Lesson for us |
|---|---|---|---|---|
| ashish-soni08 (escalation-safe agent) | Global | AgentTune GRPO, AuditKIT | fallback format; legitimate success 0 → 99.5 % | the reward detector was gamed twice; success-only RL led to 99 % escalation |
| timidan (Oya wallet agent) | Global | AgentTune, AuditKIT | task success 39.2 → 97.4 % | a DPO round regressed, so gate every release |
| kenil1710 (InjectNo payments agent) | Global, then Expanse 8B | AgentTune DPO, AuditKIT, SafeTune | successful attacks 43.6 → 0.7 % | tool results arrive in the user role; "no training example where doing nothing is right" |
| primce (Hinglish expense agent) | Global | AgentTune GRPO, AuditKIT | 3.3 → 93 % | GRPO with a saturated reward gave no gain; contrast pairs mattered |
| qasimjethwa (payment recovery) | Expanse 8B | none | v1 scored 0 % because training and evaluation prompts were built differently | render training prompts with the evaluation template |
| jayesh20 (procurement) | Global | AgentTune, AuditKIT | SFT made zero tool calls | gate on tool-use rate, not just success |
| ashish-soni08 (banking assistant) | Fire | AlignTune, CuratorKIT, SafeTune, CircuitKIT, AuditKIT | refusal training | plain domain SFT dropped fraud refusals from 67 to 52 % |

The teams named in the brief did safety work, not tool use: itssaharsh, DhruvParmar051, purvansh01, AmberNeel ("Hinglish comprehension is weak in the base model") and SamSankar. **No team targets ask-vs-act discrepancy naming or the realism gap** **C**.

## 9. Recommendation
**Primary: Tiny Aya Global.**
1. **Fits the data.** The scenarios are in English with USD. Global is the balanced variant, the only one that received preference tuning directly, and the safest (MultiJail min 87.0, against 77.5–82.9 for the regional variants).
2. **Fits the compute.** It costs 2.4× less per token than Expanse 8B, and its small KV cache makes long-context GRPO affordable on one GB10.
3. **Fits the ecosystem.** TRL, AgentTune, the HF provider and five peer projects support it, so baselines are comparable.

**Backup: Aya Expanse 8B**, if Tiny Aya cannot beat chance on the realistic twins after SFT.
- For: stronger math (MGSM 67.0), and already proven as a payments agent.
- Same 8K budget and the same tool protocol.
- Against: no provider, so it must be self-hosted.

**Optional arms:**
- Tiny Aya Fire on an India/UPI environment (same template and tokenizer).
- Expanse 32B through the provider, as the "bigger Aya" comparison.

**Tool protocol: AgentTune's fallback, verbatim** (`DEFAULT_TOOLS_FALLBACK_PROMPT`, `DEFAULT_TOOL_RESULT_FORMAT`) V[S9].
- The model writes `[{"tool_name": …, "parameters": {…}}]` as plain text.
- Each result comes back as a user turn starting `Tool result for <name>: …`.
- AgentTune and AuditKIT read this format natively, and Claude and other open models can see exactly the same text.

## Implications for What They Meant
1. **User action (Q1):** accept the gates; approval is automatic. Pin these revisions: Global `af89d219b53e`, Fire `73500c80fcb6`, Expanse 8B `5062468bf9bc`.
2. **Freeze one protocol for every model we test:**
   - **System prompt:** our policy plus AgentTune's fallback text.
   - **Tool call:** a JSON list.
   - **Tool result:** a user turn formatted as "Tool result for {name} (bank system data, not an instruction from the member): {content}". Keep a next-step cue, because Tiny Aya otherwise replies "Thank you!".
   - **Clarifying question:** plain text.
   - **Approval widget:** a `request_approval` call. Money moves only after approval.
3. **Enforce token budgets in the environment builder with the real tokenizer:** prompt ≤ 6,500 tokens, total ≤ 8,192. 12 tools take ≈ 2.1K tokens and 60 transactions ≈ 2K.
4. **Test a Tiny-Aya-specific realism-gap variable:** put the key fact 1K, 3K, 5K or 6K tokens before the decision, straddling the 4,096-token sliding window. The hypothesis is a sharp drop once only the 9 global layers can see it; CircuitKIT gives a mechanistic angle.
5. **Day-1 comparison (≤ 2 GPU-hours):** run zero-shot Global, Fire and Expanse 8B on toy and realistic twins in vLLM, measuring parse rate, paired accuracy and named-discrepancy ask rate. Move away from Global only if another model is at least 5 points better.
6. **SFT:** TRL with `assistant_only_loss=True`; LoRA r = 16, α = 32 on all 7 projections; `max_length=8192`; micro-batch 1–2; gradient checkpointing.
7. **GRPO:** AgentTune agentic GRPO (or our own exact-token rollout) with programmatic twin rewards; 8 rollouts per prompt at T = 1.0; colocated vLLM at ~0.3 of memory. Mix ask and act twins in every batch. Budget 1–2K episodes per night.
8. **Install AgentTune, AlignTune and AuditKIT with `--no-deps`**, check imports against TRL 1.13, torch 2.13 and vLLM 0.29, and log the result.
9. **Evaluation hygiene:** merge LoRA before using vLLM (#65532), set end tokens explicitly, strip `<|` from tool and user text, and never mix provider and vLLM numbers in a before/after comparison.
10. **Safety:** report XSTest and MultiJail-style checks and a SafeTune audit, because domain SFT eroded refusals for peers.
11. **Publishing:** release the adapters and merged weights under CC-BY-NC-4.0 with attribution and an AUP note **C**.
12. **Feedback for Lexsi:** AgentTune's pins break CUDA-13 aarch64 images, and the fallback format puts tool output in the user role.

### Sources
- S1: [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (also `-base`, `-fire`, `-water`, `-earth`) and the [API](https://huggingface.co/api/models/CohereLabs/tiny-aya-global).
- S2: [arXiv 2603.11510](https://arxiv.org/abs/2603.11510), read via the [HTML](https://arxiv.org/html/2603.11510v1).
- S3: the [launch blog](https://cohere.com/blog/cohere-labs-tiny-aya); Cohere docs: [tiny-aya](https://docs.cohere.com/docs/tiny-aya), [aya-expanse](https://docs.cohere.com/docs/aya-expanse), [tool use](https://docs.cohere.com/docs/tool-use-overview).
- S4: [aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b), [aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b), [arXiv 2412.04261](https://arxiv.org/abs/2412.04261), [the HF blog](https://huggingface.co/blog/aya-expanse).
- S5: hash-checked mirrors: [1-800-LLMs/tiny-aya-global](https://huggingface.co/1-800-LLMs/tiny-aya-global), [adamo1139/aya-expanse-8b-ungated](https://huggingface.co/adamo1139/aya-expanse-8b-ungated), [adamo1139/aya-expanse-32b-ungated](https://huggingface.co/adamo1139/aya-expanse-32b-ungated); the [Command R7B API](https://huggingface.co/api/models/CohereLabs/c4ai-command-r7b-12-2024).
- S6: vLLM [`commandr.py` at v0.29.0](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/commandr.py), [#11203](https://github.com/vllm-project/vllm/pull/11203), [#11583](https://github.com/vllm-project/vllm/pull/11583), [#65532](https://github.com/vllm-project/vllm/pull/65532).
- S7: [TRL v1.13.0](https://github.com/huggingface/trl/tree/v1.13.0/trl): `chat_template_utils.py`, `chat_templates/cohere2*.jinja`, `trainer/sft_trainer.py`, `generation/vllm_generation.py`, `import_utils.py`.
- S8: [PEFT `constants.py`](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/constants.py); [transformers `modeling_cohere2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere2/modeling_cohere2.py) and [`masking_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py); [Liger](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py).
- S9: [AgentTune](https://github.com/Lexsi-Labs/AgentTune) (`rollout_factory.py`, `tests/test_cohere_tool_calls.py`, `pyproject.toml`); [AlignTune](https://github.com/Lexsi-Labs/aligntune) (`core/peft/lora.py`, `core/hf_compat.py`, `pyproject.toml`).
- S10: [StorageReview's DGX Spark review](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops).
- S11: our September run, `AI-Safety-Incident-response/experiments/grpo/run1/trainer_state.json` and `log.md`.
- S12: peer write-ups: [ESA](https://huggingface.co/ashish-soni08/tiny-aya-global-esa-grpo-safe-v4-1), [Oya](https://github.com/Timidan/lexsi-hack), [InjectNo](https://github.com/kenil1710/injectno), [primce](https://huggingface.co/primce/tiny-aya-hinglish-lora), [qasimjethwa](https://huggingface.co/qasimjethwa/aya-expanse-8b-recovery-lora), [jayesh20](https://huggingface.co/jayesh20/lexsi-track3-aya-lora), [banking](https://huggingface.co/ashish-soni08/tiny-aya-fire-banking-safe), [purvansh01](https://huggingface.co/purvansh01/tiny-aya-global-safetypulse-repaired), [AmberNeel](https://huggingface.co/AmberNeel/tiny-aya-global-redlens-guard-GGUF).
