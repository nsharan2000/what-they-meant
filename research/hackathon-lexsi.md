# Lexsi Labs — AI Safety & Alignment Hackathon: rules, tracks, submission (reference)

Sources (fetched 2026-11-12 18:00 IST, raw text saved in `research/sources/`):
- Kickoff slides https://hackathon.lexsi.ai/kickoff-slides/ (37 slides; slide 5 = "Our open-source stack") → `sources/kickoff-slides-text.txt`, poster → `sources/hackathon-poster.png`
- Hackathon site https://hackathon.lexsi.ai/ (Overview / Schedule / Tracks / Structure / Prizes / Rules / Submit / FAQs) → `sources/hackathon-site-text.txt`
- Our previous project's slideshow <link-redacted> → (private extract, not published)
All items below are VERIFIED against those pages unless tagged CAVEATED.

## 1. Key dates (all AoE)
| What | When (AoE) | When (IST) |
|---|---|---|
| Hackathon build window | Nov 1 – Nov 15 | |
| **Submission deadline (round 1)** | **Sun Nov 15 2026, 11:59 PM AoE** (site; the slides said Nov 14 — site is newer, it was extended) | **Mon Nov 16 2026, 17:29 IST** |
| Finalists (≤ 20 teams) announced | Nov 16 | |
| Final submission (video presentation or slides; +$200 HF credits to polish/run extra experiments) | Nov 22, 11:59 PM AoE | Nov 23 17:29 IST |

## 2. Hard rules (Rules page) — every one must be satisfied
1. **Team** 1–3 members; each person in one team only.
2. **Project requirements**: built during the hackathon; "All submissions must chain together repos from their track into a real pipeline". Minimum 2 repos from the track used together (except Track 3); preferred 3+.
3. **Models**: "Use a dense, open-weights LLM from Cohere Labs": Tiny Aya Base / Global / Fire / Water / Earth, Aya Expanse 8B, Aya Expanse 32B. FAQ: "The core submission should use the listed models from Cohere. You can use other models in a supplementary way, such as a teacher model, for validation, or an LLM as a judge."
4. **Hugging Face artifacts**: host all artifacts on HF — models, adapters, logs, datasets.
5. **Submissions**: deployed publicly (public demo or repo link) by the deadline; **before-and-after metrics (more than one metric)**; write-up covering *the metrics chosen and why, the approach and rationale behind the choice of datasets, tasks and models, and how the resulting model is meaningfully better for usage*; an HF Article; an HF Space (if relevant); **[optional but preferred] ROI comparison** of a closed-source model vs the fine-tuned smaller model (self-hosting cost vs closed-source token billing, realistic sample volume, GPU runtime hours, per-hour GPU price, estimated tokens/sec).
6. **Originality**: own work; open-source libraries and public datasets allowed where permitted.
7. Code of conduct; teams keep IP; organisers may showcase.

## 3. What the submit form needs
Registered email · abstract < 500 words · **HF model collection** (all models/adapters, public, un-gated) · **HF dataset collection** (datasets, logs, traces, benchmarks, curated/synthetic data; public) · write-up/article (HF article, or GitHub README if no article access) · **GitHub repo** (README with reproducibility notes; notebooks via the repo) · the repos and models used, track(s), problem statement(s).

Kickoff slide 29 "Submission checklist" — the HF article must have these sections:
1 Problem statement and approach, in brief · 2 Hackathon repos used: which ones, for what, which features · 3 Models used and their licenses · 4 Datasets and benchmarks used and their licenses · 5 Model artifacts public as an HF collection · 6 Data artifacts public as an HF collection · 7 Experiments run, findings and what else was tested · 8 With more credits: your approach, before and after results · 9 Teammates tagged.

## 4. Judging
Labelling (validity: public artifacts, write-up, checklist, no duplicates; tagged to closest track) → double-blind scoring by a judge pool, more than one score per team, normalised averages → 10–20 teams to the final stage.
Scored on: **Write-up clarity /10 · Practical usefulness /10 · Code/notebook quality /10 · Overall review /100** (slide 30).
Hackathon goal (Overview): "chain the stack into a real pipeline and show, with before-and-after metrics, that a small open-weights model becomes meaningfully better, safer or more reliable for a real use case." Also: "gather real-world feedback and feature requests from builders" → our `lexsi-feedback.md` is in scope.

## 5. The stack (slide 5 + site)
| Repo | Purpose | Docs | Code |
|---|---|---|---|
| CuratorKIT | Data curation + synthetic data: ingest (PDF/JSONL/HF) → clean+dedup (exact/MinHash/embedding) → hygiene (secrets/PII/toxicity) → generate (QA, CoT, DPO pairs, GRPO) → quality gates (grounding/reward/diversity) → export (Alpaca/ShareGPT/DPO/GRPO). Paper arXiv 2606.21631 | curatorkit.lexsi.ai/latest/ | github.com/Lexsi-Labs/CuratorKIT |
| AlignTune | Post-training: backends TRL/Unsloth/ES; 13+ RL algorithms "switch with a single flag"; 27+ reward functions (verifiable math/code/SQL); unified SFT/DPO/PPO/GRPO API; multi-stage pipelines; MoA/Text2LoRA/Doc2LoRA adapters; long context; distillation/merging; **alignment audit: reward hacking · sycophancy · refusal collapse**; utility advisor (VRAM/time/cost/carbon); export GGUF/Ollama/Hub. Paper arXiv 2602.09621 | aligntune.lexsi.ai | github.com/Lexsi-Labs/aligntune |
| AgentTune | Agent post-training: backend TRL·vLLM; 5 RL algorithms one entry point; 5 agent strategies one interface; **tool-calling rollouts for every algorithm**; agent lifecycle build→collect→evaluate→train→distill→heal; DECIDE (decision workflows as YAML); EventLog (one trajectory schema); distillation teacher→<3B student; agentic RAG; data synthesis (grounded, difficulty-tagged QA); self-healing loop (detect→classify→retrain→gate); runs on CPU first; "AuditKIT reads the run directly" | agenttune.lexsi.ai | github.com/Lexsi-Labs/AgentTune |
| AuditKIT | Evaluation: dataset (benchmarks/HF/custom) → prompt format → model (API/HF/vLLM) → metrics (accuracy, LLM judge, safety) → results (fingerprint, cached reruns, compare runs) | auditkit.lexsi.ai | github.com/Lexsi-Labs/AuditKIT |
| SafeTune | Measure + repair safety drift after fine-tuning: Harden (train-time), Recover (weight-space), Unlearn, Steer (inference-time) | safetune.lexsi.ai | github.com/Lexsi-Labs/SafeTune |
| CircuitKIT | Mechanistic interpretability: 13 circuit-discovery algorithms, 6 faithfulness pillars, prune/edit/fine-tune/quantize/steer | circuitkit.lexsi.ai | github.com/Lexsi-Labs/CircuitKIT |

## 6. Tracks and the problem statements that fit "What They Meant"
- **Track 3 — Agentic Fine-tuning (AgentTune + AuditKIT)** ← primary fit: "Business workflow and decision agents: automate a multi-step process with rules and approvals … with a full audit trail"; "Reliable tool calling"; "Reward design and reward-hacking defences"; "Agentic distillation".
- **Track 2a — AI Safety & Red Teaming** ← secondary fit: "Adversarial and safety evaluation sets: build a safety benchmark for a specific domain … with clear labels and scoring, and use it to measure how current open models hold up"; "Multi-turn audits".
- Track 1 — "Domain assistants for regulated industries (banking …)"; "Data-scarce domains: generate, verify and filter synthetic data, show curation beats raw generated data" (CuratorKIT angle).
Track choice is recorded as a decision in `log.md` once research finishes. Using CuratorKIT + AgentTune + AlignTune + AuditKIT (+ SafeTune audit) exceeds the "3 or more repos" preference.

## 7. Compute / credits facts
- Registered teams that join Discord get $20 HF credits (ZeroGPU or other GPU instances; first 800 teams; expire 30 days after redemption). Colab/Kaggle free GPUs "can be enough for the Tiny Aya models". We use the DGX Spark (see CLAUDE.md) + HF Inference Providers.
- VERIFIED by API (2026-11-12): HF Inference Providers serve `CohereLabs/tiny-aya-{global,fire,water,earth}` and `CohereLabs/aya-expanse-32b` via provider `cohere` (status live). `aya-expanse-8b` and `tiny-aya-base` have no provider → serve on the Spark.
- VERIFIED: **no Aya model exposes native tool calling** through the provider: Tiny Aya Global ignored the `tools` field (64 prompt tokens, replied "I can't directly send money"); Aya Expanse 32B returned HTTP 400 `TOOL_USE_NOT_SUPPORTED`. → Our environments use a text tool-call protocol that every model (Aya, other open models, Claude) sees identically.
- All Aya repos are gated ("auto" approval) and CC-BY-NC-4.0; our HF token currently gets 403 on file downloads → the user must click "Agree" on each model page (questions-for-user.md Q1).
- Other teams already publish Tiny-Aya-Global work on HF (unlearning, safety repair, guard models, clinical triage); none visible on agentic payments.
