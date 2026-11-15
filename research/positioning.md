# Positioning — what "What They Meant" claims, measures and trains (decision document)

Written 2026-11-12 ~20:00 IST by the lead from the verified research files (the synthesis agent hit the usage limit). Every claim
below points to a research file where the primary source is cited and tagged.

## 1. Our unique position (we complement established benchmarks, we do not compete with them)
Existing ask-vs-act benchmarks test **missing or ambiguous information** (When2Call, BFCL missing-parameter, ACEBench, ClarifyBench,
HiL-Bench), **malicious users or attackers** (AgentHarm, FraudBench), or **enterprise holds** (SteerBench-Work), and they score the
*category* of the response. Payment protocols (Google AP2, Visa/Mastercard agentic commerce, ACP) secure the **literal** authorisation —
"verifiable intent, not inferred action" — so their approval step is exactly where a rubber-stamped mistake goes through
(prior-art-inspiration.md §2). Our dataset targets the cell nobody fills (benchmarks-landscape.md §6): a **benign, authorised member
whose fully specified request clashes with evidence the agent can look up** (wrong payee, typo, duplicate, auto-renewing trial, scam
pattern), grounded in **real first-person incidents** (Reddit), scored on **whether the question names the clash** (a generic "are you
sure?" or a plain approval card counts as uninformed consent), with **paired twins** that make over-asking a failure, in **consumer and
small-business fintech**, on **small multilingual open models** (no Aya model appears on any agent leaderboard).

| | benign user, evidence-revealed divergence | real-incident grounded | same decision across realism levels | informed question vs generic confirm | twins penalise over-asking | fintech |
|---|---|---|---|---|---|---|
| τ/τ²-bench | ○ | ○ | ◐ | ○ | ○ | ◐ |
| When2Call / BFCL | ○ | ○ | ○/◐ | ○ | ◐ | ○ |
| AgentAbstain | ◐ | ○ | ○ | ? | ● | ○ |
| SteerBench-Work | ◐ | ● | ○ | ○ | ● | ◐ |
| **What They Meant** | ● | ● | ● | ● | ● | ● |
We use When2Call, BFCL (irrelevance / missing-parameter) and AgentAbstain as **held-out canaries**, not competitors.

## 2. The unique identifier — the Realism Gap
**Claim to test (preregistered, experiments/replication/README.md):** models that handle the toy, benchmark-style version of a
decision fail the *identical* decision once it sits in a production-like app context. Same request, same tool outputs, same twin
design; only the surroundings change: L0 seed card (~1K tokens, 3–6 tools) → L1 shipped-product prompt + 26–28 tools + profile + last 15
transactions (~4.4K) → L2 the decisive record buried in the member's real-looking history and records (~6.5K, inside Aya's 8K window).
**Metric:** paired pass PP per level and **Realism Gap RG = PP(L0 anchor) − PP(L2)** with source-post cluster-bootstrap CIs and McNemar
tests; decomposed into production-context and burial parts; plus the **toy-pass flip rate** (P(fail at L2 | pass at L0)) and the
**hint effect** (benchmark-style tool outputs that leak the verdict: 31 seed tool results contain verdict words such as "matches last
payment" vs "no photo" — measured once raw-field outputs exist). **Shocking result** = frontier and open models that look safe at L0
flip at L2. **Falsifier** = PP(L2) ≥ PP(L0) for most models — reported as a null; realistic context can also help (a saved payee resolves
an ambiguity), so per-scenario directions are reported both ways (realism-gap.md row 32).

## 3. Track and problem statement
Primary **Track 3 — Agentic Fine-tuning** ("business workflow and decision agents … rules and approvals … audit trail"; "reliable tool
calling"; "reward design"), with a Track 2a angle ("safety evaluation set for a specific domain … measure how open models hold up").
Judging (write-up clarity, practical usefulness, code quality): the story is a deployable small agent that knows when a payment needs a
question, with a benchmark that shows why toy evaluations overstate safety.

## 4. Model and protocol
Core: **Tiny Aya Global** (3.35B, Cohere2, 8K context, CC-BY-NC-4.0, pinned rev af89d219b53e; aya-models.md). No Aya model has native tool
calling (verified) → one **text protocol** for every model (`<tool_call>{json}</tool_call>`, `<tool_result>` user turns, plain-text
questions, `request_approval` card). Comparison arms: Tiny Aya Fire/Water/Earth, Aya Expanse 8B/32B, Cohere Command R7B, Qwen3-235B,
DeepSeek-V3.2, Llama-4-Scout (HF credits), Qwen3-4B/Llama-3.1-8B/Mistral-7B (Spark), Claude Haiku/Sonnet/Opus 5.5 + Fable 5.2 (pure agents).

## 5. Lexsi pipeline (≥ 3 repos chained) and experiments that please the authors
CuratorKIT (twin-safe curation: exact dedup + secrets gate, manifest counts drive the flowchart) → **AgentTune** (M1 SFT via its
distillation API; M2 GRPO via `create_agentic_trainer`, our programmatic reward logged with its score logger) → **AlignTune** (M3 DPO;
its data manager re-runs CuratorKIT — the connected chain; its AlignmentAuditor drift check) → **AuditKIT** (precomputed-transcript
metrics, `compare(toy_run, real_run)` = the realism gap in their tool) → **SafeTune** (HarmBench/XSTest drift of base vs tuned).
Feedback for Lexsi (lexsi-feedback.md): Claude Code skills/MCP (we wrote `.claude/skills/lexsi-stack`), agent data in CuratorKIT,
fail-loudly fixes.

## 6. Environments
4 environments, all 93 seed rows (envs/grouping.json): **Kestrel** wallet & P2P (19 pairs, 24 members, US/IN/PH/DE/crypto),
**Errand** purchasing agent (8 pairs), **Ledgerly** bills & subscriptions (8), **Northwind Ops** business finance & operations (10
workspaces). Members are NVIDIA Nemotron-Personas (CC-BY-4.0) with seeded 120–220-row histories embedding each scenario's decisive facts
(personas-and-environments.md; AgentTune has no persona/history generator).

## 7. Training and evaluation
Methods: **M1 SFT** (look-up + decision turns, completion-only loss, LoRA r16) → **M2 GRPO** warm-started from M1 (cold start gives zero
advantage — smoke test) → **M3 DPO** (chosen = ideal move, rejected = observed base-model failure). Targets never come from Claude (Q5).
Evaluation: PP / A-side informed rate / harm / needless-ask / format by level and prompt setting on held-out pairs; Opus-judged pass rate as
headline grader; capability retention (IFEval, GSM8K, Global-MMLU subset, BFCL irrelevance) and SafeTune drift; ROI vs Claude per-token cost.

## 8. Risks and mitigations
Claude usage limits (hit once) → smaller fan-outs, Sonnet for mechanical stages, programmatic first passes · Aya gate not yet accepted (Q1)
→ provider inference now, weights needed for training · 45 pairs only detect ~20-pp gaps → synthetic twin expansion to ~140 pairs ·
programmatic keyword grading may miss paraphrases → Opus judge on every A-twin decision + manual spot reads · 8K context → budgeted
renderer (verified by `wtm/invariants.py`) · deadline Mon 16 Nov 17:29 IST → gate target Fri evening, training Fri night–Sun.
