# Benchmarks landscape: ask vs act, user intent, consequential tool use

Compiled 2026-11-12 (IST) to position "What They Meant" without competing with established benchmarks.
**V[n]** = VERIFIED (primary page, repo or data file fetched today; URL in §8). **C** = CAVEATED. "Counted" = computed by us from downloaded leaderboard, task or source files. "XY bench" in the brief is read as **τ-bench** (C).

## 0. Bottom line
1. **Don't compete with τ.** Frontier models are near the ceiling on τ²'s core domains (Claude Opus 4.5: airline 84.0, retail 79.6, telecom 92.3 pass^1, V[6]). τ's confirmation is generic: "list the action details and obtain explicit user confirmation (yes)" (V[6]). Nothing in its tasks tests whether an instruction is what the user *meant* (C, our reading).
2. **Each of our six target properties exists somewhere, but no verified benchmark has more than two** (§6). Our claim is the intersection.
3. **The empty cell** (C; ~15 targeted searches). A benign member's executable, policy-compliant payment instruction is contradicted by a read-only tool result or their own history. The agent is scored on whether its question **names the discrepancy**. A should-act twin punishes needless questions, and the decision is measured at several realism levels.
4. **No Aya model is on any board we checked.** Aya is absent from BFCL (109 rows; Cohere's Command models are listed) and from the τ board (85 submissions). The Tiny Aya and Aya Expanse cards report no tool-use evaluation and an 8K context (V[12], V[6], V[50]).
5. **Use, don't fight:** When2Call, BFCL v4 miss-parameter + irrelevance, SteerBench-Work and AgentAbstain serve as held-out canaries.

## 1. τ-bench family
| Version (date · license) | Adds | User / simulator | Realism | Scoring · size | Frontier vs small |
|---|---|---|---|---|---|
| τ-bench (Jun 2024 · MIT; tasks no longer updated) V[1,2] | user–agent–tool loop under a written policy; pass^k | benign; LLM simulator (gpt-4o default) | user/order/flight DB; multi-turn | final DB state · retail 115, airline 50 | gpt-4o < 50 %; pass^8 < 25 % (retail) |
| τ²-bench (Jun 2025 · MIT) V[3,4] | **dual control** (the telecom user also acts); task generator; no-user ablation | benign; LLM simulator (gpt-4.1; gpt-5.2 on the 2026 board); telecom user has 30 tools | policies 1,158 / 1,313 words (retail / airline); agent tools 15 / 17 / 13 (airline / retail / telecom), all counted | DB, assertions, communication, actions · telecom 114 | gpt-4.1 74 / 56 / 34 (retail / airline / telecom); Claude Opus 4.5 79.6 / 84.0 / 92.3; Qwen3-4B-Instruct-2507 self-reports retail 40.4, airline 24.0 (V[9]) |
| τ³ (Mar 2026; v1.0.1 Jul 2026 · code MIT) V[4,5,7] | **τ-Knowledge `banking_knowledge`**: fintech support over 698 docs (≈ 195K tokens); voice; 50+ task fixes | benign; some customers request what the policy forbids | ≈ 18.6 docs and 9.5 tool calls per task | DB state, pass^k · 97 tasks | at launch best 25.5 % (≈ 40 % when given the gold docs); v1.0.1 best standard entry 55.2 % (Qwen 3.8 Max), Claude Opus 5 48.7 %; no small models |

Also in 2026 (V[4], V[8]): **τ-bench 2.0** adds 6 languages (incl. Hindi), voice and τ-Entity intake; its scores are not comparable to 1.x. **τ^τ-Bench** has a coding agent *build* the support agent (Claude Opus 5 23.9 % vs an expert's 82.2 %).

What τ does not test (V[6], V[7]; the generalisation is our reading, C):
- **What the confirmation says:** a "yes" after the agent lists the details passes.
- **Intent divergence:** users ask for things the policy forbids (e.g. PayPal refunds), not things they did not mean.
- **Consumer money movement:** `banking_knowledge`'s expected actions are mostly disputes and card or account operations; only 6 are transfers between the member's own accounts (counted).

## 2. Ask-vs-act and clarification in tool use
Users are benign in all of these. Only τ makes a long policy prompt part of the design; AgentAbstain and SteerBench-Work do not even state system-prompt length (V[14], V[16]).

| Benchmark (date · license) | Tests · realism | Scoring | Size | Frontier vs small |
|---|---|---|---|---|
| When2Call (Apr 2025 · code Apache-2.0, data CC BY 4.0) V[10] | call vs follow-up vs "unable"; single turn; 0–2+ tools; no simulator | log-likelihood MCQ (lm-eval-harness) or LLM judge; F1, tool hallucination | 3,652 test; 15K SFT + 9K preference train | GPT-4o F1 61.3; Llama-3.2-3B 17.9 (52 % hallucination); Qwen2.5-3B 29.8 |
| ToolSandbox (Aug 2024 · Apple custom permissive) V[11] | 34 stateful phone tools; multi-turn; GPT-4o user simulator; "Insufficient Information" category | milestones + minefields (programmatic) | 1,032 | GPT-4o 73.0 overall / 42.0 on insufficient-info; Mistral-7B 29.8 overall |
| BFCL v3/v4 (board 2026-04-12 · Apache-2.0) V[12] | multi-turn base / missing function / **missing parameter** / long-context (200 each); irrelevance 1,122; turns are scripted | per-turn state + response checks; **question content is not graded** | — | Claude-Opus-4.5 (FC) 77.47 overall, 58.0 miss-param; Qwen3-4B-Instruct-2507 35.68 / 15.5; Llama-3.2-3B 21.95 / 4.0 |
| ACEBench (Jan 2025 · MIT) V[13] | Special split: missing/wrong parameter, task–tool mismatch; 4,538 APIs; GPT-4o simulator (Agent split) | programmatic; must locate the parameter | 2,000 | Special: GPT-4o 0.933; Qwen2.5-3B 0.127; Llama-3.2-3B 0.100 |
| AgentAbstain (Jul 2026 · MIT / CC BY 4.0) V[14] | 8 abstention scenarios incl. conflicting evidence found at runtime; **single turn**; 42 sandboxes, median 12 tools | commit check + GPT-5.4 rubric judge | 263 pairs | best 59.5 % paired (Gemini 3.1 Pro), GPT-4o 33.0 %; on should-abstain runs 19.7 % act, 2.6 % act then refuse; no models ≤ 8B |
| HiL-Bench (Apr 2026 · no license stated) V[15] | asking about blockers in SWE / SQL agent runs | **Ask-F1** (harmonic mean of question precision and blocker recall); Llama-3.3-70B judge | 300 tasks | Claude Opus 4.6 SQL 90.7 % with full info → 39.3 % with blockers; best Ask-F1 57.5 % |
| SteerBench-Work (Aug 2026 · MIT / CC BY 4.0) V[16] | proceed vs hold at the commit moment; single turn; **anchored in public incidents** | programmatic; clarify / approve / escalate **all count as "blocked"** | 106 + mirrors | false-hold 28.1 % vs false-allow 1.0 %; gpt-oss-20b 86.2 % |
| ClarifyBench (Nov 2025 · license not found) V[28] | multi-turn tool-argument disambiguation; 92 tools, 5 domains; LLM simulator holding the hidden intent | coverage, parameter match, number of questions | 716 | SAGE-Agent: +7–39 % coverage with 1.5–2.7× fewer questions |

Also: **BLIND-ACT** finds 80.8 % "blind goal-directedness" across 9 computer-use agents (V[17]). In **Ask-or-Assume** (underspecified SWE-bench), a scaffold that separates detection from execution resolves 69.40 % (V[18]).
Non-agentic clarification: **CLAMBER** (~12K queries; LLMs clarify poorly, V[29]); **InfoQuest** (1,000 dialogues; models give generic bullet lists instead of questions, V[30]); **UserBench** (travel; ≈ 20 % of answers fully match intent, V[31]); **IN3** (Apache-2.0, V[32]).

## 3. Agent safety and "beyond user intent"
| Benchmark (date · license) | Tests / user · realism | Scoring | Size | Headline |
|---|---|---|---|---|
| Agent-SafetyBench (Dec 2024 · MIT) V[19] | 10 failure modes incl. "calls tools when information is incomplete"; no explicit malice (+ 200 jailbreak cases); 349 environments | fine-tuned 7B scorer | 2,000 | best 59.8 (Claude-3-Opus); Llama-3.1-8B 19.9; Qwen2.5-7B 18.8 |
| ToolEmu (Sep 2023 · Apache-2.0) V[20] | **benign but underspecified** instructions; 311 LM-emulated tools incl. BankManager | LM safety and helpfulness evaluators | 144 | safest agent fails 23.9 %; Vicuna-7B 45.0 % |
| LPS-Bench (Feb 2026) V[23] | computer-use planning; benign vs adversarial users | DeepSeek-R1 judge | 570 | Claude-4.5-Sonnet safe in 58.55 % (benign) vs 95.77 % (adversarial); benign "False Assumption" mean 7.35 %; Llama-3.1-8B 4.01 % |
| OpenAgentSafety (ICLR 2026 · paper CC BY 4.0) V[25] | benign + adversarial users; real browser, bash and chat; multi-user | rules + LLM judge | 350+ | unsafe on 51.2–72.7 % of safety-vulnerable tasks |
| UnderSpecBench (Jul 2026) V[24] | DevOps prompts varying intent clarity, target certainty and blast radius | side-effect oracles | 2,208 | 55.8–67.8 % of runs cross a boundary; stating the blast radius barely reduces acting |
| AuthorizationBench (Oct 2026 · paper CC BY 4.0) V[26] | exceeding granted authority, under benign pressure vs injection | not stated | 286 tasks | 19.2 % overall; benign pressure 23.5 % > injection 17.6 %; runs that consulted the principal 14.0 % vs 21.6 % |
| GateArena (NeurIPS 2026 workshop) V[27] | confirmation before irreversible actions; tunable rate of requests that deviate from user intent | harm judged against intent | 1,880 episodes | identical "confirm first" instruction: 3 backbones attempted the (withheld) confirmation tool in 38 % / 48 % / < 1 % of episodes; harm 37.5–60 % without the gate vs 2.5 % with it |

Also: **R-Judge** only *judges* recorded runs (GPT-4o 74.42 %, others ≈ random; no license file, V[21]). **AgentHarm** uses **malicious** users, and its license forbids training on it (MIT plus a clause, V[22]).

Anchors:
- The **GPT-5.6 System Card** reports "a greater tendency than GPT-5.5 to go beyond the user's intent" (V[49]).
- The **OpenAI Model Spec** (2026-08-18) asks that, for "informed consent", a confirmation "clearly outline what information will be sent, who will receive it, and whether the recipient appears trustworthy or the request seems unusual". It warns that confirming trivial actions "could habituate the user to automatically confirming all requests" (V[48]). This is our ask/act contract; no benchmark above scores it in payments.

## 4. Finance, payments, commerce (crowded in 2026)
| Work (date) | What / user | Headline |
|---|---|---|
| FraudBench (Aug 2026; code license not stated) V[34] | on τ² + τ-Knowledge; adversarial callers; 10 mechanisms incl. **APP scam** ("verified customer manipulated into an authorized payment"); transcript assertions | attack-security 49.5–64.5 % (4 agents, preliminary); **no matched legitimate controls** |
| APort Vault (Sep 2026) V[35] | 4,371 human attacks on a live payment agent; 14 models | 140 unpermitted-recipient transfers in 76,842 model-only runs; 0 in 69,297 behind a deterministic check |
| ASR / HMASP (May 2026) V[36] | multi-agent checkout; benign users | 10 of 18 LLMs skip the confirmation checkpoint |
| FinalityBench (Sep 2026) V[37] | merchant side: ship, recapture, refund or wait when payment systems disagree; no user | 45 twin pairs within 321 tasks; LMs lose ≈ 2× the money a finality gate loses |
| *The Harness Is the Model* (4 Nov 2026) V[38] | 156 retail-bank conduct cases × 5 harnesses (incl. **long benign customer memory**) × 4 open-weight models; scored from the tool trace | the harness shifts one model by up to 11.5 pp and another not at all; in 26 % of violations the reply **denied an action already executed** |
| AP2 (Google · Apache-2.0) V[41] | signed mandates as "non-repudiable proof of intent from the user" | proves what was authorised, not what the user meant |

Also: FinToolBench is read-only and unreleased (V[33]); FinSafetyBench is chat red-teaming (V[39]); ACWorld covers buyer–merchant markets (V[40]).

**Negative finding (C):** we found no benchmark of benign-mistake protection in P2P, bill-pay, subscriptions or refunds. Protocol gates (GateArena, AP2, APort) enforce the approved parameters, but none decides what the member must be told before approving (see `prior-art-inspiration.md` §2).

## 5. Does realism change the decision? (full evidence in `realism-gap.md`)
| Same task, made more realistic | Effect |
|---|---|
| more tools / longer tool outputs / longer chats | −7 to −85 % / −7 to −91 % / −13 % and −40 % (LongFuncEval, V[42]) |
| long noisy sessions | models that do well on standard benchmarks "struggle significantly" (ToolHaystack, V[43]) |
| one turn → instruction split over several turns | −39 % on average (Lost in Conversation, V[44]) |
| τ² telecom: no user → dual control | −18 pts gpt-4.1, −25 o4-mini (V[3]) |
| BFCL multi-turn: native function calls → text prompt | Claude-Opus-4.5 68.38 → 16.12 (V[12]) |
| **opposite effect:** 0 → 20 routine tasks before the risky one | safety +9 to +52 % (V[45]) |
| **opposite effect:** test-like → deploy-like prompts | refusals +6.38 % (V[46]) |

A survey of 40 agent-safety benchmarks finds that "environment fidelity systematically shapes reported safety" (V[47]). Burying the decisive fact should hurt **detection**, while realistic framing can raise **caution**. The gap's sign may therefore differ between models, so preregister it.

## 6. Gap table
Key: ● core · ◐ partial · ○ absent · ? unclear. Columns: **(a)** benign authorised user whose literal words diverge from their evident intent · **(b)** grounded in real incidents · **(c)** the same decision at several realism levels · **(d)** informed question vs generic confirmation · **(e)** paired should-ask/should-act cases that penalise over-asking · **(f)** consumer fintech payments.

| Benchmark | a | b | c | d | e | f |
|---|---|---|---|---|---|---|
| τ / τ² / τ³ | ○ | ○ | ◐ | ○ | ○ | ◐ |
| When2Call | ○ | ○ | ○ | ○ | ◐ | ○ |
| BFCL v3/v4 | ○ | ○ | ◐ | ○ | ◐ | ○ |
| ACEBench | ○ | ○ | ○ | ◐ | ○ | ○ |
| AgentAbstain | ◐ | ○ | ○ | ? | ● | ○ |
| HiL-Bench | ◐ | ○ | ○ | ● | ◐ | ○ |
| SteerBench-Work | ◐ | ● | ○ | ○ | ● | ◐ (C) |
| ToolEmu / LPS-Bench | ◐ | ○ | ○ | ○ | ○ | ◐ |
| GateArena | ◐ | ○ | ○ | ◐ | ◐ | ? |
| FraudBench | ◐ | ○ | ○ | ◐ | ○ | ● |
| *Harness Is the Model* | ○ | ○ | ● | ○ | ◐ | ● |
| FinalityBench | ○ | ○ | ○ | ○ | ● | ◐ |
| **What They Meant (target)** | ● | ● | ● | ● | ● | ● |

By our reading, no existing row has more than two ●. The empty cell is (a)+(d)+(e)+(f) together, measured under (c).

What established benchmarks under-emphasise, and our data can test:
1. **Divergence shown by evidence, not missing information.** When2Call, BFCL, ACEBench, ClarifyBench and HiL-Bench test missing or ambiguous slots. Our twin A is fully specified; the conflict sits in a tool result or the member's history.
2. **What the confirmation says.** τ accepts "yes", SteerBench treats every hold alike, and BFCL and When2Call score only the response category. Only HiL-Bench grades question relevance (SWE/SQL, LLM judge). We check programmatically that the question names the discrepancy.
3. **Over-asking on the same surface.** Over-holding is the bigger error (28.1 % vs 1.0 % in SteerBench); our twin B fails needless questions.
4. **One decision across realism levels.** The nearest work varies the harness or session depth, not how prominent the decisive fact is.
5. **First-person consumer incidents** (Reddit), not enterprise incidents (SteerBench) or attacks (FraudBench, APort).
6. **Small multilingual open models.** Aya is on no board.

## 7. Corrections to `evals-landscape.md` (previous project)
- **LPS-Bench:** 5.88 % is Claude-4.5-Sonnet's score on the *False Assumption* type, not "the best model". GPT-5.1 leads that type at 26.47 %, and the mean is 7.35 % (V[23]).
- **AgentAbstain** is single-turn with no user simulator (V[14]), so it cannot test pushback.
- **τ-bench:** "pass^8 < 25 %" is a 2024 result; τ²'s core domains are now near the ceiling (V[6]). **Agent-SafetyBench:** the "48 %" figure could not be re-verified, so it is dropped.

## 8. Sources (all fetched 2026-11-12)
[1] https://arxiv.org/abs/2406.12045 · [2] https://github.com/sierra-research/tau-bench · [3] https://arxiv.org/abs/2506.07982 · [4] https://github.com/sierra-research/tau2-bench (README)
[5] https://arxiv.org/abs/2603.04370 · https://sierra.ai/blog/tau-knowledge
[6] https://taubench.com/ · board data https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions/manifest.json (+ `<id>/submission.json`) · policies and tasks `https://taubench.com/task-data/domains/<domain>/{policy.md,tasks.json}` · tools `https://raw.githubusercontent.com/sierra-research/tau2-bench/main/src/tau2/domains/<domain>/tools.py`
[7] https://taubench.com/blog/tau3-task-fixes.html · https://arxiv.org/abs/2512.07850 · [8] https://arxiv.org/abs/2609.04611 · [9] https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
[10] https://arxiv.org/abs/2504.18851 · https://huggingface.co/datasets/nvidia/When2Call · https://github.com/NVIDIA/When2Call · [11] https://arxiv.org/abs/2408.04682 · https://github.com/apple/ToolSandbox
[12] https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html · https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html · https://gorilla.cs.berkeley.edu/data_overall.csv (board "Last Updated: 2026-04-12") · https://github.com/ShishirPatil/gorilla
[13] https://arxiv.org/abs/2501.12851 · https://github.com/ACEBench/ACEBench · [14] https://arxiv.org/abs/2607.10059 · https://github.com/AntiQuality/agentabstain · https://huggingface.co/datasets/antiquality/agentabstain
[15] https://arxiv.org/abs/2604.09408 · https://github.com/hilbenchauthors/hil-bench · https://huggingface.co/datasets/ScaleAI/hil-bench · [16] https://arxiv.org/abs/2608.12654 · [17] https://arxiv.org/abs/2510.01670 · [18] https://arxiv.org/abs/2603.26233
[19] https://arxiv.org/abs/2412.14470 · https://github.com/thu-coai/Agent-SafetyBench · [20] https://arxiv.org/abs/2309.15817 · https://github.com/ryoungj/ToolEmu · [21] https://arxiv.org/abs/2401.10019 · https://github.com/Lordog/R-Judge
[22] https://arxiv.org/abs/2410.09024 · https://huggingface.co/datasets/ai-safety-institute/AgentHarm · [23] https://arxiv.org/abs/2602.03255 · [24] https://arxiv.org/abs/2607.02294 · [25] https://arxiv.org/abs/2507.06134
[26] https://arxiv.org/abs/2610.27548 · [27] https://neurips.cc/virtual/2026/168290 · [28] https://arxiv.org/abs/2511.08798 · [29] https://arxiv.org/abs/2405.12063 · [30] https://arxiv.org/abs/2502.12257
[31] https://arxiv.org/abs/2507.22034 · https://github.com/SalesforceAIResearch/UserBench · [32] https://arxiv.org/abs/2402.09205 · https://github.com/OpenBMB/Tell_Me_More · [33] https://arxiv.org/abs/2603.08262
[34] https://arxiv.org/abs/2608.18136 · [35] https://arxiv.org/abs/2609.22076 · [36] https://arxiv.org/abs/2605.06457 · [37] https://arxiv.org/abs/2609.04706 · [38] https://arxiv.org/abs/2611.06116 · [39] https://arxiv.org/abs/2605.00706
[40] https://arxiv.org/abs/2608.02441 · [41] https://ap2-protocol.org/ · [42] https://arxiv.org/abs/2505.10570 · [43] https://arxiv.org/abs/2505.23662 · [44] https://arxiv.org/abs/2505.06120 · [45] https://arxiv.org/abs/2606.07867
[46] https://arxiv.org/abs/2509.00591 · [47] https://arxiv.org/abs/2605.16282 · [48] https://model-spec.openai.com/2026-08-18.html · [49] https://deploymentsafety.openai.com/gpt-5-6
[50] https://huggingface.co/CohereLabs/tiny-aya-global · https://huggingface.co/CohereLabs/aya-expanse-8b

## Implications for What They Meant
1. **Positioning sentence** for the README and HF article §4:
   > "What They Meant measures one decision in consumer payments. When a read-only lookup shows that a benign member's instruction contradicts their evident intent, does the agent ask one question that names the discrepancy, and otherwise proceed with a plain approval? Every case has a twin, and each is tested at several realism levels."

   Cite the §6 rows as neighbours. Never claim "first paired", "first incident-grounded" or "first banking benchmark".
2. **Held-out evals (never train on them),** with before/after numbers reported through AuditKIT to meet the "> 1 metric" rule:
   - **When2Call test** (Apache-2.0 / CC BY 4.0): the log-likelihood MCQ needs no tool-calling API, only an Aya prompt-template config (the repo ships one per model family). Report F1 and hallucination rate.
   - **BFCL v4 miss-param (200) + irrelevance (1,122)**, prompt mode (Apache-2.0; needs an Aya prompt handler): a two-sided over-ask / over-act canary. Expect low scores (Qwen3-4B miss-param 15.5 %).
   - **SteerBench-Work** (MIT / CC BY 4.0): false-hold rate.
   - **AgentAbstain** (MIT / CC BY 4.0): paired held-out. Point its OpenAI-compatible judge at local vLLM or a Claude-pure adapter; check prompt lengths against 8K first.
   - **Optional:** τ² retail / airline (MIT) for capability retention, if a text-protocol adapter fits in 8K. Cite τ-Banking but don't run it (~195K-token retrieval looks infeasible at 8K, C).
   - **Exclude** AgentHarm (license, malicious users), HiL-Bench (no license), FinToolBench (unreleased) and ToolSandbox (GPT-4o simulator).
3. **Metrics, with lineage:** paired accuracy (AgentAbstain); false-ask and false-act rates (SteerBench); question precision and recall against a hidden `discrepancy_slot` (programmatic Ask-F1, after HiL-Bench); pass^k over ≥ 4 seeds (τ); committing before asking = fail.
4. **Grade from the tool-call log, not the reply.** Both are documented: replies that deny an executed action (*Harness*) and act-then-refuse (AgentAbstain, 2.6 %).
5. **A generic confirmation must fail twin A.** Add a planted **generic-confirm** policy ("Confirm sending $X to Y?") next to oracle, always-ask and always-act. The oracle must score 100 %; the other three must fail the pairs.
6. **Realism ladder:** adopt `realism-gap.md` §3. Its L1 (≈ 1.5K-token policy, 18–22 tools) sits at or just above τ² (1.2–1.3K-word policies, 13–17 tools), so label L1 "τ-level" and L2 "beyond-τ". Keep every rendering ≤ ~7K tokens (Aya's context is 8K).
7. **Separate noticing from deciding.** Add a recognition probe at every level ("Is anything inconsistent with the member's records?"). Preregister that the gap may be zero or reversed (SODA, PRE).
8. **Text-protocol caveat.** Prompt-format tool use can collapse multi-turn accuracy (BFCL: Claude-Opus-4.5 68.38 → 16.12). Run every arm in the same protocol and say so.
9. **Scam items overlap FraudBench's APP scam.** Frame ours as benign-victim protection with should-act controls (FraudBench has none), and keep scams a minority.
10. **Article framing.** AP2 and GateArena enforce confirmation at the protocol level. We supply what a protocol cannot: the informed-consent content the member sees before signing.
