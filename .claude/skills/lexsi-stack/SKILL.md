---
name: lexsi-stack
description: How to use the Lexsi Labs open-source stack (CuratorKIT, AgentTune, AlignTune, AuditKIT, SafeTune) correctly in this repo — install on the DGX Spark without breaking torch/TRL, which entry points work, which defaults silently corrupt twin agent data, and how our text tool protocol, rewards and audits plug in. Load before writing any curation, training or evaluation code that touches a Lexsi library.
---

# Lexsi stack — working notes for "What They Meant"

Lexsi publishes no Claude Code skill, MCP server or plugin (VERIFIED 2026-11-12 across all 17 Lexsi-Labs repos; only
SafeTune has an `llms.txt`). This skill is our substitute, distilled from `research/lexsi-curatorkit-auditkit.md` and
`research/lexsi-agenttune-aligntune-safetune.md` — read those for citations and line numbers.

## Pinned versions (repos change daily during the hackathon)
AgentTune `1dfd054` · aligntune `b6e924e` · CuratorKIT `ed45ae3` · AuditKIT `556eb48` · SafeTune `cc3de5f` (clones in `third_party/`,
also at `/workspace/third_party` on the Spark). Licences: AgentTune/CuratorKIT LSAL v1.1, AuditKIT/SafeTune LSAL v1.2 (non-commercial),
AlignTune MIT (vendors AGPL unsloth parts).

## Install (inside `lx-exp`)
- `bash /workspace/experiments/infra/install_lexsi.sh` → `pip install --no-deps -e` for each repo. A plain install downgrades torch 2.13→2.11
  and TRL 1.13→1.7.1. Never install `auditkit[vllm]` or `auditkit[all]` (forces vllm>=0.30 / torch 2.14).
- Always `export PURE_TRL_MODE=1` and pass `backend="trl"`: `backend="auto"` picks Unsloth whenever importable, and AlignTune's wheel makes it importable.
- Run each library in its own process (importing aligntune patches `trl.import_utils.is_vllm_available` to False).

## Shared protocol (every model, every library)
Roles system/user/assistant only (Aya templates have no tool role). Tool call = assistant text `<tool_call>{"name","arguments"}</tool_call>`;
tool result = user turn `<tool_result name="…">…</tool_result>`; approval card = `request_approval` call; asking = plain text.
Implementation: `wtm/protocol.py`, `wtm/render.py`, rewards `wtm/reward.py`. Tiny Aya tokenizer counts are ~25 % above our 4-chars/token
estimate: L2 system+request ≈ 7.1K tokens (max 7.8K), decision rows up to ~8.5K → serve vLLM with `--max-model-len 16384` and train
with max length 12,288 (an 8K window overflowed at L2 — log 2026-11-12 22:1x).

## CuratorKIT (data curation) — Python API only
- No tool-call/trajectory support: exporters keep role/content. Feed it decision-point views (prompt + target turn), not raw trajectories.
- Settings that keep twins intact: `strip_html=False`, `max_tokens=32768`, `dedup="exact"`, `secrets_gate=True`.
- NEVER use MinHash/embedding dedup, diversity filters (they merge twin A/B), PII pseudonymisation (rewrites near-miss names), or
  `output_split` (random per-sample split leaks twins across splits). Split by `source.post_id` ourselves.
- Useful gates: `HallucinationGate` (twin-A question grounded in the tool result), `RewardGate` (scores DPO pairs; contrast filter).
  Judges must be an open model through the HF router (training-data provenance policy, questions-for-user Q5), not Claude.
- Keep `manifest.json` stage counts + `rejected.jsonl` reasons: they are the numbers in the curation flowchart.

## AgentTune (agent training) — M1 SFT/distillation, M2 GRPO
- No plain SFT entry point: use `create_distill_trainer(student, trajectories=…, trainer_factory=…)` with a custom factory that builds
  prompt/completion pairs (completion-only loss; the built-in factory trains on every token and drops the system role).
- GRPO: `create_agentic_trainer("grpo", backend="trl", reward_funcs=[…], use_vllm=True, vllm_mode="colocate")` WITHOUT `tools`
  (batched TRL path) on decision/lookup turns. The multi-turn tool path is broken on TRL 1.13 + vLLM (imports the removed
  `trl.experimental.openenv`), generates one sequence at a time, and trains later turns without template tokens.
- Reward kwargs carry every dataset column; log reasons with `agenttune.utils.score_logger.log_score` (`AGENTTUNE_SCORE_LOG`).
- No persona / profile / transaction-history generator exists → we use NVIDIA Nemotron-Personas (`wtm/personas.py`).
- LoRA for cohere2: `target_modules="all-linear"` (peft has no default for Cohere2).

## AlignTune (post-training) — M3 DPO + alignment audit
- `create_rl_trainer(algorithm="dpo", backend="trl", use_peft=True, …)`; override defaults (`max_seq_length=512`, SFT `use_peft=False`).
- `aligntune compose` is a stub that fabricates metrics — chain stages by hand (SFT → merge → next stage on the merged path).
- Reward registry (52 names): only `json_schema_verifiable` / `regex_verifiable` (format) and `length`/`brevity` fit us, as minor terms.
- Alignment audit: `AlignmentAuditor(judge=None).score(adapter, probe_set=json.load(open("src/aligntune/recipes/probe_sets/default_alignment_probes.json")))`
  + `AlignmentDriftTracker.set_baseline(base).check_drift(tuned)`. Use a chat-template adapter (the built-ins send raw strings and
  the vLLM adapter turns T=0 into T=1). Its "reward hacking" score is a text-degeneracy heuristic — say so when reporting.

## AuditKIT (evaluation)
- Score our own transcripts with `model="precomputed"`; metrics `AskActCorrect`, `ToolCallF1(arg_mode="subset")` with `any_of`
  routes, `ToolPermission`, `LLMJudge`. Harness mode has no system prompt — skip it.
- Realism gap: give toy and realistic renderings the same `sample_id`, then `ak.compare(toy_run, real_run)` → paired deltas and the
  list of decisions that pass as toy and fail as realistic; confirm with our cluster-bootstrap analysis.
- Capability retention: `run_lmeval` (GSM8K, IFEval, MMLU/Global-MMLU), `load_bfcl` (prompt mode: irrelevance + single-turn).
- `push_to_hub` defaults to public — never call it before the user approves names and visibility.

## SafeTune (safety drift audit)
- `evaluate(model, tokenizer=tok, benchmarks=["harmbench","xstest"], judge="wildguard", max_prompts=…)` on base and each tuned model;
  WildGuard is gated (user must accept). Single user turn, no system prompt; `max_prompts` takes the first k rows.
- Harden trainers replace the SFT loop and do not plug into AgentTune/AlignTune/GRPO; ReSta recovery works on Tiny Aya with α≈0.25.
